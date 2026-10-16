"""LLM-as-judge for grounding (do the model's quotes support its selected answer).

This measures faithfulness, not correctness: the judge sees the question, the
options, the selected answer and each quote together with the excerpt it cites,
but not the correct answer. Whether cited chunks support the *correct* answer
is judged separately (chunk relevance, see `evidence.py`). The judge model must
not be one of the evaluated models (checked in `Config`). Judgments are appended
to `grounding.jsonl` in the run folder and are resumable.

Validation: `export_human_sample` writes a blinded CSV for physician rating;
`cohens_kappa` compares the filled-in human labels with the judge.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from vaxrag.evaluation.citations import excerpt_around
from vaxrag.io_utils import append_jsonl, read_jsonl, utc_now
from vaxrag.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest
from vaxrag.output_parsing import extract_json_object
from vaxrag.prompts import load_prompt, render
from vaxrag.schemas import CallRecord, Chunk, Question

VERDICTS = ("supports", "partial", "not_supports")
GROUNDING_FILE = "grounding.jsonl"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
    "additionalProperties": False,
}


class Judgment(BaseModel):
    """One line of `grounding.jsonl`."""

    call_id: str
    judge_model: str
    judge_model_version: str | None = None
    prompt_version: str
    verdict: str | None
    rationale: str | None = None
    status: str
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    timestamp: str


def format_evidence(
    record: CallRecord, chunks: dict[str, Chunk], doc_texts: dict[str, str] | None = None
) -> str:
    """Render each quote with the excerpt it cites (or a note if the ID is unknown).

    For arm A3 the cited source is a whole document; the excerpt is the passage
    of its parsed text around the best match of the quote.
    """
    if record.parsed is None or not record.parsed.references:
        return "(no quotes given)"
    doc_texts = doc_texts or {}
    blocks = []
    for i, ref in enumerate(record.parsed.references, start=1):
        cid = record.context_labels.get(ref.source_id.strip().strip("[]"))
        if record.arm == "A3":
            text = doc_texts.get(cid or "")
            excerpt = (
                excerpt_around(ref.quote, text) or "(empty quote)"
                if text
                else "(cited ID does not exist in the prompt)"
            )
        else:
            excerpt = (
                chunks[cid].text if cid in chunks else "(cited ID does not exist in the prompt)"
            )
        blocks.append(
            f'Quote {i} (cites {ref.source_id}): "{ref.quote}"\nCited excerpt:\n{excerpt}'
        )
    return "\n\n".join(blocks)


def build_judge_request(
    record: CallRecord,
    question: Question,
    chunks: dict[str, Chunk],
    system_t: str,
    user_t: str,
    doc_texts: dict[str, str] | None = None,
) -> LLMRequest:
    """Build the grounding-judge request for one answer."""
    options = "\n".join(f"{k}. {v}" for k, v in question.options.items())
    selected = ", ".join(record.parsed.answer) if record.parsed else "(none)"
    user = render(
        user_t,
        stem=question.stem,
        options=options,
        selected=selected,
        evidence=format_evidence(record, chunks, doc_texts),
    )
    return LLMRequest(system=system_t, user=user, json_schema=JUDGE_SCHEMA, schema_name="judgment")


def judge_one(
    record: CallRecord,
    question: Question,
    chunks: dict[str, Chunk],
    client: LLMClient,
    system_t: str,
    user_t: str,
    prompt_version: str,
    doc_texts: dict[str, str] | None = None,
) -> Judgment:
    """Judge one answer."""
    base = {"call_id": record.call_id, "judge_model": client.key, "prompt_version": prompt_version}
    try:
        result = client.generate(
            build_judge_request(record, question, chunks, system_t, user_t, doc_texts)
        )
    except (LLMError, LLMRefusal) as exc:
        return Judgment(**base, verdict=None, status="error", error=str(exc), timestamp=utc_now())
    payload = extract_json_object(result.text)
    verdict, rationale, status = None, None, "parse_error"
    if payload:
        try:
            data = json.loads(payload)
            if data.get("verdict") in VERDICTS:
                verdict, rationale, status = data["verdict"], data.get("rationale"), "ok"
        except json.JSONDecodeError:
            pass
    return Judgment(
        **base,
        judge_model_version=result.model_version,
        verdict=verdict,
        rationale=rationale,
        status=status,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=client.cost_usd(result),
        timestamp=utc_now(),
    )


def run_judge(
    records: list[CallRecord],
    questions: dict[str, Question],
    chunks: dict[str, Chunk],
    client: LLMClient,
    prompts_dir: str | Path,
    prompt_version: str,
    output_path: str | Path,
    max_workers: int = 4,
    doc_texts: dict[str, str] | None = None,
) -> int:
    """Judge all eligible records not yet judged successfully; returns the number judged.

    `doc_texts` (doc_id -> parsed text) is needed for A3 answers.
    """
    done = {row["call_id"] for row in read_jsonl(output_path) if row["status"] == "ok"}
    todo = [
        r for r in records if r.call_id not in done and r.status == "ok" and r.parsed is not None
    ]
    system_t, user_t = load_prompt(prompts_dir, prompt_version)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                judge_one,
                r,
                questions[r.question_id],
                chunks,
                client,
                system_t,
                user_t,
                prompt_version,
                doc_texts,
            )
            for r in todo
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="judge"):
            append_jsonl(output_path, future.result())
    return len(todo)


def export_human_sample(
    records: list[CallRecord],
    questions: dict[str, Question],
    chunks: dict[str, Chunk],
    n: int,
    seed: int,
    output_csv: str | Path,
    doc_texts: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Write a blinded random sample for physician rating (model and arm hidden)."""
    eligible = sorted(
        [r for r in records if r.status == "ok" and r.parsed is not None], key=lambda r: r.call_id
    )
    sample = random.Random(seed).sample(eligible, min(n, len(eligible)))
    rows = []
    for i, r in enumerate(sample, start=1):
        q = questions[r.question_id]
        rows.append(
            {
                "item": i,
                "call_id": r.call_id,
                "question": q.stem,
                "options": "\n".join(f"{k}. {v}" for k, v in q.options.items()),
                "selected_answer": ", ".join(r.parsed.answer),
                "evidence": format_evidence(r, chunks, doc_texts),
                "rater_verdict": "",
                "rater_comment": "",
            }
        )
    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def cohens_kappa(a: list[str], b: list[str], labels: tuple[str, ...] | None = None) -> float:
    """Unweighted Cohen's kappa for two raters over the same items.

    `labels` defaults to every label used by either rater.
    """
    if len(a) != len(b) or not a:
        raise ValueError("Ratings must be non-empty and of equal length")
    labels = labels or tuple(sorted(set(a) | set(b)))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
