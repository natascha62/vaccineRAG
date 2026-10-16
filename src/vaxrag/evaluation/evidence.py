"""Gold-independent evidence judgments with an LLM judge.

Gold passages are annotated as *one sufficient set of evidence*, not as every
chunk that could answer a question. Metrics that compare against gold chunks
are therefore lower bounds ("strict"). This module provides two judgments that
do not depend on exhaustive gold annotation:

* **chunk relevance** - does a single chunk support the *correct* answer?
  (`supports` / `partial` / `not_relevant`). Used for citation support
  (are the chunks a model cites actually supportive?) and for pooled relevance
  (TREC-style: every chunk in the union of top-k retrieval results and model
  citations is judged, giving a "lenient" relevant set = gold + judged chunks).
* **context sufficiency** - does a *set* of chunks contain enough information
  to derive the correct answer? (`sufficient` / `partial` / `insufficient`).
  Used to attribute A1 errors to retrieval vs. generation without gold chunks,
  and as a quality check of the annotated gold sets (A2 contexts).

Unlike the grounding judge (`judge.py`), these judges see the correct answer.
Judgments are cached in `results/judgments/<kind>.jsonl`, keyed by question and
chunk IDs, and filtered by judge model and prompt version when loaded, so they
are computed once and shared by all runs and retrieval configurations.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from vaxrag.evaluation.citations import cited_chunk_ids
from vaxrag.io_utils import append_jsonl, read_jsonl, stable_hash, utc_now
from vaxrag.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest
from vaxrag.output_parsing import extract_json_object
from vaxrag.prompts import load_prompt, render
from vaxrag.schemas import CallRecord, Chunk, Question, RetrievalRecord

EvidenceKind = Literal["chunk_relevance", "context_sufficiency"]

RELEVANCE_LABELS = ("supports", "partial", "not_relevant")
SUFFICIENCY_LABELS = ("sufficient", "partial", "insufficient")
LABELS: dict[str, tuple[str, ...]] = {
    "chunk_relevance": RELEVANCE_LABELS,
    "context_sufficiency": SUFFICIENCY_LABELS,
}


def verdict_schema(kind: EvidenceKind) -> dict:
    """JSON schema of the judge output for one judgment kind."""
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(LABELS[kind])},
            "rationale": {"type": "string"},
        },
        "required": ["verdict", "rationale"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class EvidenceTask:
    """One item to judge: a question and one (relevance) or several (sufficiency) chunks."""

    question_id: str
    chunk_ids: tuple[str, ...]

    def key(self, kind: EvidenceKind) -> str:
        """Cache key; chunk order is irrelevant for both judgment kinds."""
        if kind == "chunk_relevance":
            return f"{self.question_id}|{self.chunk_ids[0]}"
        return f"{self.question_id}|{stable_hash(sorted(self.chunk_ids))}"


def relevance_key(question_id: str, chunk_id: str) -> str:
    """Cache key of a chunk-relevance judgment."""
    return EvidenceTask(question_id, (chunk_id,)).key("chunk_relevance")


def sufficiency_key(question_id: str, chunk_ids: Iterable[str]) -> str:
    """Cache key of a context-sufficiency judgment."""
    return EvidenceTask(question_id, tuple(chunk_ids)).key("context_sufficiency")


class EvidenceJudgment(BaseModel):
    """One line of `results/judgments/<kind>.jsonl`."""

    key: str
    kind: EvidenceKind
    question_id: str
    chunk_ids: list[str]
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
    timestamp: str = Field(default_factory=utc_now)


def format_question_with_answer(question: Question) -> tuple[str, str]:
    """Return (options block, correct-answer block) in the original option order."""
    options = "\n".join(f"{k}. {v}" for k, v in question.options.items())
    correct = "\n".join(f"{k}. {question.options[k]}" for k in question.correct)
    return options, correct


def format_passages(chunk_ids: Iterable[str], chunks: dict[str, Chunk]) -> str:
    """Render chunks as numbered passages [P1], [P2], ..."""
    return "\n\n".join(f"[P{i}] {chunks[cid].text}" for i, cid in enumerate(chunk_ids, start=1))


def build_evidence_request(
    kind: EvidenceKind,
    task: EvidenceTask,
    question: Question,
    chunks: dict[str, Chunk],
    system_t: str,
    user_t: str,
) -> LLMRequest:
    """Build the judge request for one task."""
    options, correct = format_question_with_answer(question)
    user = render(
        user_t,
        stem=question.stem,
        options=options,
        correct=correct,
        answer_mode="all of the listed options" if len(question.correct) > 1 else "this option",
        passages=format_passages(task.chunk_ids, chunks),
    )
    return LLMRequest(
        system=system_t, user=user, json_schema=verdict_schema(kind), schema_name=kind
    )


def judge_evidence_one(
    kind: EvidenceKind,
    task: EvidenceTask,
    question: Question,
    chunks: dict[str, Chunk],
    client: LLMClient,
    system_t: str,
    user_t: str,
    prompt_version: str,
) -> EvidenceJudgment:
    """Run one evidence judgment."""
    base = {
        "key": task.key(kind),
        "kind": kind,
        "question_id": task.question_id,
        "chunk_ids": list(task.chunk_ids),
        "judge_model": client.key,
        "prompt_version": prompt_version,
    }
    try:
        result = client.generate(
            build_evidence_request(kind, task, question, chunks, system_t, user_t)
        )
    except (LLMError, LLMRefusal) as exc:
        return EvidenceJudgment(**base, verdict=None, status="error", error=str(exc))
    verdict, rationale, status = None, None, "parse_error"
    payload = extract_json_object(result.text)
    if payload:
        try:
            data = json.loads(payload)
            if data.get("verdict") in LABELS[kind]:
                verdict, rationale, status = data["verdict"], data.get("rationale"), "ok"
        except json.JSONDecodeError:
            pass
    return EvidenceJudgment(
        **base,
        judge_model_version=result.model_version,
        verdict=verdict,
        rationale=rationale,
        status=status,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=client.cost_usd(result),
    )


def load_evidence_judgments(
    path: str | Path, judge_model: str, prompt_version: str
) -> dict[str, EvidenceJudgment]:
    """Latest successful judgment per key for one judge model and prompt version."""
    latest: dict[str, EvidenceJudgment] = {}
    for row in read_jsonl(path):
        if (
            row.get("status") == "ok"
            and row.get("judge_model") == judge_model
            and row.get("prompt_version") == prompt_version
        ):
            latest[row["key"]] = EvidenceJudgment.model_validate(row)
    return latest


def load_evidence_verdicts(
    path: str | Path, judge_model: str, prompt_version: str
) -> dict[str, str]:
    """Map cache key -> verdict (see `load_evidence_judgments`)."""
    return {
        k: j.verdict
        for k, j in load_evidence_judgments(path, judge_model, prompt_version).items()
        if j.verdict
    }


def run_evidence_judgments(
    kind: EvidenceKind,
    tasks: Iterable[EvidenceTask],
    questions: dict[str, Question],
    chunks: dict[str, Chunk],
    client: LLMClient,
    prompts_dir: str | Path,
    prompt_version: str,
    output_path: str | Path,
    max_workers: int = 4,
) -> int:
    """Judge every task not yet cached (for this judge and prompt); returns the number judged."""
    done = set(load_evidence_judgments(output_path, client.key, prompt_version))
    todo: dict[str, EvidenceTask] = {}
    for task in tasks:
        key = task.key(kind)
        if key not in done and key not in todo and all(c in chunks for c in task.chunk_ids):
            todo[key] = task
    system_t, user_t = load_prompt(prompts_dir, prompt_version)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                judge_evidence_one,
                kind,
                t,
                questions[t.question_id],
                chunks,
                client,
                system_t,
                user_t,
                prompt_version,
            )
            for t in todo.values()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=kind):
            append_jsonl(output_path, future.result())
    return len(todo)


# --------------------------------------------------------------------------- task builders
def sufficiency_tasks(
    question_ids: Iterable[str],
    retrieval: dict[str, RetrievalRecord],
    top_k: int,
    gold: dict[str, list[str]] | None = None,
) -> list[EvidenceTask]:
    """Sufficiency tasks for the A1 contexts (top-k retrieval) and optionally the gold sets."""
    tasks: list[EvidenceTask] = []
    for qid in question_ids:
        if qid in retrieval:
            ids = tuple(r.chunk_id for r in retrieval[qid].results[:top_k])
            if ids:
                tasks.append(EvidenceTask(qid, ids))
        if gold and gold.get(qid):
            tasks.append(EvidenceTask(qid, tuple(gold[qid])))
    return tasks


def relevance_pool(
    question_ids: Iterable[str],
    retrieval: dict[str, RetrievalRecord],
    pool_k: int,
    gold: dict[str, list[str]] | None = None,
    records: Iterable[CallRecord] = (),
) -> list[EvidenceTask]:
    """Pooled chunk-relevance tasks.

    The pool of a question is the union of its top-`pool_k` retrieved chunks,
    its gold chunks (judged as an annotation quality check) and every chunk
    cited by a model answer in `records`.
    """
    selected = set(question_ids)
    pool: dict[str, list[str]] = {qid: [] for qid in question_ids}

    def add(qid: str, cid: str) -> None:
        if qid in pool and cid not in pool[qid]:
            pool[qid].append(cid)

    for qid in pool:
        if qid in retrieval:
            for r in retrieval[qid].results[:pool_k]:
                add(qid, r.chunk_id)
        for cid in (gold or {}).get(qid, []):
            add(qid, cid)
    for record in records:
        if record.question_id in selected and record.parsed is not None:
            for cid in cited_chunk_ids(record.parsed.references, record.context_labels):
                add(record.question_id, cid)
    return [EvidenceTask(qid, (cid,)) for qid, cids in pool.items() for cid in cids]


# --------------------------------------------------------------------------- derived sets
def lenient_relevant_sets(
    gold: dict[str, list[str]],
    relevance: dict[str, str],
    positive: Iterable[str] = ("supports",),
) -> dict[str, set[str]]:
    """Relevant chunks per question: annotated gold plus chunks judged relevant.

    Human annotation takes precedence: a gold chunk stays relevant even if the
    judge disagrees.
    """
    positive = set(positive)
    sets: dict[str, set[str]] = {qid: set(ids) for qid, ids in gold.items() if ids}
    for key, verdict in relevance.items():
        if verdict in positive:
            qid, cid = key.split("|", 1)
            sets.setdefault(qid, set()).add(cid)
    return sets


def judged_membership(
    question_id: str,
    chunk_ids: list[str],
    relevant: dict[str, set[str]],
    relevance: dict[str, str],
) -> bool | None:
    """Whether any chunk is relevant under the lenient definition.

    Returns None when the answer is unknown: no chunk is known to be relevant
    and at least one chunk has never been judged (it might be relevant).
    """
    known = relevant.get(question_id, set())
    if any(c in known for c in chunk_ids):
        return True
    if all(relevance_key(question_id, c) in relevance for c in chunk_ids):
        return False
    return None


def judged_citation_support(
    question_id: str,
    cited: list[str],
    relevant: dict[str, set[str]],
    relevance: dict[str, str],
) -> dict[str, float | bool | None]:
    """Citation support using lenient relevance.

    * `citation_support_any_judged` - at least one cited chunk supports the
      correct answer (False if nothing is cited; None if undeterminable).
    * `citation_precision_judged` - share of cited chunks that support it
      (None unless every cited chunk is known to be relevant or judged).
    """
    if not cited:
        return {"citation_support_any_judged": False, "citation_precision_judged": None}
    known = relevant.get(question_id, set())
    status = [
        True if c in known else (False if relevance_key(question_id, c) in relevance else None)
        for c in cited
    ]
    any_support = (
        True
        if any(s is True for s in status)
        else (False if all(s is False for s in status) else None)
    )
    precision = None if any(s is None for s in status) else sum(status) / len(status)
    return {"citation_support_any_judged": any_support, "citation_precision_judged": precision}


# --------------------------------------------------------------------------- human validation
def export_evidence_sample(
    kind: EvidenceKind,
    judgments: dict[str, EvidenceJudgment],
    questions: dict[str, Question],
    chunks: dict[str, Chunk],
    n: int,
    seed: int,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Write a blinded random sample of evidence judgments for physician rating.

    Raters see the question, options, correct answer and passage(s), but not
    the judge's verdict; they fill `rater_verdict` with one of `LABELS[kind]`.
    """
    eligible = sorted(
        (j for j in judgments.values() if all(c in chunks for c in j.chunk_ids)),
        key=lambda j: j.key,
    )
    sample = random.Random(seed).sample(eligible, min(n, len(eligible)))
    rows = []
    for i, j in enumerate(sample, start=1):
        q = questions[j.question_id]
        options, correct = format_question_with_answer(q)
        rows.append(
            {
                "item": i,
                "key": j.key,
                "question": q.stem,
                "options": options,
                "correct_answer": correct,
                "passages": format_passages(j.chunk_ids, chunks),
                "allowed_verdicts": "/".join(LABELS[kind]),
                "rater_verdict": "",
                "rater_comment": "",
            }
        )
    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
