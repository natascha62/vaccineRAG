"""Experiment runner: execute LLM calls for models x arms x questions x repetitions.

Design (from the research plan):

* **Resumable** - every call has a deterministic `call_id` built from question,
  model, arm, repetition, option order and prompt version. Re-running the same
  `run_id` skips calls that are already complete, so nothing is paid twice.
* **Append-only** - each attempt is appended to `calls.jsonl` as soon as it
  finishes; failed attempts are kept. Parse failures are retried once
  (configurable); a call that still fails is final and scored as incorrect.
  API errors are not final: they are retried on the next invocation.
* **Frozen settings** - the run folder stores a config snapshot. Continuing a
  run with result-affecting settings changed is refused.
* **Arm A3** - two steps per call: the model first selects up to `max_docs`
  documents from the catalog, then answers with those PDFs attached. The
  selection is stored in `selections.jsonl` and reused when a call is resumed,
  so a retry never silently changes the documents. Attached PDFs are ordered by
  `doc_id` (not by the model's ranking) so that requests sharing documents
  share a cacheable prompt prefix.
"""

from __future__ import annotations

import logging
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from vaxrag.config import Config, dump_config
from vaxrag.data.manifest import ManifestEntry
from vaxrag.io_utils import append_jsonl, git_commit, read_jsonl, read_models, utc_now, write_jsonl
from vaxrag.llm.base import LLMClient, LLMError, LLMRefusal, LLMRequest, PdfDocument
from vaxrag.output_parsing import parse_model_answer, parse_selection
from vaxrag.prompting import (
    answer_schema,
    build_answer_prompt,
    build_selection_prompt,
    selection_schema,
)
from vaxrag.prompts import load_prompt, load_prompt_parts
from vaxrag.schemas import (
    CallRecord,
    Chunk,
    DocumentSelection,
    Question,
    RetrievalRecord,
    document_label,
)

logger = logging.getLogger(__name__)

FINAL_STATUSES = {"ok", "refusal"}


@dataclass(frozen=True)
class CallSpec:
    """Everything that identifies one logical call (independent of attempts)."""

    question_id: str
    model: str
    arm: str
    rep: int
    prompt_version: str

    @property
    def call_id(self) -> str:
        """Deterministic identifier used for resuming."""
        return "|".join(
            [
                self.question_id,
                self.model,
                self.arm,
                f"r{self.rep}",
                self.prompt_version,
            ]
        )


@dataclass
class RunContext:
    """Static inputs shared by all calls of a run."""

    config: Config
    run_id: str
    run_dir: Path
    questions: dict[str, Question]
    chunks: dict[str, Chunk]
    retrieval: dict[str, RetrievalRecord] = field(default_factory=dict)
    gold: dict[str, list[str]] = field(default_factory=dict)
    commit: str = "unknown"
    documents: dict[str, ManifestEntry] = field(default_factory=dict)
    raw_dir: Path | None = None
    selections: dict[str, list[str]] = field(default_factory=dict)

    @property
    def calls_path(self) -> Path:
        """Path of this run's call log."""
        return self.run_dir / "calls.jsonl"

    @property
    def selections_path(self) -> Path:
        """Path of this run's A3 document-selection log."""
        return self.run_dir / "selections.jsonl"

    def catalog(self) -> list[tuple[str, str, str]]:
        """Document catalog for A3: (doc_id, 'Title (year)', doc_type), sorted by doc_id."""
        return [
            (doc_id, document_label(e.title, e.year), e.doc_type)
            for doc_id, e in sorted(self.documents.items())
        ]

    def load_selections(self) -> None:
        """Load successful document selections from earlier invocations."""
        for row in read_jsonl(self.selections_path):
            if row["status"] == "ok":
                self.selections[row["call_id"]] = row["doc_ids"]


def seeded_subset(question_ids: list[str], n: int, seed: int) -> set[str]:
    """Deterministic random subset of `n` question IDs (all of them if n >= len)."""
    ids = sorted(question_ids)
    if n >= len(ids):
        return set(ids)
    return set(random.Random(seed).sample(ids, n))


def select_questions(questions: list[Question], config: Config) -> list[Question]:
    """Apply the configured question filter (subsets, annotation status, sampling)."""
    f = config.experiment.question_filter
    selected = [q for q in questions if q.subset in f.subsets]
    if f.exclude_version_status:
        selected = [q for q in selected if q.version_status not in f.exclude_version_status]
    if f.only_kb_covered:
        selected = [q for q in selected if q.kb_covered]
    if f.sample_n is not None:
        keep = seeded_subset([q.question_id for q in selected], f.sample_n, f.seed)
        selected = [q for q in selected if q.question_id in keep]
    return selected


def plan_calls(questions: list[Question], models: list[str], config: Config) -> list[CallSpec]:
    """Enumerate all calls: the main pass plus repetitions for the consistency subset.

    Arm A3 is planned only for `fulldoc.models`; repetitions only for
    `consistency.arms`.
    """
    exp = config.experiment
    version = config.prompt.version
    ids = [q.question_id for q in questions]
    consistency = (
        seeded_subset(ids, exp.consistency.n_questions, exp.consistency.seed)
        if exp.consistency.enabled
        else set()
    )
    specs: list[CallSpec] = []
    for model in models:
        for arm in exp.arms:
            if model not in exp.arm_models(arm, models):
                continue
            repeat = arm in exp.consistency.arms
            for qid in ids:
                specs.append(CallSpec(qid, model, arm, 1, version))
                if repeat and qid in consistency:
                    for rep in range(2, exp.consistency.reps + 1):
                        specs.append(CallSpec(qid, model, arm, rep, version))
    return specs


def completed_call_ids(
    calls_path: Path, max_parse_attempts: int
) -> tuple[set[str], dict[str, int]]:
    """Return IDs of finished calls and the number of parse failures per unfinished call.

    A parse failure is final once `max_parse_attempts` failures were logged or a
    failure carries the last attempt number (an A3 document selection that
    failed all its attempts is logged as a single record).
    """
    done: set[str] = set()
    parse_failures: dict[str, int] = defaultdict(int)
    for row in read_jsonl(calls_path):
        if row["status"] in FINAL_STATUSES:
            done.add(row["call_id"])
        elif row["status"] == "parse_error":
            parse_failures[row["call_id"]] += 1
            if row["attempt"] >= max_parse_attempts:
                done.add(row["call_id"])
    for call_id, n in parse_failures.items():
        if n >= max_parse_attempts:
            done.add(call_id)
    return done, dict(parse_failures)


def context_for(spec: CallSpec, ctx: RunContext) -> list[Chunk] | None:
    """Return the chunks shown to the model for this arm (None for closed-book A0).

    Raises KeyError if retrieval or gold annotations are missing, so that a
    context arm never silently degrades into closed-book.
    """
    if spec.arm == "A0":
        return None
    if spec.arm == "A1":
        record = ctx.retrieval[spec.question_id]
        ids = [r.chunk_id for r in record.results[: ctx.config.retrieval.top_k]]
    elif spec.arm == "A2":
        ids = ctx.gold[spec.question_id]
    else:
        raise ValueError(f"Unknown arm {spec.arm}")
    return [ctx.chunks[cid] for cid in ids]


def select_documents(
    spec: CallSpec, client: LLMClient, ctx: RunContext
) -> tuple[list[str] | None, str, str | None]:
    """Run (or reuse) the A3 document selection for one call.

    Returns (doc_ids, status, error); doc_ids is None when selection failed.
    Every attempt is appended to `selections.jsonl`.
    """
    if spec.call_id in ctx.selections:
        return ctx.selections[spec.call_id], "ok", None
    cfg = ctx.config
    question = ctx.questions[spec.question_id]
    catalog = ctx.catalog()
    valid_ids = [doc_id for doc_id, _, _ in catalog]
    version = cfg.prompt.selection_version
    system_t, user_t = load_prompt(cfg.path("prompts_dir"), version)
    system, user = build_selection_prompt(
        question, catalog, cfg.experiment.fulldoc.max_docs, system_t, user_t
    )
    request = LLMRequest(
        system=system,
        user=user,
        json_schema=selection_schema(valid_ids),
        schema_name="doc_selection",
    )
    status, error = "parse_error", None
    for attempt in range(1, cfg.experiment.max_parse_attempts + 1):
        base = {
            "call_id": spec.call_id,
            "attempt": attempt,
            "question_id": spec.question_id,
            "model": spec.model,
            "prompt_version": version,
            "timestamp": utc_now(),
        }
        try:
            result = client.generate(request)
        except LLMRefusal as exc:
            append_jsonl(
                ctx.selections_path, DocumentSelection(**base, status="refusal", error=str(exc))
            )
            return None, "refusal", f"document selection refused: {exc}"
        except LLMError as exc:
            append_jsonl(
                ctx.selections_path, DocumentSelection(**base, status="api_error", error=str(exc))
            )
            return None, "api_error", f"document selection failed: {exc}"
        doc_ids, error = parse_selection(result.text, valid_ids, cfg.experiment.fulldoc.max_docs)
        status = "ok" if doc_ids else "parse_error"
        append_jsonl(
            ctx.selections_path,
            DocumentSelection(
                **base,
                status=status,
                doc_ids=doc_ids or [],
                raw_response=result.text,
                error=error,
                model_version=result.model_version,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=client.cost_usd(result),
            ),
        )
        if doc_ids:
            ctx.selections[spec.call_id] = doc_ids
            return doc_ids, "ok", None
    return None, "parse_error", f"document selection failed: {error}"


def load_pdf(ctx: RunContext, doc_id: str) -> PdfDocument:
    """Read one knowledge-base PDF for attachment (arm A3)."""
    entry = ctx.documents[doc_id]
    if ctx.raw_dir is None:
        raise ValueError("raw_dir is required for arm A3")
    return PdfDocument(
        title=document_label(entry.title, entry.year),
        filename=entry.filename,
        data=(ctx.raw_dir / entry.filename).read_bytes(),
    )


def execute_call(
    spec: CallSpec, client: LLMClient, ctx: RunContext, first_attempt: int
) -> CallRecord:
    """Run one call (with parse retries) and append every attempt to the log.

    Returns the last attempt's record.
    """
    question = ctx.questions[spec.question_id]
    system_t, user_t = load_prompt(ctx.config.path("prompts_dir"), spec.prompt_version)
    parts = load_prompt_parts(ctx.config.path("prompts_dir"), spec.prompt_version)
    letters = list(question.options)
    selected: list[str] = []
    pdfs: list[PdfDocument] = []
    if spec.arm == "A3":
        doc_ids, sel_status, sel_error = select_documents(spec, client, ctx)
        if doc_ids is None:
            record = CallRecord(
                call_id=spec.call_id,
                # A failed selection after all attempts is final, like a parse failure.
                attempt=ctx.config.experiment.max_parse_attempts,
                question_id=spec.question_id,
                model=spec.model,
                arm=spec.arm,
                rep=spec.rep,
                prompt_version=spec.prompt_version,
                git_commit=ctx.commit,
                run_id=ctx.run_id,
                timestamp=utc_now(),
                prompt_system="",
                prompt_user="",
                status=sel_status,  # type: ignore[arg-type]
                error=sel_error,
                generation_settings=client.generation_settings(),
            )
            append_jsonl(ctx.calls_path, record)
            return record
        selected = doc_ids
        ordered = sorted(doc_ids)
        pdfs = [load_pdf(ctx, d) for d in ordered]
        chunks = None
        system, user, source_map = build_answer_prompt(
            question,
            None,
            system_t,
            user_t,
            parts,
            documents=[(d, pdf.title) for d, pdf in zip(ordered, pdfs, strict=True)],
        )
    else:
        chunks = context_for(spec, ctx)
        system, user, source_map = build_answer_prompt(question, chunks, system_t, user_t, parts)
    request = LLMRequest(
        system=system,
        user=user,
        json_schema=answer_schema(letters),
        schema_name="mcq_answer",
        documents=pdfs,
        cache_documents=ctx.config.experiment.fulldoc.cache_documents,
    )
    base = {
        "call_id": spec.call_id,
        "question_id": spec.question_id,
        "model": spec.model,
        "arm": spec.arm,
        "rep": spec.rep,
        "prompt_version": spec.prompt_version,
        "git_commit": ctx.commit,
        "run_id": ctx.run_id,
        "prompt_system": system,
        "prompt_user": user,
        "context_chunk_ids": [c.chunk_id for c in chunks] if chunks else [],
        "context_labels": source_map,
        "selected_doc_ids": selected,
        "generation_settings": client.generation_settings(),
    }
    record: CallRecord | None = None
    for attempt in range(first_attempt, ctx.config.experiment.max_parse_attempts + 1):
        try:
            result = client.generate(request)
        except LLMRefusal as exc:
            r = exc.result
            record = CallRecord(
                **base,
                attempt=attempt,
                timestamp=utc_now(),
                status="refusal",
                error=str(exc),
                raw_response=r.text if r else None,
                model_version=r.model_version if r else None,
                input_tokens=r.input_tokens if r else None,
                output_tokens=r.output_tokens if r else None,
                finish_reason=r.finish_reason if r else None,
                cost_usd=client.cost_usd(r) if r else None,
            )
            append_jsonl(ctx.calls_path, record)
            return record
        except LLMError as exc:
            record = CallRecord(
                **base, attempt=attempt, timestamp=utc_now(), status="api_error", error=str(exc)
            )
            append_jsonl(ctx.calls_path, record)
            return record
        parsed, error = parse_model_answer(result.text, letters, question.multi_answer)
        record = CallRecord(
            **base,
            attempt=attempt,
            timestamp=utc_now(),
            model_version=result.model_version,
            raw_response=result.text,
            parsed=parsed,
            answer=parsed.answer if parsed else [],
            status="ok" if parsed else "parse_error",
            error=error,
            finish_reason=result.finish_reason,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            reasoning_tokens=result.reasoning_tokens,
            cache_read_tokens=result.cache_read_tokens,
            cache_write_tokens=result.cache_write_tokens,
            latency_s=result.latency_s,
            cost_usd=client.cost_usd(result),
        )
        append_jsonl(ctx.calls_path, record)
        if parsed is not None:
            return record
    assert record is not None
    return record


def prepare_run_dir(config: Config, run_id: str, allow_config_change: bool = False) -> Path:
    """Create the run folder and freeze (or check) its configuration snapshot."""
    run_dir = config.path("runs_dir") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    hash_path = run_dir / "config_hash.txt"
    new_hash = config.frozen_hash()
    if hash_path.exists():
        old_hash = hash_path.read_text().strip()
        if old_hash != new_hash and not allow_config_change:
            raise RuntimeError(
                f"Run '{run_id}' was started with different settings "
                f"(hash {old_hash} != {new_hash}). "
                "Use a new run_id, or pass --allow-config-change if this is intended."
            )
    hash_path.write_text(new_hash + "\n")
    dump_config(config, run_dir / "config.yaml")
    env_path = run_dir / "environment.txt"
    if not env_path.exists():
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
        ).stdout
        env_path.write_text(f"python {sys.version}\n{freeze}")
    return run_dir


def snapshot_retrieval(ctx: RunContext, question_ids: Iterable[str]) -> None:
    """Copy the retrieval records used by this run into the run folder."""
    path = ctx.run_dir / "retrieval.jsonl"
    if not path.exists():
        write_jsonl(path, [ctx.retrieval[q] for q in question_ids if q in ctx.retrieval])


def run_experiment(
    ctx: RunContext,
    clients: dict[str, LLMClient],
    questions: list[Question],
    dry_run: bool = False,
) -> dict[str, int]:
    """Execute all pending calls; returns counts of final statuses for this invocation."""
    exp = ctx.config.experiment
    specs = plan_calls(questions, list(clients), ctx.config)
    done, parse_failures = completed_call_ids(ctx.calls_path, exp.max_parse_attempts)
    pending = [s for s in specs if s.call_id not in done]
    skipped: list[CallSpec] = []
    runnable: list[CallSpec] = []
    for spec in pending:
        if spec.arm == "A1" and spec.question_id not in ctx.retrieval:
            skipped.append(spec)
        elif spec.arm == "A2" and not ctx.gold.get(spec.question_id):
            skipped.append(spec)
        elif spec.arm == "A3" and not ctx.documents:
            skipped.append(spec)
        else:
            runnable.append(spec)
    if skipped:
        logger.warning(
            "%d calls skipped: A1 needs retrieval results, A2 resolved gold chunks, "
            "A3 the document manifest",
            len(skipped),
        )
        write_jsonl(ctx.run_dir / "skipped_calls.jsonl", [{"call_id": s.call_id} for s in skipped])
    logger.info("%d planned, %d done, %d to run", len(specs), len(done), len(runnable))
    counts: dict[str, int] = defaultdict(int)
    if dry_run:
        counts["planned"] = len(runnable)
        return dict(counts)
    if "A1" in exp.arms:
        snapshot_retrieval(ctx, [q.question_id for q in questions])
    ctx.load_selections()
    by_model: dict[str, list[CallSpec]] = defaultdict(list)
    for spec in runnable:
        by_model[spec.model].append(spec)
    for model, model_specs in by_model.items():
        client = clients[model]
        workers = ctx.config.models[model].max_workers or exp.max_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(execute_call, s, client, ctx, parse_failures.get(s.call_id, 0) + 1)
                for s in model_specs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=model):
                counts[future.result().status] += 1
    return dict(counts)


def load_run_context(
    config: Config,
    run_id: str,
    questions: list[Question],
    chunks: list[Chunk],
    retrieval_path: Path | None,
    gold: dict[str, list[str]],
    allow_config_change: bool = False,
    documents: list[ManifestEntry] | None = None,
    raw_dir: Path | None = None,
) -> RunContext:
    """Assemble a `RunContext`, preparing the run folder on disk.

    `documents` (the manifest) and `raw_dir` are needed only for arm A3.
    """
    run_dir = prepare_run_dir(config, run_id, allow_config_change)
    retrieval = {}
    if retrieval_path is not None and retrieval_path.exists():
        retrieval = {r.question_id: r for r in read_models(retrieval_path, RetrievalRecord)}
    return RunContext(
        config=config,
        run_id=run_id,
        run_dir=run_dir,
        questions={q.question_id: q for q in questions},
        chunks={c.chunk_id: c for c in chunks},
        retrieval=retrieval,
        gold=gold,
        commit=git_commit(config.project_root),
        documents={e.doc_id: e for e in documents or []},
        raw_dir=raw_dir,
    )
