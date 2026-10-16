"""Command-line interface: `vaxrag <command> [options]`.

Pipeline order (see README):

    prepare-questions -> manifest-hash -> parse -> [table-vision] -> chunk
    -> resolve-gold -> retrieve -> eval-retrieval -> run -> judge
    -> judge-evidence -> eval-retrieval (lenient) -> collect -> [table-review
    -> collect] -> analyze

Every command accepts one or more `--config` files (deep-merged in order).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from vaxrag.config import Config, load_config
from vaxrag.io_utils import read_jsonl, read_models, write_jsonl
from vaxrag.schemas import Chunk, Question

logger = logging.getLogger("vaxrag")


# --------------------------------------------------------------------------- helpers
def load_questions(config: Config) -> list[Question]:
    """Load prepared questions and apply the latest manual annotations."""
    from vaxrag.data.annotations import apply_annotations

    path = config.path("questions_jsonl")
    if not path.exists():
        sys.exit(f"{path} not found - run `vaxrag prepare-questions` first")
    return apply_annotations(read_models(path, Question), config.path("annotations"))


def load_chunks(config: Config) -> list[Chunk]:
    """Load the chunked knowledge base."""
    path = config.path("chunks")
    if not path.exists():
        sys.exit(f"{path} not found - run `vaxrag chunk` first")
    return read_models(path, Chunk)


def load_gold(config: Config) -> dict[str, list[str]]:
    """Load resolved gold chunk IDs per question (empty if not resolved yet)."""
    return {
        row["question_id"]: row["chunk_ids"]
        for row in read_jsonl(config.path("gold_chunks"))
        if row["chunk_ids"]
    }


def load_doc_texts(config: Config) -> dict[str, str]:
    """Load the parsed full text (Docling Markdown) per doc_id, used to check A3 quotes."""
    texts: dict[str, str] = {}
    for path in sorted(config.path("parsed_dir").glob("*.md")):
        texts[path.stem] = path.read_text(encoding="utf-8")
    return texts


def manifest_titles(config: Config) -> dict[str, str]:
    """doc_id -> title from the manifest (empty if no manifest yet)."""
    from vaxrag.data.manifest import load_manifest

    path = config.path("manifest")
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return {e.doc_id: e.title for e in load_manifest(path)}


def run_dirs(config: Config, run_ids: list[str]) -> list[Path]:
    """Resolve run IDs to existing run folders."""
    dirs = [config.path("runs_dir") / r for r in run_ids]
    missing = [str(d) for d in dirs if not d.exists()]
    if missing:
        sys.exit(f"Run folders not found: {missing}")
    return dirs


EVIDENCE_KINDS = {"relevance": "chunk_relevance", "sufficiency": "context_sufficiency"}


def evidence_path(config: Config, kind: str) -> Path:
    """Cache file of one evidence-judgment kind ('chunk_relevance' or 'context_sufficiency')."""
    return config.path("judgments_dir") / f"{kind}.jsonl"


def evidence_version(config: Config, kind: str) -> str:
    """Prompt version used for one evidence-judgment kind."""
    if kind == "chunk_relevance":
        return config.prompt.relevance_version
    return config.prompt.sufficiency_version


def load_evidence(config: Config, kind: str) -> dict[str, str]:
    """Verdicts of one evidence kind for the configured judge model and prompt version."""
    from vaxrag.evaluation.evidence import load_evidence_verdicts

    return load_evidence_verdicts(
        evidence_path(config, kind), config.judge.model, evidence_version(config, kind)
    )


def load_retrieval_map(config: Config, chunks: list[Chunk]) -> dict:
    """Load retrieval records of the configured settings, keyed by question ID."""
    from vaxrag.retrieval.hybrid import retrieval_path
    from vaxrag.schemas import RetrievalRecord

    path = retrieval_path(config.path("retrieval_dir"), config.retrieval, chunks)
    if not path.exists():
        sys.exit(f"{path} not found - run `vaxrag retrieve` first")
    return {r.question_id: r for r in read_models(path, RetrievalRecord)}


def final_ok_records(run_dir: Path, arms: list[str]) -> list:
    """Return the final successful call records of a run for the given arms."""
    from vaxrag.evaluation.scoring import final_calls
    from vaxrag.schemas import CallRecord

    records = [r for r in read_models(run_dir / "calls.jsonl", CallRecord) if r.arm in arms]
    if not records:
        return []
    attempts = pd.DataFrame(
        [{"call_id": r.call_id, "attempt": r.attempt, "status": r.status} for r in records]
    )
    final = final_calls(attempts)
    keep = set(zip(final["call_id"], final["attempt"], strict=True))
    return [r for r in records if (r.call_id, r.attempt) in keep and r.status == "ok"]


# --------------------------------------------------------------------------- commands
def cmd_prepare_questions(config: Config, args: argparse.Namespace) -> None:
    """Convert the xlsx question bank to JSONL and create annotation templates."""
    from vaxrag.data.annotations import create_annotation_templates
    from vaxrag.data.questions import exclusions_frame, load_question_bank

    questions, exclusions = load_question_bank(
        config.path("questions_dir"), config.questions.files, config.questions.multi_answer_subsets
    )
    n = write_jsonl(config.path("questions_jsonl"), questions)
    excl_path = config.path("question_exclusions")
    excl_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_frame(exclusions).to_csv(excl_path, index=False)
    added, _ = create_annotation_templates(
        questions, config.path("annotations"), config.path("gold_passages")
    )
    print(f"Wrote {n} questions to {config.path('questions_jsonl')}")
    print(f"Excluded {len(exclusions)} rows (see {excl_path})")
    print(f"Added {added} rows to {config.path('annotations')} (consensus file)")
    for name in config.annotation.annotators:
        folder = config.path("annotators_dir") / name
        added, _ = create_annotation_templates(
            questions, folder / "question_annotations.csv", folder / "gold_passages.csv"
        )
        print(f"Added {added} rows to {folder / 'question_annotations.csv'} (annotator {name})")
    summary = (
        pd.DataFrame([q.model_dump() for q in questions])
        .groupby("subset")
        .agg(
            n=("question_id", "count"),
            multi_correct=("correct", lambda s: sum(len(x) > 1 for x in s)),
        )
    )
    print(summary.to_string())


def cmd_manifest_hash(config: Config, args: argparse.Namespace) -> None:
    """Create the manifest template if needed, fill missing hashes and verify files."""
    from vaxrag.data.manifest import (
        create_manifest_template,
        fill_hashes,
        load_manifest,
        verify_manifest,
    )

    path = config.path("manifest")
    if create_manifest_template(path):
        print(f"Created empty manifest template at {path}; add one row per document.")
        return
    filled = fill_hashes(path, config.path("guidelines_raw"))
    problems = verify_manifest(load_manifest(path), config.path("guidelines_raw"))
    print(f"Filled {filled} hashes.")
    print("\n".join(problems) if problems else "All files match the manifest.")


def cmd_parse(config: Config, args: argparse.Namespace) -> None:
    """Parse every manifest document with Docling and write a table-quality report."""
    from vaxrag.data.manifest import load_manifest, verify_manifest
    from vaxrag.kb.parsing import create_converter, load_blocks, parse_document, table_quality

    entries = load_manifest(config.path("manifest"))
    problems = verify_manifest(entries, config.path("guidelines_raw"))
    if problems:
        sys.exit("Manifest problems:\n" + "\n".join(problems))
    converter = None
    report = []
    for entry in entries:
        out = config.path("parsed_dir") / f"{entry.doc_id}.blocks.jsonl"
        if out.exists() and not args.force:
            blocks = load_blocks(config.path("parsed_dir"), entry.doc_id)
        else:
            converter = converter or create_converter()
            print(f"Parsing {entry.doc_id} ...")
            blocks = parse_document(
                entry, config.path("guidelines_raw"), config.path("parsed_dir"), converter
            )
        report.extend(table_quality(blocks))
    report_path = config.path("parsed_dir") / "table_quality.csv"
    pd.DataFrame(report).to_csv(report_path, index=False)
    n_suspect = sum(r["suspect"] for r in report)
    print(f"{len(report)} tables, {n_suspect} flagged as suspect -> {report_path}")


def cmd_table_vision(config: Config, args: argparse.Namespace) -> None:
    """Convert listed pages to Markdown tables with the configured multimodal model."""
    from vaxrag.data.manifest import load_manifest
    from vaxrag.kb.table_vision import convert_pages
    from vaxrag.llm.factory import create_client

    model_key = args.model or config.table_vision_model
    if not model_key:
        sys.exit("Set `table_vision_model` in the config or pass --model")
    pages_csv = config.path("table_vision_pages")
    if not pages_csv.exists():
        pd.DataFrame(columns=["doc_id", "page", "notes"]).to_csv(pages_csv, index=False)
        sys.exit(f"Created {pages_csv}; list the pages to convert and run again.")
    filenames = {e.doc_id: e.filename for e in load_manifest(config.path("manifest"))}
    written = convert_pages(
        pages_csv,
        filenames,
        config.path("guidelines_raw"),
        config.path("table_images_dir"),
        config.path("table_overrides_dir"),
        create_client(config, model_key),
        config.path("prompts_dir"),
        config.prompt.table_vision_version,
        overwrite=args.overwrite,
    )
    print(
        f"Wrote {len(written)} override files. Check them manually before running `vaxrag chunk`."
    )


def cmd_chunk(config: Config, args: argparse.Namespace) -> None:
    """Chunk all parsed documents (applying vision table overrides)."""
    from vaxrag.data.manifest import load_manifest
    from vaxrag.kb.chunking import TokenCounter, chunk_document
    from vaxrag.kb.parsing import apply_table_overrides, load_blocks

    count = TokenCounter(config.chunking.tokenizer)
    chunks: list[Chunk] = []
    for entry in load_manifest(config.path("manifest")):
        blocks = apply_table_overrides(
            load_blocks(config.path("parsed_dir"), entry.doc_id), config.path("table_overrides_dir")
        )
        chunks.extend(
            chunk_document(
                blocks, entry.doc_id, entry.title, config.chunking, count, doc_year=entry.year
            )
        )
    write_jsonl(config.path("chunks"), chunks)
    df = pd.DataFrame(
        [{"doc_id": c.doc_id, "kind": c.kind, "n_tokens": c.n_tokens} for c in chunks]
    )
    print(f"Wrote {len(chunks)} chunks to {config.path('chunks')}")
    if not df.empty:
        print(
            df.groupby(["doc_id", "kind"])["n_tokens"].describe()[["count", "mean", "min", "max"]]
        )


def cmd_resolve_gold(config: Config, args: argparse.Namespace) -> None:
    """Map annotated gold passages to chunk IDs."""
    from vaxrag.data.annotations import resolve_gold_chunks

    resolutions = resolve_gold_chunks(config.path("gold_passages"), load_chunks(config))
    write_jsonl(config.path("gold_chunks"), resolutions)
    unresolved = [r for r in resolutions if r.unresolved]
    ambiguous = [r for r in resolutions if r.ambiguous]
    print(f"{len(resolutions)} questions with gold annotations -> {config.path('gold_chunks')}")
    print(f"{len(unresolved)} with unresolved rows, {len(ambiguous)} ambiguous (>3 chunks per row)")
    for r in (unresolved + ambiguous)[:20]:
        print(f"  check {r.question_id}")


def cmd_retrieve(config: Config, args: argparse.Namespace) -> None:
    """Run hybrid retrieval for all questions and cache the results."""
    from vaxrag.retrieval.hybrid import HybridRetriever, retrieval_path, retrieve_all

    chunks = load_chunks(config)
    retriever = HybridRetriever(chunks, config.retrieval, config.path("index_dir"))
    out = retrieval_path(config.path("retrieval_dir"), config.retrieval, chunks)
    records = retrieve_all(load_questions(config), retriever, out)
    print(f"Retrieval results for {len(records)} questions -> {out}")


def cmd_eval_retrieval(config: Config, args: argparse.Namespace) -> None:
    """Compute Recall@k, MRR and document hit rate (strict and, if judged, lenient)."""
    from vaxrag.evaluation.evidence import lenient_relevant_sets
    from vaxrag.retrieval.evaluate import DEFAULT_KS, evaluate_retrieval, summarize_retrieval

    chunks = load_chunks(config)
    records = list(load_retrieval_map(config, chunks).values())
    chunk_doc = {c.chunk_id: c.doc_id for c in chunks}
    questions = load_questions(config)
    gold = load_gold(config)
    frames = [evaluate_retrieval(records, gold, chunk_doc, questions, relevance_label="strict")]
    relevance = load_evidence(config, "chunk_relevance")
    if relevance:
        lenient = lenient_relevant_sets(gold, relevance, config.judge.positive_relevance)
        ks = tuple(k for k in DEFAULT_KS if k <= config.judge.pool_k)
        frames.append(
            evaluate_retrieval(
                records,
                {q: sorted(ids) for q, ids in lenient.items()},
                chunk_doc,
                questions,
                ks=ks,
                relevance_label="lenient",
            )
        )
    per_q = pd.concat(frames, ignore_index=True)
    out_dir = config.path("analysis_dir") / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = config.retrieval.config_hash()
    per_q.to_csv(out_dir / f"retrieval_per_question_{tag}.csv", index=False)
    summary = summarize_retrieval(per_q)
    summary.to_csv(out_dir / f"retrieval_summary_{tag}.csv", index=False)
    print(
        summary.to_string(index=False)
        if not summary.empty
        else "No questions with gold chunks or relevance judgments yet."
    )
    if not relevance:
        print("Only strict metrics: run `vaxrag judge-evidence --kind relevance` for lenient ones.")


def cmd_run(config: Config, args: argparse.Namespace) -> None:
    """Run (or resume) the LLM experiment."""
    from vaxrag.llm.factory import create_client
    from vaxrag.retrieval.hybrid import retrieval_path
    from vaxrag.runner import load_run_context, run_experiment, select_questions

    models = args.models or config.experiment.models
    if not models:
        sys.exit("No models selected (set experiment.models or pass --models)")
    unknown = [m for m in models if m not in config.models]
    if unknown:
        sys.exit(f"Unknown models: {unknown}")
    try:
        config.check_training_cutoffs(models)
    except ValueError as exc:
        sys.exit(str(exc))
    questions = select_questions(load_questions(config), config)
    needs_chunks = any(a in config.experiment.arms for a in ("A1", "A2"))
    chunks = load_chunks(config) if needs_chunks else []
    documents = None
    if "A3" in config.experiment.arms and config.experiment.arm_models("A3", models):
        from vaxrag.data.manifest import load_manifest, verify_manifest

        documents = load_manifest(config.path("manifest"))
        problems = verify_manifest(documents, config.path("guidelines_raw"))
        if problems:
            sys.exit("Manifest problems (needed for arm A3):\n" + "\n".join(problems))
    ctx = load_run_context(
        config,
        args.run_id,
        questions,
        chunks,
        retrieval_path(config.path("retrieval_dir"), config.retrieval, chunks)
        if needs_chunks
        else None,
        load_gold(config),
        allow_config_change=args.allow_config_change,
        documents=documents,
        raw_dir=config.path("guidelines_raw"),
    )
    clients = {m: create_client(config, m) for m in models}
    counts = run_experiment(ctx, clients, questions, dry_run=args.dry_run)
    print(f"Run {args.run_id}: {counts}")


def cmd_judge(config: Config, args: argparse.Namespace) -> None:
    """Run the grounding judge (do A1/A2 quotes support the selected answer)."""
    from vaxrag.evaluation.judge import GROUNDING_FILE, run_judge
    from vaxrag.llm.factory import create_client

    client = create_client(config, config.judge.model)
    questions = {q.question_id: q for q in load_questions(config)}
    chunks = {c.chunk_id: c for c in load_chunks(config)}
    doc_texts = load_doc_texts(config)
    for run_dir in run_dirs(config, args.run_ids):
        n = run_judge(
            final_ok_records(run_dir, config.judge.arms),
            questions,
            chunks,
            client,
            config.path("prompts_dir"),
            config.prompt.grounding_version,
            run_dir / GROUNDING_FILE,
            config.judge.max_workers,
            doc_texts=doc_texts,
        )
        print(f"{run_dir.name}: judged {n} answers")


def cmd_judge_evidence(config: Config, args: argparse.Namespace) -> None:
    """Chunk-relevance and/or context-sufficiency judgments (cached, shared by all runs)."""
    from vaxrag.evaluation.evidence import relevance_pool, run_evidence_judgments, sufficiency_tasks
    from vaxrag.llm.factory import create_client
    from vaxrag.runner import select_questions

    client = create_client(config, config.judge.model)
    chunk_list = load_chunks(config)
    chunks = {c.chunk_id: c for c in chunk_list}
    questions = select_questions(load_questions(config), config)
    by_id = {q.question_id: q for q in questions}
    retrieval = load_retrieval_map(config, chunk_list)
    gold = load_gold(config)
    kinds = list(EVIDENCE_KINDS.values()) if args.kind == "all" else [EVIDENCE_KINDS[args.kind]]
    for kind in kinds:
        if kind == "chunk_relevance":
            records = [
                r for d in run_dirs(config, args.run_ids) for r in final_ok_records(d, ["A1", "A2"])
            ]
            tasks = relevance_pool(by_id, retrieval, config.judge.pool_k, gold, records)
        else:
            tasks = sufficiency_tasks(
                by_id,
                retrieval,
                config.retrieval.top_k,
                gold if config.judge.judge_gold_sufficiency else None,
            )
        if args.dry_run:
            print(f"{kind}: {len(tasks)} tasks (cached ones are skipped when run)")
            continue
        n = run_evidence_judgments(
            kind,
            tasks,
            by_id,
            chunks,
            client,
            config.path("prompts_dir"),
            evidence_version(config, kind),
            evidence_path(config, kind),
            config.judge.max_workers,
        )
        print(f"{kind}: judged {n} new items -> {evidence_path(config, kind)}")


def cmd_human_sample(config: Config, args: argparse.Namespace) -> None:
    """Export a blinded sample for physician ratings (grounding, relevance or sufficiency)."""
    from vaxrag.evaluation.evidence import LABELS, export_evidence_sample, load_evidence_judgments
    from vaxrag.evaluation.judge import VERDICTS, export_human_sample

    questions = {q.question_id: q for q in load_questions(config)}
    chunks = {c.chunk_id: c for c in load_chunks(config)}
    out = config.path("analysis_dir") / f"human_rating_{args.kind}.csv"
    if args.kind == "grounding":
        if not args.run_ids:
            sys.exit("--run-ids is required for grounding samples")
        records = [
            r
            for d in run_dirs(config, args.run_ids)
            for r in final_ok_records(d, config.judge.arms)
            if r.rep == 1
        ]
        df = export_human_sample(
            records,
            questions,
            chunks,
            config.judge.human_sample_n,
            config.judge.seed,
            out,
            doc_texts=load_doc_texts(config),
        )
        labels = VERDICTS
    else:
        kind = EVIDENCE_KINDS[args.kind]
        judgments = load_evidence_judgments(
            evidence_path(config, kind), config.judge.model, evidence_version(config, kind)
        )
        df = export_evidence_sample(
            kind, judgments, questions, chunks, config.judge.human_sample_n, config.judge.seed, out
        )
        labels = LABELS[kind]
    print(f"Wrote {len(df)} items to {out}. Raters fill `rater_verdict` with: {'/'.join(labels)}")


def cmd_judge_agreement(config: Config, args: argparse.Namespace) -> None:
    """Cohen's kappa between a filled human rating CSV and the LLM judge."""
    from vaxrag.evaluation.evidence import load_evidence_judgments
    from vaxrag.evaluation.judge import GROUNDING_FILE, cohens_kappa

    human = pd.read_csv(args.ratings, dtype=str, keep_default_na=False)
    human = human[human["rater_verdict"].str.strip() != ""]
    if args.kind == "grounding":
        if not args.run_ids:
            sys.exit("--run-ids is required for grounding agreement")
        judged = pd.DataFrame(
            [
                row
                for d in run_dirs(config, args.run_ids)
                for row in read_jsonl(d / GROUNDING_FILE)
                if row["status"] == "ok"
            ]
        )
        id_col = "call_id"
    else:
        kind = EVIDENCE_KINDS[args.kind]
        judged = pd.DataFrame(
            [
                j.model_dump()
                for j in load_evidence_judgments(
                    evidence_path(config, kind), config.judge.model, evidence_version(config, kind)
                ).values()
            ]
        )
        id_col = "key"
    if judged.empty:
        sys.exit("No judgments found")
    merged = human.merge(judged[[id_col, "verdict"]].drop_duplicates(id_col), on=id_col)
    kappa = cohens_kappa(merged["rater_verdict"].str.strip().tolist(), merged["verdict"].tolist())
    print(f"n={len(merged)}  Cohen's kappa = {kappa:.3f}")


def cmd_collect(config: Config, args: argparse.Namespace) -> None:
    """Merge run logs, judgments and table reviews into the analysis Parquet file."""
    from vaxrag.analysis.collect import build_analysis_table
    from vaxrag.evaluation.table_review import faulty_tables
    from vaxrag.retrieval.hybrid import knowledge_base_hash

    chunks = load_chunks(config) if config.path("chunks").exists() else []
    faulty = (
        faulty_tables(config.path("table_review"), knowledge_base_hash(chunks)) if chunks else None
    )
    table = build_analysis_table(
        run_dirs(config, args.run_ids),
        load_questions(config),
        chunks,
        load_gold(config),
        manifest_titles(config),
        relevance=load_evidence(config, "chunk_relevance"),
        sufficiency=load_evidence(config, "context_sufficiency"),
        positive_relevance=tuple(config.judge.positive_relevance),
        quote_threshold=config.analysis.quote_match_threshold,
        title_threshold=config.analysis.citation_title_threshold,
        faulty_tables=faulty,
        doc_texts=load_doc_texts(config),
    )
    out = config.path("analysis_dir") / "all_calls.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"Wrote {len(table)} final calls to {out}")


def cmd_annotation_agreement(config: Config, args: argparse.Namespace) -> None:
    """Agreement between the two annotators, their disagreements and a consensus draft."""
    from vaxrag.data.agreement import (
        combined_gold,
        consensus_draft,
        field_agreement,
        field_disagreements,
        gold_document_agreement,
        load_annotator_tables,
    )

    names = config.annotation.annotators
    if len(names) != 2:
        sys.exit("annotation-agreement needs exactly two annotators (annotation.annotators)")
    base = config.path("annotators_dir")
    tables = load_annotator_tables({n: base / n / "question_annotations.csv" for n in names})
    golds = {
        n: pd.read_csv(base / n / "gold_passages.csv", dtype=str, keep_default_na=False)
        for n in names
    }
    a, b = (tables[n] for n in names)
    pair = (names[0], names[1])
    agreement = field_agreement(a, b)
    docs = gold_document_agreement(golds[names[0]], golds[names[1]], pair)
    out_dir = config.path("analysis_dir") / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    agreement.to_csv(out_dir / "field_agreement.csv", index=False)
    docs.to_csv(out_dir / "gold_document_agreement.csv", index=False)
    ann_dir = config.path("annotations").parent
    field_disagreements(a, b, pair).to_csv(ann_dir / "disagreements.csv", index=False)
    consensus_draft(a, b).to_csv(ann_dir / "consensus_draft.csv", index=False)
    combined_gold(golds).to_csv(ann_dir / "gold_passages_combined.csv", index=False)
    print(agreement.to_string(index=False))
    both = docs[docs["both_annotated"]]
    if not both.empty:
        print(
            f"Gold documents: {len(both)} questions annotated by both; "
            f"{both['shared_document'].mean():.1%} share at least one document "
            f"(mean Jaccard {both['jaccard'].mean():.2f})"
        )
    print(
        f"Wrote {ann_dir / 'disagreements.csv'}, {ann_dir / 'consensus_draft.csv'} and "
        f"{ann_dir / 'gold_passages_combined.csv'}. After the consensus discussion, copy the "
        "agreed labels to question_annotations.csv and gold_passages.csv."
    )


def cmd_table_review(config: Config, args: argparse.Namespace) -> None:
    """Queue tables retrieved for wrong A1 answers for a targeted parsing check."""
    from vaxrag.data.manifest import load_manifest
    from vaxrag.evaluation.evidence import lenient_relevant_sets
    from vaxrag.evaluation.table_review import build_review_queue, merge_with_existing
    from vaxrag.kb.table_vision import render_page_png
    from vaxrag.retrieval.hybrid import knowledge_base_hash

    parquet = config.path("analysis_dir") / "all_calls.parquet"
    if not parquet.exists():
        sys.exit(f"{parquet} not found - run `vaxrag collect` first")
    chunk_list = load_chunks(config)
    chunks = {c.chunk_id: c for c in chunk_list}
    relevant = lenient_relevant_sets(
        load_gold(config), load_evidence(config, "chunk_relevance"), config.judge.positive_relevance
    )
    queue = build_review_queue(
        pd.read_parquet(parquet),
        chunks,
        relevant,
        knowledge_base_hash(chunk_list),
        tuple(config.analysis.table_review_arms),
    )
    out = config.path("table_review")
    queue = merge_with_existing(queue, out)
    if args.images and not queue.empty:
        filenames = {e.doc_id: e.filename for e in load_manifest(config.path("manifest"))}
        image_dir = config.path("table_review_images")
        image_dir.mkdir(parents=True, exist_ok=True)
        for i, row in queue.iterrows():
            if str(row["parse_ok"]).strip() or not str(row["page_start"]).strip():
                continue  # already reviewed or page unknown
            start = int(float(row["page_start"]))
            end = int(float(row["page_end"] or start))
            paths = []
            for page in range(start, end + 1):
                path = image_dir / f"{row['doc_id']}_p{page}.png"
                if not path.exists():
                    path.write_bytes(
                        render_page_png(
                            config.path("guidelines_raw") / filenames[row["doc_id"]], page
                        )
                    )
                paths.append(str(path))
            queue.at[i, "page_images"] = ";".join(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(out, index=False)
    pending = (queue["parse_ok"].astype(str).str.strip() == "").sum() if not queue.empty else 0
    print(f"{len(queue)} tables in {out} ({pending} not yet reviewed).")
    print("Fill `parse_ok` (yes/no) and `error_description`, then rerun `collect` and `analyze`.")


def cmd_analyze(config: Config, args: argparse.Namespace) -> None:
    """Compute all analysis tables from the Parquet file."""
    from vaxrag.analysis.report import write_population_reports

    table = pd.read_parquet(config.path("analysis_dir") / "all_calls.parquet")
    written = write_population_reports(
        table,
        config.analysis,
        config.path("analysis_dir"),
        positive_sufficiency=tuple(config.judge.positive_sufficiency),
    )
    for population, paths in written.items():
        print(f"[{population}] {len(paths)} files")
        print("\n".join(f"  {p}" for p in paths))


# --------------------------------------------------------------------------- parser
COMMANDS = {
    "prepare-questions": (cmd_prepare_questions, "Convert the xlsx question bank to JSONL"),
    "manifest-hash": (cmd_manifest_hash, "Create/verify manifest.csv and fill SHA-256 hashes"),
    "parse": (cmd_parse, "Parse guideline PDFs with Docling"),
    "table-vision": (cmd_table_vision, "Convert hard tables with a multimodal LLM"),
    "chunk": (cmd_chunk, "Chunk parsed documents"),
    "annotation-agreement": (
        cmd_annotation_agreement,
        "Inter-annotator agreement + consensus draft",
    ),
    "resolve-gold": (cmd_resolve_gold, "Map gold passage annotations to chunk IDs"),
    "retrieve": (cmd_retrieve, "Run and cache hybrid retrieval"),
    "eval-retrieval": (cmd_eval_retrieval, "Evaluate retrieval against gold chunks"),
    "run": (cmd_run, "Run or resume the LLM experiment"),
    "judge": (cmd_judge, "Grounding judge: do quotes support the selected answer?"),
    "judge-evidence": (cmd_judge_evidence, "Chunk-relevance / context-sufficiency judgments"),
    "human-sample": (cmd_human_sample, "Export a blinded sample for physician ratings"),
    "judge-agreement": (cmd_judge_agreement, "Cohen's kappa between physicians and the judge"),
    "collect": (cmd_collect, "Merge run logs into all_calls.parquet"),
    "table-review": (cmd_table_review, "Queue tables seen in wrong A1 answers for review"),
    "analyze": (cmd_analyze, "Write analysis tables"),
}


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with one sub-command per pipeline stage."""
    parser = argparse.ArgumentParser(prog="vaxrag", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--config",
            action="append",
            default=None,
            help="YAML config file(s); later files override earlier ones "
            "(default: configs/default.yaml)",
        )
        if name == "parse":
            p.add_argument("--force", action="store_true", help="Re-parse already parsed documents")
        if name == "table-vision":
            p.add_argument("--model", help="Model key (default: table_vision_model)")
            p.add_argument(
                "--overwrite", action="store_true", help="Overwrite existing override files"
            )
        if name == "run":
            p.add_argument("--run-id", required=True, help="Run folder name; reuse it to resume")
            p.add_argument("--models", nargs="+", help="Model keys (default: experiment.models)")
            p.add_argument("--dry-run", action="store_true", help="Only count the calls to be made")
            p.add_argument(
                "--allow-config-change",
                action="store_true",
                help="Continue a run even though result-affecting settings changed",
            )
        if name in {"judge", "collect"}:
            p.add_argument("--run-ids", nargs="+", required=True, help="Run folder names")
        if name == "judge-evidence":
            p.add_argument("--kind", choices=["relevance", "sufficiency", "all"], default="all")
            p.add_argument(
                "--run-ids",
                nargs="*",
                default=[],
                help="Runs whose cited chunks are added to the relevance pool",
            )
            p.add_argument("--dry-run", action="store_true", help="Only count the tasks")
        if name == "table-review":
            p.add_argument("--images", action="store_true", help="Render the PDF pages to PNG")
        if name in {"human-sample", "judge-agreement"}:
            p.add_argument(
                "--kind", choices=["grounding", "relevance", "sufficiency"], default="grounding"
            )
            p.add_argument("--run-ids", nargs="*", default=[], help="Run folder names (grounding)")
        if name == "judge-agreement":
            p.add_argument("--ratings", required=True, help="Filled human_rating_sample.csv")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `vaxrag` command."""
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "httpcore2",
        "openai",
        "anthropic",
        "google_genai",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        config = load_config(args.config or ["configs/default.yaml"])
    except ValidationError as exc:
        messages = "\n".join(f"- {e['msg']}" for e in exc.errors())
        sys.exit(f"Invalid configuration:\n{messages}")
    COMMANDS[args.command][0](config, args)


if __name__ == "__main__":
    main()
