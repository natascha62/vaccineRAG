"""Inter-annotator agreement for the two independent physician annotators.

Each annotator fills their own copy of the annotation files in
`data/annotations/annotators/<name>/`. This module

* computes per-field agreement (percent agreement and Cohen's kappa) for the
  categorical labels,
* compares the gold-passage *documents* each annotator chose (sections and
  quotes legitimately differ between annotators, so agreement is measured at
  document level),
* lists every disagreement, and
* drafts a consensus table in which agreed values are pre-filled and
  disagreements are left blank for the consensus discussion.

The consensus draft is never written over the final annotation files; the
annotators review it and copy the agreed result into
`data/annotations/question_annotations.csv` and `gold_passages.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaxrag.data.annotations import ANNOTATION_COLUMNS, GOLD_COLUMNS, validate_annotations
from vaxrag.evaluation.judge import cohens_kappa

AGREEMENT_FIELDS = ("kb_covered", "version_status", "answer_source", "question_type")
_BOOL_NORMALIZE = {"y": "yes", "true": "yes", "1": "yes", "n": "no", "false": "no", "0": "no"}


def normalize(field: str, value: str) -> str:
    """Normalize a label for comparison (case, whitespace, yes/no spellings)."""
    value = str(value).strip().lower()
    if field == "kb_covered":
        return _BOOL_NORMALIZE.get(value, value)
    return value


def load_annotator_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load and validate each annotator's question annotation file."""
    tables: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Annotation file of '{name}' not found: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        problems = validate_annotations(df)
        if problems:
            raise ValueError(f"Invalid annotations of '{name}':\n" + "\n".join(problems[:20]))
        tables[name] = df.set_index("question_id")
    return tables


def field_agreement(a: pd.DataFrame, b: pd.DataFrame, fields=AGREEMENT_FIELDS) -> pd.DataFrame:
    """Percent agreement and Cohen's kappa per field over questions both annotated."""
    common = a.index.intersection(b.index)
    rows = []
    for field in fields:
        va = a.loc[common, field].map(lambda v, f=field: normalize(f, v))
        vb = b.loc[common, field].map(lambda v, f=field: normalize(f, v))
        both = (va != "") & (vb != "")
        x, y = va[both].tolist(), vb[both].tolist()
        rows.append(
            {
                "field": field,
                "n_both_annotated": len(x),
                "percent_agreement": (
                    sum(i == j for i, j in zip(x, y, strict=True)) / len(x) if x else float("nan")
                ),
                "cohens_kappa": cohens_kappa(x, y) if x else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def field_disagreements(
    a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str], fields=AGREEMENT_FIELDS
) -> pd.DataFrame:
    """One row per (question, field) where both annotators answered differently or one is blank."""
    common = a.index.intersection(b.index)
    rows = []
    for qid in common:
        for field in fields:
            va, vb = normalize(field, a.at[qid, field]), normalize(field, b.at[qid, field])
            if va != vb:
                rows.append(
                    {
                        "question_id": qid,
                        "field": field,
                        names[0]: va,
                        names[1]: vb,
                        "stem": a.at[qid, "stem"],
                    }
                )
    return pd.DataFrame(rows, columns=["question_id", "field", names[0], names[1], "stem"])


def gold_document_agreement(
    gold_a: pd.DataFrame, gold_b: pd.DataFrame, names: tuple[str, str]
) -> pd.DataFrame:
    """Per question: the gold documents chosen by each annotator and their overlap (Jaccard)."""
    docs_a = gold_a.groupby("question_id")["doc_id"].apply(lambda s: set(s.str.strip()) - {""})
    docs_b = gold_b.groupby("question_id")["doc_id"].apply(lambda s: set(s.str.strip()) - {""})
    rows = []
    for qid in sorted(set(docs_a.index) | set(docs_b.index)):
        da, db = docs_a.get(qid, set()), docs_b.get(qid, set())
        union = da | db
        rows.append(
            {
                "question_id": qid,
                names[0]: ";".join(sorted(da)),
                names[1]: ";".join(sorted(db)),
                "both_annotated": bool(da) and bool(db),
                "shared_document": bool(da & db),
                "jaccard": len(da & db) / len(union) if union else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def consensus_draft(a: pd.DataFrame, b: pd.DataFrame, fields=AGREEMENT_FIELDS) -> pd.DataFrame:
    """Annotation table with agreed values filled in and disagreements left blank.

    Disagreements are described in `notes` so they can be settled in the
    consensus meeting.
    """
    draft = a.copy()
    common = a.index.intersection(b.index)
    for qid in common:
        notes = []
        for field in fields:
            va, vb = normalize(field, a.at[qid, field]), normalize(field, b.at[qid, field])
            if va == vb:
                draft.at[qid, field] = va
            else:
                draft.at[qid, field] = ""
                notes.append(f"{field}: {va or '-'} vs {vb or '-'}")
        draft.at[qid, "notes"] = "; ".join(notes)
    return draft.reset_index().reindex(columns=ANNOTATION_COLUMNS).fillna("")


def combined_gold(golds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """All annotators' gold passages in one table (with an `annotator` column) for consensus."""
    frames = [df.assign(annotator=name) for name, df in golds.items()]
    if not frames:
        return pd.DataFrame(columns=["annotator", *GOLD_COLUMNS])
    out = pd.concat(frames, ignore_index=True).reindex(columns=["annotator", *GOLD_COLUMNS])
    return out.fillna("").sort_values(["question_id", "annotator"]).reset_index(drop=True)
