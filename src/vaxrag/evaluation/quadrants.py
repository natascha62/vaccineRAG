"""Four-quadrant error attribution for the standard RAG arm (A1).

|                          | correct                  | incorrect          |
| evidence in the context  | grounded success         | generation failure |
| evidence not in context  | own knowledge / guess    | retrieval failure  |

"Evidence in the context" can be defined three ways (columns of the analysis
table), reported side by side:

* `strict`  - an annotated gold chunk was retrieved (`gold_in_context_strict`);
  underestimates evidence because gold sets are not exhaustive;
* `lenient` - a gold or judge-relevant chunk was retrieved
  (`relevant_in_context_lenient`);
* `judged`  - the judge found the retrieved context sufficient
  (`context_sufficiency_judged` in `positive_sufficiency`). This is the
  primary definition because it needs no exhaustive annotation and accounts
  for information spread over several chunks.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

QUADRANT_LABELS = {
    (True, True): "grounded_success",
    (True, False): "generation_failure",
    (False, True): "own_knowledge_or_guess",
    (False, False): "retrieval_failure",
}

EVIDENCE_DEFINITIONS = {
    "strict": "gold_in_context_strict",
    "lenient": "relevant_in_context_lenient",
    "judged": "context_sufficiency_judged",
}


def assign_quadrant(evidence_present: bool, correct: bool) -> str:
    """Return the quadrant label for one (evidence present, correct) combination."""
    return QUADRANT_LABELS[(bool(evidence_present), bool(correct))]


def evidence_flags(
    df: pd.DataFrame, definition: str, positive_sufficiency: Iterable[str] = ("sufficient",)
) -> pd.Series:
    """Boolean evidence indicator (NaN where undetermined) for one definition."""
    column = EVIDENCE_DEFINITIONS[definition]
    if column not in df:
        return pd.Series(float("nan"), index=df.index)
    values = df[column]
    if definition == "judged":
        positive = set(positive_sufficiency)
        return values.map(lambda v: None if v is None or pd.isna(v) else v in positive)
    return values


def quadrant_table(
    scored: pd.DataFrame,
    definition: str = "judged",
    positive_sufficiency: Iterable[str] = ("sufficient",),
) -> pd.DataFrame:
    """Count quadrants (and their shares) per model for A1 calls with known evidence status."""
    df = scored[scored["arm"] == "A1"].copy()
    df["_evidence"] = evidence_flags(df, definition, positive_sufficiency)
    df = df[df["_evidence"].notna()]
    if df.empty:
        return pd.DataFrame()
    df["quadrant"] = [
        assign_quadrant(e, c) for e, c in zip(df["_evidence"], df["correct"], strict=True)
    ]
    counts = df.groupby(["model", "quadrant"]).size().unstack(fill_value=0)
    for label in QUADRANT_LABELS.values():
        if label not in counts:
            counts[label] = 0
    counts = counts[list(QUADRANT_LABELS.values())]
    shares = counts.div(counts.sum(axis=1), axis=0).add_suffix("_share")
    out = pd.concat([counts, shares], axis=1).reset_index()
    out.insert(1, "evidence_definition", definition)
    out.insert(2, "n", counts.sum(axis=1).to_numpy())
    return out


def sufficiency_by_correctness(scored: pd.DataFrame) -> pd.DataFrame:
    """Three-level sufficiency verdict crossed with correctness, per model and arm."""
    df = (
        scored[scored["context_sufficiency_judged"].notna()]
        if ("context_sufficiency_judged" in scored)
        else pd.DataFrame()
    )
    if df.empty:
        return pd.DataFrame()
    table = (
        df.groupby(["model", "arm", "context_sufficiency_judged"])["correct"]
        .agg(n="count", accuracy="mean")
        .reset_index()
    )
    return table.rename(columns={"context_sufficiency_judged": "sufficiency"})
