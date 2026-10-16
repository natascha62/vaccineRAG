"""Answer scoring and flip analysis.

A call is correct when its final status is `ok` and the selected set of
(original) option letters equals the set of correct letters. Refusals and
calls that still fail to parse after all attempts count as incorrect.
"""

from __future__ import annotations

import pandas as pd

from vaxrag.schemas import Question

FINAL_ORDER = {"api_error": 0, "parse_error": 1, "refusal": 2, "ok": 3}


def final_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Reduce an attempt-level call log to one final row per `call_id`.

    The final row is the successful attempt if any, otherwise the latest
    attempt. Calls whose only records are API errors are dropped (they are not
    finished and will be retried by the runner).
    """
    if calls.empty:
        return calls
    df = calls.copy()
    df["_status_rank"] = df["status"].map(FINAL_ORDER)
    df = df.sort_values(["call_id", "_status_rank", "attempt"])
    final = df.groupby("call_id", as_index=False).tail(1)
    final = final[final["status"] != "api_error"]
    return final.drop(columns="_status_rank").reset_index(drop=True)


def is_correct(answer: list[str] | None, correct: list[str]) -> bool:
    """Exact set match between selected and correct options."""
    return bool(answer) and set(answer) == set(correct)


def score_calls(final: pd.DataFrame, questions: dict[str, Question]) -> pd.DataFrame:
    """Add `correct` (bool) and `parse_failed` columns to final call rows."""
    df = final.copy()
    df["correct"] = [
        status == "ok" and is_correct(list(ans), questions[qid].correct)
        for status, ans, qid in zip(df["status"], df["answer"], df["question_id"], strict=True)
    ]
    df["parse_failed"] = df["status"] == "parse_error"
    df["refused"] = df["status"] == "refusal"
    return df


def flip_table(scored: pd.DataFrame, arm_from: str = "A0", arm_to: str = "A1") -> pd.DataFrame:
    """Count RAG rescues (wrong -> right) and harms (right -> wrong) per model.

    Uses repetition 1 only.
    """
    base = scored[scored["rep"] == 1]
    wide = base.pivot_table(
        index=["model", "question_id"], columns="arm", values="correct", aggfunc="first"
    )
    if arm_from not in wide or arm_to not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[arm_from, arm_to]).astype(bool)
    rows = []
    for model, group in wide.groupby(level="model"):
        a, b = group[arm_from], group[arm_to]
        rows.append(
            {
                "model": model,
                "contrast": f"{arm_to} vs {arm_from}",
                "n": len(group),
                "both_correct": int((a & b).sum()),
                "rescued": int((~a & b).sum()),
                "harmed": int((a & ~b).sum()),
                "both_wrong": int((~a & ~b).sum()),
            }
        )
    return pd.DataFrame(rows)
