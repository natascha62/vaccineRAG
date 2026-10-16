"""Statistical analysis (Python only).

Primary analysis (H1, H2) - Python has no standard frequentist GLMM with
likelihood-ratio tests, so the analysis uses:

* **GEE logistic regression** (binomial, logit link, exchangeable working
  correlation, clustered by question, robust SEs). Its coefficients are
  population-averaged, so `exp(beta_arm)` from the main-effects model is the
  marginal odds ratio of A1 vs A0 pooled over models (H1). H2 is a robust
  Wald test of all arm x model interaction terms.
* **Bayesian mixed logistic regression** (`BinomialBayesMixedGLM`, random
  intercept per question, variational Bayes) as the subject-specific GLMM
  sensitivity analysis matching the plan's model specification.

Per-model pairwise comparisons use exact McNemar tests with Holm correction,
and 95% CIs for accuracy differences from a question-level bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

DEFAULT_CONTRASTS: tuple[tuple[str, str], ...] = (
    ("A1", "A0"),
    ("A2", "A1"),
    ("A2", "A0"),
    ("A3", "A1"),
    ("A3", "A0"),
)


def main_analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of the primary analysis (first repetition only)."""
    return df[df["rep"] == 1].copy()


def accuracy_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Accuracy with Wilson 95% CI per group."""
    grouped = df.groupby(by, dropna=False)["correct"].agg(["sum", "count"]).reset_index()
    low, high = proportion_confint(grouped["sum"], grouped["count"], method="wilson")
    grouped["accuracy"] = grouped["sum"] / grouped["count"]
    grouped["ci_low"], grouped["ci_high"] = low, high
    return grouped.rename(columns={"sum": "n_correct", "count": "n"})


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    """Exact (binomial) McNemar p-value for paired binary outcomes."""
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, iterations: int = 1000, seed: int = 2026, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI for mean(b) - mean(a), resampling questions (paired)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(iterations, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def _paired(df: pd.DataFrame, index: str, columns: str, left: str, right: str) -> pd.DataFrame:
    """Wide boolean table of paired outcomes with both conditions present."""
    wide = df.pivot_table(index=index, columns=columns, values="correct", aggfunc="first")
    if left not in wide or right not in wide:
        return pd.DataFrame()
    return wide[[left, right]].dropna().astype(bool)


def pairwise_arm_tests(
    df: pd.DataFrame,
    contrasts: tuple[tuple[str, str], ...] = DEFAULT_CONTRASTS,
    iterations: int = 1000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Per-model paired comparisons between arms with Holm-adjusted McNemar p-values.

    A contrast (X, Y) reports accuracy(X) - accuracy(Y).
    """
    rows = []
    for model, group in df.groupby("model"):
        for treat, ref in contrasts:
            wide = _paired(group, "question_id", "arm", ref, treat)
            if wide.empty:
                continue
            a, b = wide[ref].to_numpy(), wide[treat].to_numpy()
            low, high = bootstrap_diff_ci(a, b, iterations, seed)
            rows.append(
                {
                    "model": model,
                    "contrast": f"{treat} - {ref}",
                    "n_pairs": len(wide),
                    "acc_ref": a.mean(),
                    "acc_treat": b.mean(),
                    "diff": b.mean() - a.mean(),
                    "diff_ci_low": low,
                    "diff_ci_high": high,
                    "n_rescued": int((~a & b).sum()),
                    "n_harmed": int((a & ~b).sum()),
                    "p_mcnemar": mcnemar_exact(a, b),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = multipletests(out["p_mcnemar"], method="holm")[1]
    return out


def model_pair_tests(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    arms: list[str],
    iterations: int = 1000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Compare two models within each arm (H5: MedGemma vs Gemma). Reports acc(b) - acc(a)."""
    rows = []
    for arm, group in df.groupby("arm"):
        if arm not in arms:
            continue
        wide = _paired(group, "question_id", "model", model_a, model_b)
        if wide.empty:
            continue
        a, b = wide[model_a].to_numpy(), wide[model_b].to_numpy()
        low, high = bootstrap_diff_ci(a, b, iterations, seed)
        rows.append(
            {
                "arm": arm,
                "comparison": f"{model_b} - {model_a}",
                "n_pairs": len(wide),
                "acc_a": a.mean(),
                "acc_b": b.mean(),
                "diff": b.mean() - a.mean(),
                "diff_ci_low": low,
                "diff_ci_high": high,
                "p_mcnemar": mcnemar_exact(a, b),
            }
        )
    return pd.DataFrame(rows)


def _design_frame(df: pd.DataFrame, reference_arm: str) -> pd.DataFrame:
    """Prepare categorical columns with a fixed reference level for regression."""
    data = df[["correct", "arm", "model", "question_id"]].copy()
    data["y"] = data["correct"].astype(int)
    arms = [reference_arm] + sorted(a for a in data["arm"].unique() if a != reference_arm)
    data["arm"] = pd.Categorical(data["arm"], categories=arms)
    data["model"] = pd.Categorical(data["model"])
    return data.sort_values("question_id")


def gee_arm_effects(df: pd.DataFrame, reference_arm: str = "A0") -> dict[str, pd.DataFrame | float]:
    """GEE estimates: marginal arm odds ratios (H1) and the interaction Wald test (H2)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    data = _design_frame(df, reference_arm)
    family, cov = sm.families.Binomial(), sm.cov_struct.Exchangeable()
    main = smf.gee(
        "y ~ C(arm) + C(model)", groups="question_id", data=data, family=family, cov_struct=cov
    ).fit()
    conf = main.conf_int()
    params = main.params
    arm_terms = [t for t in params.index if t.startswith("C(arm)")]
    odds = pd.DataFrame(
        {
            "term": arm_terms,
            "odds_ratio": np.exp(params[arm_terms]).to_numpy(),
            "ci_low": np.exp(conf.loc[arm_terms, 0]).to_numpy(),
            "ci_high": np.exp(conf.loc[arm_terms, 1]).to_numpy(),
            "p_value": main.pvalues[arm_terms].to_numpy(),
        }
    )
    result: dict[str, pd.DataFrame | float] = {"arm_odds_ratios": odds}
    if data["model"].nunique() > 1:
        inter = smf.gee(
            "y ~ C(arm) * C(model)",
            groups="question_id",
            data=data,
            family=family,
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        names = list(inter.params.index)
        idx = [i for i, n in enumerate(names) if ":" in n]
        restriction = np.zeros((len(idx), len(names)))
        for row, col in enumerate(idx):
            restriction[row, col] = 1.0
        wald = inter.wald_test(restriction, scalar=True)
        result["interaction_wald_stat"] = float(np.squeeze(wald.statistic))
        result["interaction_df"] = float(len(idx))
        result["interaction_p_value"] = float(np.squeeze(wald.pvalue))
    return result


def bayes_mixed_arm_effects(df: pd.DataFrame, reference_arm: str = "A0") -> pd.DataFrame:
    """GLMM sensitivity analysis: random intercept per question (variational Bayes).

    Returns posterior mean and approximate 95% credible interval of each fixed
    effect on the log-odds scale, plus the odds-ratio transform.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    data = _design_frame(df, reference_arm)
    model = BinomialBayesMixedGLM.from_formula(
        "y ~ C(arm) + C(model)", {"question": "0 + C(question_id)"}, data
    )
    fit = model.fit_vb()
    names = model.exog_names
    mean, sd = np.asarray(fit.fe_mean), np.asarray(fit.fe_sd)
    out = pd.DataFrame({"term": names, "post_mean": mean, "post_sd": sd})
    out["ci_low"], out["ci_high"] = mean - 1.96 * sd, mean + 1.96 * sd
    out["odds_ratio"] = np.exp(out["post_mean"])
    out["or_ci_low"], out["or_ci_high"] = np.exp(out["ci_low"]), np.exp(out["ci_high"])
    return out


def consistency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of questions where all repetitions give the same answer."""
    multi = df.groupby(["model", "arm", "question_id"]).filter(lambda g: g["rep"].nunique() > 1)
    if multi.empty:
        return pd.DataFrame()
    agreement = (
        multi.groupby(["model", "arm", "question_id"])["answer"]
        .nunique()
        .eq(1)
        .rename("consistent")
    )
    return (
        agreement.groupby(["model", "arm"])
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "consistency_rate", "count": "n_questions"})
    )


def efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Token usage, latency and cost per correct answer per model and arm."""
    grouped = (
        df.groupby(["model", "arm"])
        .agg(
            n=("call_id", "count"),
            n_correct=("correct", "sum"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            median_latency_s=("latency_s", "median"),
            total_cost_usd=("cost_usd", "sum"),
            parse_failure_rate=("parse_failed", "mean"),
            refusal_rate=("refused", "mean"),
        )
        .reset_index()
    )
    grouped["cost_per_correct_usd"] = grouped["total_cost_usd"] / grouped["n_correct"].replace(
        0, np.nan
    )
    return grouped


def citation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean citation metrics per model and arm (columns missing for an arm are NaN).

    Strict metrics compare with annotated gold chunks (lower bounds); judged
    metrics use chunk-relevance judgments. Coverage columns give the share of
    answers for which a judged metric could be determined. Two "correct but
    not supported" shares are reported:

    * `correct_without_supporting_citation_share` - correct answers whose
      cited chunks do not support the correct answer (chunk relevance);
    * `correct_but_ungrounded_share` - correct answers whose quotes do not
      support the selected answer (grounding judge).
    """
    metrics = [
        c
        for c in (
            "n_references",
            "share_quotes_verbatim",
            "all_quotes_verbatim",
            "citation_recall_strict",
            "citation_precision_strict",
            "citation_support_any_judged",
            "citation_precision_judged",
            "memory_citation_doc_match_share",
            "memory_citation_section_match_share",
        )
        if c in df
    ]
    ok = df[df["status"] == "ok"].copy()
    for col in metrics:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    grouped = ok.groupby(["model", "arm"])
    table = grouped[metrics].mean().reset_index()
    for col in ("citation_support_any_judged", "citation_precision_judged"):
        if col in ok:
            coverage = grouped[col].apply(lambda s: s.notna().mean()).rename(f"{col}_coverage")
            table = table.merge(coverage.reset_index(), on=["model", "arm"], how="left")
    correct = ok[ok["correct"]]
    if "citation_support_any_judged" in ok:
        known = correct[correct["citation_support_any_judged"].notna()]
        if not known.empty:
            share = (
                known.assign(u=known["citation_support_any_judged"].eq(0))
                .groupby(["model", "arm"])["u"]
                .mean()
                .rename("correct_without_supporting_citation_share")
            )
            table = table.merge(share.reset_index(), on=["model", "arm"], how="left")
    if "grounding_verdict" in ok and ok["grounding_verdict"].notna().any():
        verdicts = pd.crosstab([ok["model"], ok["arm"]], ok["grounding_verdict"], normalize="index")
        table = table.merge(
            verdicts.add_prefix("grounding_").reset_index(), on=["model", "arm"], how="left"
        )
        known = correct[correct["grounding_verdict"].notna()]
        if not known.empty:
            share = (
                known.assign(u=known["grounding_verdict"].eq("not_supports"))
                .groupby(["model", "arm"])["u"]
                .mean()
                .rename("correct_but_ungrounded_share")
            )
            table = table.merge(share.reset_index(), on=["model", "arm"], how="left")
    return table


def doc_selection_table(df: pd.DataFrame) -> pd.DataFrame:
    """Arm A3 per model: documents selected, gold-document hit rate and accuracy split by hit.

    Separates "selection loss" (the document with the answer was not chosen)
    from "reading loss" (it was chosen, but the answer was still wrong).
    """
    a3 = df[df["arm"] == "A3"] if "arm" in df else pd.DataFrame()
    if a3.empty or "n_selected_docs" not in a3:
        return pd.DataFrame()
    rows = []
    for model, group in a3.groupby("model"):
        known = group[group["gold_doc_selected"].notna()]
        hit = known[known["gold_doc_selected"].astype(bool)]
        miss = known[~known["gold_doc_selected"].astype(bool)]
        rows.append(
            {
                "model": model,
                "n": len(group),
                "mean_docs_selected": group["n_selected_docs"].mean(),
                "n_with_gold": len(known),
                "gold_doc_selected_rate": len(hit) / len(known) if len(known) else float("nan"),
                "accuracy_gold_selected": hit["correct"].mean() if len(hit) else float("nan"),
                "accuracy_gold_not_selected": miss["correct"].mean() if len(miss) else float("nan"),
            }
        )
    return pd.DataFrame(rows)
