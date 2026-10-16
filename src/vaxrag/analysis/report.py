"""Write all analysis tables (CSV) and model summaries (text) to the analysis folder."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vaxrag.analysis import stats
from vaxrag.config import AnalysisConfig
from vaxrag.evaluation.quadrants import (
    EVIDENCE_DEFINITIONS,
    quadrant_table,
    sufficiency_by_correctness,
)
from vaxrag.evaluation.scoring import flip_table
from vaxrag.evaluation.table_review import parse_error_table

logger = logging.getLogger(__name__)

STRATA = ["subset", "vaccine", "category", "question_type", "answer_source", "kb_covered"]


def primary_population(table: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Rows belonging to the primary analysis population (see `PopulationFilter`).

    Questions without the required annotations are excluded, so the primary
    population is empty until annotation is done.
    """
    pop = cfg.primary_population
    mask = pd.Series(True, index=table.index)
    if pop.only_kb_covered:
        mask &= table["kb_covered"].eq(True)
    if pop.version_status:
        mask &= table["version_status"].isin(pop.version_status)
    return table[mask]


def write_population_reports(
    table: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: str | Path,
    positive_sufficiency: tuple[str, ...] = ("sufficient",),
) -> dict[str, list[Path]]:
    """Write the full report for the primary population and for all questions.

    Outputs go to `<out_dir>/primary/` and `<out_dir>/all_questions/`.
    """
    written: dict[str, list[Path]] = {}
    for name, part in (("primary", primary_population(table, cfg)), ("all_questions", table)):
        if part.empty:
            logger.warning("No rows for population '%s' (annotations missing?)", name)
            written[name] = []
            continue
        written[name] = write_report(part, cfg, Path(out_dir) / name, positive_sufficiency)
    return written


def write_report(
    table: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: str | Path,
    positive_sufficiency: tuple[str, ...] = ("sufficient",),
) -> list[Path]:
    """Compute every table of the analysis plan and write it to `out_dir`.

    Returns the list of files written. Analyses whose inputs are missing (e.g.
    no gold annotations yet, or only one model) are skipped with a log message.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(df: pd.DataFrame, name: str) -> None:
        if df is None or df.empty:
            logger.info("Skipped %s (no data)", name)
            return
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)

    main = stats.main_analysis_rows(table)
    save(stats.accuracy_table(main, ["model", "arm"]), "accuracy_by_model_arm")
    for stratum in STRATA:
        if stratum in main and main[stratum].notna().any():
            save(stats.accuracy_table(main, ["model", "arm", stratum]), f"accuracy_by_{stratum}")
    save(
        stats.pairwise_arm_tests(main, iterations=cfg.bootstrap_iterations, seed=cfg.seed),
        "paired_arm_tests",
    )
    for treat, ref in (("A1", "A0"), ("A2", "A1"), ("A3", "A1")):
        save(flip_table(main, ref, treat), f"flips_{treat}_vs_{ref}")
    for definition in EVIDENCE_DEFINITIONS:
        save(quadrant_table(main, definition, positive_sufficiency), f"quadrants_A1_{definition}")
    save(sufficiency_by_correctness(main), "sufficiency_by_correctness")
    save(parse_error_table(main), "table_parse_errors_A1")
    save(stats.citation_table(main), "citations")
    save(stats.efficiency_table(main), "efficiency")
    save(stats.doc_selection_table(main), "doc_selection_A3")
    save(stats.consistency_table(table), "consistency")

    if cfg.paired_models and set(cfg.paired_models) <= set(main["model"]):
        base, med = cfg.paired_models
        save(
            stats.model_pair_tests(
                main, base, med, sorted(main["arm"].unique()), cfg.bootstrap_iterations, cfg.seed
            ),
            "paired_models_H5",
        )

    summary_lines: list[str] = []
    # H1/H2 use the arms run by every model; A3 (API models only) is analysed separately.
    primary = main[main["arm"].isin(cfg.primary_arms)]
    if "A3" in set(main["arm"]):
        a3_models = set(main.loc[main["arm"] == "A3", "model"])
        a3_rows = main[main["arm"].isin(["A1", "A3"]) & main["model"].isin(a3_models)]
        try:
            save(stats.gee_arm_effects(a3_rows, "A1")["arm_odds_ratios"], "gee_A3_vs_A1")
        except Exception as exc:  # noqa: BLE001 - report and continue with other analyses
            summary_lines.append(f"GEE A3 vs A1 failed: {exc}")
    if primary["arm"].nunique() > 1:
        try:
            gee = stats.gee_arm_effects(primary, cfg.reference_arm)
            save(gee["arm_odds_ratios"], "gee_arm_odds_ratios")
            if "interaction_p_value" in gee:
                summary_lines.append(
                    "H2 arm x model interaction (GEE robust Wald): "
                    f"chi2={gee['interaction_wald_stat']:.3f}, "
                    f"df={gee['interaction_df']:.0f}, p={gee['interaction_p_value']:.4g}"
                )
        except Exception as exc:  # noqa: BLE001 - report and continue with other analyses
            summary_lines.append(f"GEE failed: {exc}")
        if "answer_source" in main and main["answer_source"].isin(["table", "text"]).any():
            a1 = main[(main["arm"] == "A1") & main["answer_source"].isin(["table", "text"])]
            save(stats.accuracy_table(a1, ["model", "answer_source"]), "H4_A1_table_vs_text")
        if cfg.run_glmm:
            try:
                save(
                    stats.bayes_mixed_arm_effects(primary, cfg.reference_arm),
                    "glmm_bayes_fixed_effects",
                )
            except Exception as exc:  # noqa: BLE001
                summary_lines.append(f"Bayesian GLMM failed: {exc}")
    if summary_lines:
        path = out / "model_summaries.txt"
        path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        written.append(path)
    return written
