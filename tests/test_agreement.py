"""Tests for inter-annotator agreement and the consensus draft."""

import math
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_question

from vaxrag.data.agreement import (
    combined_gold,
    consensus_draft,
    field_agreement,
    field_disagreements,
    gold_document_agreement,
    load_annotator_tables,
)
from vaxrag.data.annotations import create_annotation_templates

NAMES = ("adult_id", "pediatric_id")


def _annotator_files(tmp_path: Path, labels: dict[str, list[dict]]) -> dict[str, Path]:
    questions = [make_question(f"q{i}") for i in range(4)]
    paths = {}
    for name, rows in labels.items():
        folder = tmp_path / name
        create_annotation_templates(
            questions, folder / "question_annotations.csv", folder / "gold_passages.csv"
        )
        df = pd.read_csv(folder / "question_annotations.csv", dtype=str, keep_default_na=False)
        for i, row in enumerate(rows):
            for field, value in row.items():
                df.loc[i, field] = value
        df.to_csv(folder / "question_annotations.csv", index=False)
        paths[name] = folder / "question_annotations.csv"
    return paths


ADULT = [
    {"kb_covered": "yes", "version_status": "current", "answer_source": "table"},
    {"kb_covered": "Yes", "version_status": "current", "answer_source": "text"},
    {"kb_covered": "no", "version_status": "unmappable"},
    {"kb_covered": "yes", "version_status": "outdated", "answer_source": "text"},
]
PEDS = [
    {"kb_covered": "y", "version_status": "current", "answer_source": "table"},
    {"kb_covered": "yes", "version_status": "outdated", "answer_source": "text"},
    {"kb_covered": "no", "version_status": "unmappable"},
    {"kb_covered": "yes"},  # not finished
]


def test_field_agreement_and_disagreements(tmp_path: Path):
    tables = load_annotator_tables(
        _annotator_files(tmp_path, {"adult_id": ADULT, "pediatric_id": PEDS})
    )
    a, b = tables["adult_id"], tables["pediatric_id"]
    agreement = field_agreement(a, b).set_index("field")
    assert agreement.loc["kb_covered", "n_both_annotated"] == 4
    assert agreement.loc["kb_covered", "percent_agreement"] == 1.0  # yes/Yes/y normalized
    assert agreement.loc["version_status", "n_both_annotated"] == 3  # q3 blank for one annotator
    assert agreement.loc["version_status", "percent_agreement"] == pytest.approx(2 / 3)
    assert math.isnan(agreement.loc["question_type", "percent_agreement"])
    dis = field_disagreements(a, b, NAMES)
    assert set(zip(dis["question_id"], dis["field"], strict=True)) == {
        ("q1", "version_status"),
        ("q3", "version_status"),
        ("q3", "answer_source"),
    }


def test_consensus_draft_fills_agreed_values_only(tmp_path: Path):
    tables = load_annotator_tables(
        _annotator_files(tmp_path, {"adult_id": ADULT, "pediatric_id": PEDS})
    )
    draft = consensus_draft(tables["adult_id"], tables["pediatric_id"]).set_index("question_id")
    assert draft.loc["q0", "kb_covered"] == "yes" and draft.loc["q0", "answer_source"] == "table"
    assert draft.loc["q1", "version_status"] == ""
    assert "version_status: current vs outdated" in draft.loc["q1", "notes"]
    assert draft.loc["q0", "notes"] == ""
    assert list(draft.reset_index().columns)[0] == "question_id"


def test_invalid_annotator_values_are_rejected(tmp_path: Path):
    paths = _annotator_files(tmp_path, {"adult_id": [{"answer_source": "figure"}]})
    with pytest.raises(ValueError, match="adult_id"):
        load_annotator_tables(paths)
    with pytest.raises(FileNotFoundError):
        load_annotator_tables({"x": tmp_path / "missing.csv"})


def test_gold_document_agreement_and_combined_table():
    gold_a = pd.DataFrame({"question_id": ["q1", "q1", "q2"], "doc_id": ["d1", "d2", "d3"]})
    gold_b = pd.DataFrame({"question_id": ["q1", "q3"], "doc_id": ["d2", "d4"]})
    docs = gold_document_agreement(gold_a, gold_b, NAMES).set_index("question_id")
    assert docs.loc["q1", "shared_document"] and docs.loc["q1", "jaccard"] == 0.5
    assert not docs.loc["q2", "both_annotated"] and docs.loc["q3", "adult_id"] == ""
    combined = combined_gold({"adult_id": gold_a.assign(section=""), "pediatric_id": gold_b})
    assert list(combined.columns[:3]) == ["annotator", "question_id", "doc_id"]
    assert len(combined) == 5
