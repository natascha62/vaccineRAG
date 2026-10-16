from pathlib import Path

import pytest

from vaxrag.data.questions import (
    detect_format,
    load_question_bank,
    load_question_file,
    normalize_category,
    normalize_vaccine,
)


def test_s1_rows_are_normalized(s1_xlsx: Path):
    questions, exclusions = load_question_file(s1_xlsx, "S1", multi_answer=False)
    by_id = {q.question_id: q for q in questions}
    assert set(by_id) == {"S1-influenza-001", "S1-mmr-001"}
    flu = by_id["S1-influenza-001"]
    assert flu.correct == ["B"]
    assert flu.options["B"] == "Everyone 6 months and older"
    assert flu.source_url == "http://x"  # header ' url' is stripped
    assert flu.multi_answer is False
    # Lowercase answer letters are accepted and category typos fixed.
    mmr = by_id["S1-mmr-001"]
    assert mmr.correct == ["A"]
    assert mmr.category == "Effectiveness and benefits"
    # Blank row skipped silently; row with answer D but only two options excluded with a reason.
    assert len(exclusions) == 1
    assert exclusions[0].row == 3
    assert "not among options" in exclusions[0].reason


def test_s2_multi_answer_and_boolean_options(s2_xlsx: Path):
    questions, exclusions = load_question_file(s2_xlsx, "S2", multi_answer=True)
    assert [q.question_id for q in questions] == ["S2-hepatitis-b-001", "S2-hepatitis-b-002"]
    first, second = questions
    assert first.correct == ["A", "B"]
    assert first.multi_answer is True
    assert first.source == "CDC"
    assert second.options == {"A": "True", "B": "False", "C": "Don't know"}
    assert second.correct == ["B"]  # blank 'answer2' cell (' ') ignored
    assert len(exclusions) == 1 and "empty option" in exclusions[0].reason


def test_load_question_bank_combines_files(s1_xlsx: Path, s2_xlsx: Path):
    questions, exclusions = load_question_bank(
        s1_xlsx.parent, {"S1": s1_xlsx.name, "S2": s2_xlsx.name}, multi_answer_subsets=["S2"]
    )
    assert len(questions) == 4
    assert len(exclusions) == 2
    assert {q.subset for q in questions} == {"S1", "S2"}


def test_normalizers():
    assert normalize_vaccine("Hepatits A") == "Hepatitis A"
    assert normalize_vaccine("Pneumonia") == normalize_vaccine("Pneumococcus") == "Pneumococcal"
    assert normalize_vaccine("Unknown sheet ") == "Unknown sheet"
    assert normalize_category("Misconceptions or concerns") == "Misconceptions and corrections"
    assert normalize_category(None) is None


def test_detect_format_rejects_unknown_columns():
    assert detect_format(["Question", "Option A"]) == "s1"
    assert detect_format(["Question", "Option1"]) == "s2"
    with pytest.raises(ValueError):
        detect_format(["Question", "Choice 1"])
