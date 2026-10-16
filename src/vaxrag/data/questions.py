"""Load the raw multiple-choice question bank (xlsx) into normalized `Question`s.

Two raw formats are supported:

* **S1** - one sheet per vaccine; columns `Question`, `Option A`-`Option D`,
  `Answer` (a single letter), `Evidence`, `source`, `url`, `Category`.
* **S2** - one sheet per vaccine; columns `Question`, `Option1`-`Option7`,
  `answer1`-`answer3` (values like `Option2`), `evidence_resource`,
  `evidence_url`, `Category`. Several answers can be correct; the model must
  select all of them.

Rows that cannot be converted into a valid question are not silently dropped:
they are returned as exclusions with a reason so they can be reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from vaxrag.schemas import OPTION_LETTERS, Question

# Sheet names in the raw files contain typos and synonyms; map them to one label.
VACCINE_NAME_MAP: dict[str, str] = {
    "covid-19": "COVID-19",
    "flu": "Influenza",
    "hpv": "HPV",
    "polio": "Polio",
    "rsv": "RSV",
    "mmr": "MMR",
    "hepatits a": "Hepatitis A",
    "hepatitis a": "Hepatitis A",
    "hepatits b": "Hepatitis B",
    "hepatitis b": "Hepatitis B",
    "shingles": "Zoster",
    "malaria": "Malaria",
    "diphtheria": "Diphtheria",
    "pneumococcus": "Pneumococcal",
    "pneumonia": "Pneumococcal",
    "meningococcal": "Meningococcal",
    "dengue": "Dengue",
}

# Category labels contain spelling variants; map them to the canonical label.
CATEGORY_MAP: dict[str, str] = {
    "effectivness and benefits": "Effectiveness and benefits",
    "misconceptions and concerns": "Misconceptions and corrections",
    "misconceptions or concerns": "Misconceptions and corrections",
}


@dataclass
class Exclusion:
    """A raw row that could not be converted into a question."""

    subset: str
    sheet: str
    row: int
    reason: str
    question: str | None


def _clean(value: Any) -> str | None:
    """Convert a cell value to a stripped string, mapping blanks/NaN to None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalize_vaccine(sheet_name: str) -> str:
    """Map a sheet name to a canonical vaccine label."""
    key = sheet_name.strip().lower()
    return VACCINE_NAME_MAP.get(key, sheet_name.strip())


def normalize_category(category: str | None) -> str | None:
    """Fix spelling variants of category labels."""
    if category is None:
        return None
    return CATEGORY_MAP.get(category.strip().lower(), category.strip())


def slugify(text: str) -> str:
    """Make a lowercase, dash-separated identifier fragment."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _read_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Read all sheets as strings-preserving DataFrames with stripped column names."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in sheets.items():
        df = df.rename(columns=lambda c: str(c).strip())
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
        cleaned[name] = df
    return cleaned


def parse_s1_row(
    row: dict[str, Any], subset: str, sheet: str, row_number: int, multi_answer: bool
) -> Question | Exclusion:
    """Convert one S1-format row (Option A-D, single letter answer)."""
    stem = _clean(row.get("Question"))
    if stem is None:
        return Exclusion(subset, sheet, row_number, "empty question", None)
    options: dict[str, str] = {}
    for letter in OPTION_LETTERS:
        text = _clean(row.get(f"Option {letter}"))
        if text is not None:
            options[letter] = text
    answer = (_clean(row.get("Answer")) or "").upper()
    correct = sorted({a for a in re.split(r"[\s,;/]+", answer) if a})
    if not correct:
        return Exclusion(subset, sheet, row_number, "missing answer", stem)
    if any(a not in options for a in correct):
        return Exclusion(subset, sheet, row_number, f"answer {correct} not among options", stem)
    vaccine = normalize_vaccine(sheet)
    return Question(
        question_id=f"{subset}-{slugify(vaccine)}-{row_number:03d}",
        subset=subset,
        vaccine=vaccine,
        stem=stem,
        options=options,
        correct=correct,
        multi_answer=multi_answer,
        category=normalize_category(_clean(row.get("Category"))),
        source=_clean(row.get("source")),
        source_url=_clean(row.get("url")),
        evidence=_clean(row.get("Evidence")),
    )


def parse_s2_row(
    row: dict[str, Any], subset: str, sheet: str, row_number: int, multi_answer: bool
) -> Question | Exclusion:
    """Convert one S2-format row (Option1-7, answers given as 'OptionN')."""
    stem = _clean(row.get("Question"))
    if stem is None:
        return Exclusion(subset, sheet, row_number, "empty question", None)
    options: dict[str, str] = {}
    number_to_letter: dict[int, str] = {}
    for number in range(1, len(OPTION_LETTERS) + 1):
        text = _clean(row.get(f"Option{number}"))
        if text is not None:
            letter = OPTION_LETTERS[len(options)]
            options[letter] = text
            number_to_letter[number] = letter
    correct_set: set[str] = set()
    for key in ("answer1", "answer2", "answer3", "answer4"):
        value = _clean(row.get(key))
        if value is None:
            continue
        match = re.fullmatch(r"option\s*(\d+)", value, flags=re.IGNORECASE)
        if match is None:
            return Exclusion(subset, sheet, row_number, f"unparseable answer '{value}'", stem)
        number = int(match.group(1))
        if number not in number_to_letter:
            return Exclusion(
                subset, sheet, row_number, f"answer {value} refers to empty option", stem
            )
        correct_set.add(number_to_letter[number])
    if not correct_set:
        return Exclusion(subset, sheet, row_number, "missing answer", stem)
    if len(options) < 2:
        return Exclusion(subset, sheet, row_number, "fewer than two options", stem)
    vaccine = normalize_vaccine(sheet)
    return Question(
        question_id=f"{subset}-{slugify(vaccine)}-{row_number:03d}",
        subset=subset,
        vaccine=vaccine,
        stem=stem,
        options=options,
        correct=sorted(correct_set),
        multi_answer=multi_answer,
        category=normalize_category(_clean(row.get("Category"))),
        source=_clean(row.get("evidence_resource")),
        source_url=_clean(row.get("evidence_url")),
        evidence=None,
    )


def detect_format(columns: list[str]) -> str:
    """Return 's1' or 's2' depending on which option columns a sheet has."""
    if "Option A" in columns:
        return "s1"
    if "Option1" in columns:
        return "s2"
    raise ValueError(f"Unrecognized question sheet format; columns: {columns}")


def load_question_file(
    path: str | Path, subset: str, multi_answer: bool
) -> tuple[list[Question], list[Exclusion]]:
    """Load every sheet of one xlsx file.

    Row numbers are 1-based positions among data rows of each sheet, so
    question IDs are stable as long as rows are not reordered in the source.
    """
    questions: list[Question] = []
    exclusions: list[Exclusion] = []
    for sheet, df in _read_sheets(Path(path)).items():
        parser = parse_s1_row if detect_format(list(df.columns)) == "s1" else parse_s2_row
        for position, row in enumerate(df.to_dict(orient="records"), start=1):
            if all(_clean(v) is None for v in row.values()):
                continue  # Fully blank spreadsheet rows are not questions.
            result = parser(row, subset, sheet, position, multi_answer)
            (questions if isinstance(result, Question) else exclusions).append(result)
    return questions, exclusions


def load_question_bank(
    questions_dir: str | Path, files: dict[str, str], multi_answer_subsets: list[str]
) -> tuple[list[Question], list[Exclusion]]:
    """Load all configured question files and verify that IDs are unique."""
    questions: list[Question] = []
    exclusions: list[Exclusion] = []
    for subset, filename in files.items():
        q, e = load_question_file(
            Path(questions_dir) / filename, subset, multi_answer=subset in multi_answer_subsets
        )
        questions.extend(q)
        exclusions.extend(e)
    ids = [q.question_id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {duplicates[:10]}")
    return questions, exclusions


def exclusions_frame(exclusions: list[Exclusion]) -> pd.DataFrame:
    """Return exclusions as a DataFrame for writing to CSV."""
    return pd.DataFrame([e.__dict__ for e in exclusions], columns=list(Exclusion.__annotations__))
