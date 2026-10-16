"""Shared test fixtures: tiny synthetic question files, chunks, configs and a fake LLM."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from vaxrag.config import Config, ModelConfig
from vaxrag.llm.base import LLMClient, LLMRequest
from vaxrag.schemas import Chunk, LLMResult, Question

REPO_ROOT = Path(__file__).resolve().parents[1]


def word_count(text: str) -> int:
    """Deterministic token counter for tests (whitespace words)."""
    return len(text.split())


@pytest.fixture
def s1_xlsx(tmp_path: Path) -> Path:
    """S1-format workbook with two sheets, one blank row and one invalid row."""
    path = tmp_path / "s1.xlsx"
    cols = [
        None,
        "Question",
        "Option A",
        "Option B",
        "Option C",
        "Option D",
        "Answer",
        "Evidence",
        "source",
        " url",
        "Category",
    ]
    flu = pd.DataFrame(
        [
            [
                0,
                "Who should get a flu vaccine?",
                "Nobody",
                "Everyone 6 months and older",
                "Only adults",
                "Only children",
                "B",
                "Everyone 6 months and older...",
                "CDC",
                "http://x",
                "Dosing and recommendations",
            ],
            [1, None, None, None, None, None, None, None, None, None, None],
            [2, "Bad row", "x", "y", None, None, "D", None, None, None, "Other"],
        ],
        columns=cols,
    )
    mmr = pd.DataFrame(
        [
            [
                0,
                "MMR dose 2 minimum interval?",
                "4 weeks",
                "1 week",
                "1 year",
                "6 months",
                "a",
                None,
                "CDC",
                None,
                "Effectivness and benefits",
            ]
        ],
        columns=cols,
    )
    with pd.ExcelWriter(path) as writer:
        flu.to_excel(writer, sheet_name="Flu", index=False)
        mmr.to_excel(writer, sheet_name="MMR", index=False)
    return path


@pytest.fixture
def s2_xlsx(tmp_path: Path) -> Path:
    """S2-format workbook with multi-answer, boolean options and an invalid answer."""
    path = tmp_path / "s2.xlsx"
    cols = [
        "filename",
        "index",
        "evidence_resource",
        "evidence_url",
        "Question",
        "Option1",
        "Option2",
        "Option3",
        "Option4",
        "answer1",
        "answer2",
        "Category",
    ]
    df = pd.DataFrame(
        [
            [
                None,
                None,
                "CDC",
                "http://y",
                "Vaccines are safe.",
                "Strongly agree",
                "Somewhat agree",
                "Unsure",
                "Disagree",
                "Option1",
                "Option2",
                "Safety and side effects",
            ],
            [
                None,
                None,
                None,
                None,
                "Pap smears are unnecessary after HPV vaccination.",
                True,
                False,
                "Don't know",
                None,
                "Option2",
                " ",
                "Awareness, behavior, and prevention",
            ],
            [None, None, None, None, "Broken", "Yes", "No", None, None, "Option4", None, "Other"],
        ],
        columns=cols,
    )
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Hepatits B", index=False)
    return path


def make_question(
    qid: str = "S1-flu-001", correct: list[str] | None = None, multi: bool = False
) -> Question:
    """Construct a simple 4-option question."""
    return Question(
        question_id=qid,
        subset="S2" if multi else "S1",
        vaccine="Influenza",
        stem=f"Question {qid}?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        correct=correct or ["B"],
        multi_answer=multi,
    )


def make_chunk(
    cid: str,
    doc_id: str = "doc1",
    text: str = "",
    section: list[str] | None = None,
    kind: str = "text",
    pages: tuple[int, int] | None = (1, 1),
) -> Chunk:
    """Construct a chunk whose retrieval text equals its text."""
    body = text or f"content of {cid}"
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        doc_title=f"Title {doc_id}",
        section_path=section or ["Intro"],
        kind=kind,
        text=body,
        retrieval_text=body,
        page_start=pages[0] if pages else None,
        page_end=pages[1] if pages else None,
        n_tokens=word_count(body),
    )


class FakeClient(LLMClient):
    """Scripted LLM client: returns queued responses or a callable's output."""

    def __init__(self, key: str = "fake", responder=None) -> None:
        super().__init__(
            key,
            ModelConfig(
                provider="openai",
                model_id="fake-1",
                price_input_per_mtok=1.0,
                price_output_per_mtok=2.0,
            ),
        )
        self.responder = responder or (
            lambda req: json.dumps(
                {
                    "answer": ["B"],
                    "references": [],
                    "answer_basis": "own_knowledge",
                    "justification": "x",
                }
            )
        )
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        out = self.responder(request)
        if isinstance(out, Exception):
            raise out
        return LLMResult(
            text=out,
            model_version="fake-1-2026",
            input_tokens=100,
            output_tokens=10,
            latency_s=0.01,
            finish_reason="stop",
        )


@pytest.fixture
def project(tmp_path: Path) -> Config:
    """A config rooted in a temporary project folder with the real prompt files."""
    shutil.copytree(REPO_ROOT / "prompts", tmp_path / "prompts")
    return Config.model_validate(
        {
            "project_root": str(tmp_path),
            "models": {
                "fake": {
                    "provider": "openai",
                    "model_id": "fake-1",
                    "training_data_cutoff": "2025-01",
                },
                "fake2": {
                    "provider": "openai",
                    "model_id": "fake-2",
                    "training_data_cutoff": "2025-06-30",
                },
                "judge": {"provider": "openai", "model_id": "judge-1"},
            },
            "experiment": {
                "models": ["fake"],
                "max_workers": 2,
                "consistency": {"enabled": False},
            },
            "analysis": {"bootstrap_iterations": 200, "paired_models": None},
        }
    )
