"""Core data models shared across the pipeline.

All records that are written to disk (questions, chunks, retrieval results,
LLM call records) are defined here as pydantic models so that every stage reads
and writes the same, validated structure.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Answer letters used for options. S1 has 4 options, S2 has up to 7.
OPTION_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

Arm = Literal["A0", "A1", "A2"]


class Question(BaseModel):
    """A single multiple-choice question in normalized form."""

    question_id: str
    subset: str = Field(description="Source file of the question, e.g. 'S1' or 'S2'.")
    vaccine: str
    stem: str
    options: dict[str, str] = Field(description="Ordered mapping from letter to option text.")
    correct: list[str] = Field(description="Sorted list of correct letters.")
    multi_answer: bool = Field(
        description="True if the model must select all correct options (S2 format)."
    )
    category: str | None = None
    source: str | None = None
    source_url: str | None = None
    evidence: str | None = None
    # Fields filled from the manual annotation file (None until annotated).
    question_type: str | None = None
    population: str | None = None
    answer_source: str | None = None
    kb_covered: bool | None = None
    version_status: str | None = None


def document_label(title: str, year: str | None) -> str:
    """Document part of a provenance label: the title followed by its year, if known.

    The year lets the model tell newer from older guidance when several current
    documents cover the same vaccine.
    """
    return f"{title} ({year})" if year else title


class Block(BaseModel):
    """A structural element of a parsed guideline document.

    Parsing (Docling or overrides) produces a flat, ordered list of blocks; the
    chunker consumes blocks and never touches the PDF parser directly.
    """

    doc_id: str
    kind: Literal["heading", "text", "table"]
    text: str = ""
    level: int = 0
    page: int | None = None
    # Table-only fields.
    caption: str | None = None
    header: list[str] | None = None
    rows: list[list[str]] | None = None
    footnotes: list[str] = Field(default_factory=list)
    table_source: Literal["docling", "vision"] | None = None


class Chunk(BaseModel):
    """A retrievable unit of the knowledge base."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: list[str]
    kind: Literal["text", "table"]
    doc_year: str | None = Field(
        default=None, description="Publication year shown in the provenance label."
    )
    text: str = Field(description="Content shown to the LLM (tables as Markdown).")
    retrieval_text: str = Field(description="Content indexed by the retrievers.")
    page_start: int | None = None
    page_end: int | None = None
    n_tokens: int = 0
    table_source: Literal["docling", "vision"] | None = Field(
        default=None, description="For table chunks: Docling output or multimodal override."
    )

    @property
    def label(self) -> str:
        """Human-readable provenance label, e.g. 'Doc title (2021) › Section › Sub'."""
        return " › ".join([document_label(self.doc_title, self.doc_year), *self.section_path])


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever together with its scores."""

    chunk_id: str
    rank: int
    score: float
    stage_scores: dict[str, float] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    """Retrieval output for one question under one retrieval configuration."""

    question_id: str
    retrieval_config_hash: str
    query: str
    results: list[RetrievedChunk]


class Reference(BaseModel):
    """A citation produced by the model."""

    source_id: str
    quote: str


class ModelAnswer(BaseModel):
    """The structured answer the model must return."""

    answer: list[str]
    references: list[Reference] = Field(default_factory=list)
    answer_basis: Literal["provided_context", "own_knowledge"]
    justification: str = ""


class LLMResult(BaseModel):
    """Provider-independent result of one LLM request."""

    text: str
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    latency_s: float | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class CallRecord(BaseModel):
    """One line of `calls.jsonl`: everything about a single LLM call attempt."""

    # Identification
    call_id: str
    attempt: int
    question_id: str
    model: str
    model_version: str | None = None
    arm: str
    rep: int
    prompt_version: str
    git_commit: str
    run_id: str
    timestamp: str
    # Input
    prompt_system: str
    prompt_user: str
    context_chunk_ids: list[str] = Field(default_factory=list)
    context_labels: dict[str, str] = Field(default_factory=dict)
    generation_settings: dict[str, Any] = Field(default_factory=dict)
    # Output
    raw_response: str | None = None
    parsed: ModelAnswer | None = None
    answer: list[str] = Field(default_factory=list, description="Selected option letters.")
    status: Literal["ok", "parse_error", "api_error", "refusal"]
    error: str | None = None
    finish_reason: str | None = None
    # A3 only: documents the model selected (see `DocumentSelection`).
    selected_doc_ids: list[str] = Field(default_factory=list)
    # Cost (input_tokens includes cached tokens)
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    latency_s: float | None = None
    cost_usd: float | None = None


class DocumentSelection(BaseModel):
    """One document-selection attempt for arm A3 (`selections.jsonl`)."""

    call_id: str
    attempt: int
    question_id: str
    model: str
    model_version: str | None = None
    prompt_version: str
    status: Literal["ok", "parse_error", "api_error", "refusal"]
    doc_ids: list[str] = Field(default_factory=list)
    raw_response: str | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    timestamp: str
