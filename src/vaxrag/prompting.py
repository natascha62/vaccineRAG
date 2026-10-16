"""Build answer prompts and the JSON output schema for the MCQ task.

All arms share one zero-shot template. They differ only in the reference block:
A0 has none (and asks the model to cite guidelines from memory), A1 and A2 list
numbered excerpts `[S1]`, `[S2]`, ... each starting with its provenance line,
and A3 attaches whole PDFs (labelled `[D1]`, `[D2]`, ...) that the model chose
in a preceding document-selection step (see `build_selection_prompt`).
"""

from __future__ import annotations

from typing import Any

from vaxrag.prompts import render
from vaxrag.schemas import Chunk, Question

# Keys that every `<version>.parts.yaml` file for the answer prompt must define.
REQUIRED_PARTS = (
    "single_instruction",
    "multi_instruction",
    "context_header",
    "context_citation_instruction",
    "no_context_citation_instruction",
)
# Additional keys needed by the full-document arm (A3).
FULLDOC_PARTS = ("fulldoc_context_header", "fulldoc_citation_instruction")


def answer_schema(letters: list[str]) -> dict[str, Any]:
    """JSON schema for the model answer, restricted to the presented option letters.

    The schema only uses keywords supported by every provider's structured
    output mode (no minItems/maxItems); answer counts are validated afterwards.
    """
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "array", "items": {"type": "string", "enum": list(letters)}},
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["source_id", "quote"],
                    "additionalProperties": False,
                },
            },
            "answer_basis": {"type": "string", "enum": ["provided_context", "own_knowledge"]},
            "justification": {"type": "string"},
        },
        "required": ["answer", "references", "answer_basis", "justification"],
        "additionalProperties": False,
    }


def format_context(chunks: list[Chunk]) -> tuple[str, dict[str, str]]:
    """Render chunks as numbered excerpts; returns (text, source_id -> chunk_id)."""
    excerpts = []
    mapping: dict[str, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        sid = f"S{i}"
        mapping[sid] = chunk.chunk_id
        excerpts.append(f"[{sid}] {chunk.text}")
    return "\n\n".join(excerpts), mapping


def build_answer_prompt(
    question: Question,
    chunks: list[Chunk] | None,
    system_template: str,
    user_template: str,
    parts: dict[str, str],
    documents: list[tuple[str, str]] | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Render (system, user, source_id -> chunk_id or doc_id) for one question.

    `chunks=None` means the closed-book arm (A0); an empty list is not allowed
    for context arms because that would silently turn them into A0. `documents`
    (A3) lists `(doc_id, label)` of the attached PDFs in attachment order; it
    takes precedence over `chunks`. `parts` holds the instruction snippets from
    `<version>.parts.yaml`.
    """
    required = REQUIRED_PARTS + (FULLDOC_PARTS if documents is not None else ())
    missing = [k for k in required if k not in parts]
    if missing:
        raise KeyError(f"Prompt parts missing: {missing}")
    options = "\n".join(f"{letter}. {text}" for letter, text in question.options.items())
    if documents is not None:
        if not documents:
            raise ValueError(f"No documents for {question.question_id}")
        mapping = {f"D{i}": doc_id for i, (doc_id, _) in enumerate(documents, start=1)}
        listing = "\n".join(f"[D{i}] {label}" for i, (_, label) in enumerate(documents, start=1))
        context_block = f"{parts['fulldoc_context_header']}\n\n{listing}\n\n---\n\n"
        citation = parts["fulldoc_citation_instruction"]
    elif chunks is None:
        context_block = ""
        citation = parts["no_context_citation_instruction"]
        mapping: dict[str, str] = {}
    else:
        if not chunks:
            raise ValueError(f"No context chunks for {question.question_id}")
        excerpts, mapping = format_context(chunks)
        context_block = f"{parts['context_header']}\n\n{excerpts}\n\n---\n\n"
        citation = parts["context_citation_instruction"]
    user = render(
        user_template,
        context_block=context_block,
        stem=question.stem,
        options=options,
        selection_instruction=(
            parts["multi_instruction"] if question.multi_answer else parts["single_instruction"]
        ),
        citation_instruction=citation,
    )
    return system_template, user, mapping


def selection_schema(doc_ids: list[str]) -> dict[str, Any]:
    """JSON schema for the document-selection step (counts are validated afterwards)."""
    return {
        "type": "object",
        "properties": {
            "doc_ids": {"type": "array", "items": {"type": "string", "enum": list(doc_ids)}},
            "rationale": {"type": "string"},
        },
        "required": ["doc_ids", "rationale"],
        "additionalProperties": False,
    }


def format_catalog(catalog: list[tuple[str, str, str]]) -> str:
    """Render the document catalog as one line per document: `doc_id | label | type`."""
    return "\n".join(f"{doc_id} | {label} | {doc_type}" for doc_id, label, doc_type in catalog)


def build_selection_prompt(
    question: Question,
    catalog: list[tuple[str, str, str]],
    max_docs: int,
    system_template: str,
    user_template: str,
) -> tuple[str, str]:
    """Render (system, user) for the A3 document-selection step.

    `catalog` lists `(doc_id, label, doc_type)` for every knowledge-base document.
    The model sees the question and its options, so it can choose documents
    that cover the options.
    """
    options = "\n".join(f"{letter}. {text}" for letter, text in question.options.items())
    user = render(
        user_template,
        catalog=format_catalog(catalog),
        stem=question.stem,
        options=options,
        max_docs=str(max_docs),
    )
    return render(system_template, max_docs=str(max_docs)), user
