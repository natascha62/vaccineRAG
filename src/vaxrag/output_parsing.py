"""Parse and validate the model's JSON answer."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from vaxrag.schemas import ModelAnswer

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


def extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object in `text` (handles code fences)."""
    text = _FENCE_RE.sub("", text.strip())
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_model_answer(
    text: str, valid_letters: list[str], multi_answer: bool
) -> tuple[ModelAnswer | None, str | None]:
    """Parse model output into a `ModelAnswer`.

    Returns (answer, None) on success or (None, error message) on failure.
    Answer letters are upper-cased and de-duplicated; a single-answer question
    must have exactly one letter, a multi-answer question at least one.
    A bare string answer ("B") is accepted and converted to a list.
    """
    payload = extract_json_object(text or "")
    if payload is None:
        return None, "no JSON object found"
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if isinstance(data.get("answer"), str):
        data["answer"] = [a for a in re.split(r"[\s,;]+", data["answer"]) if a]
    try:
        answer = ModelAnswer.model_validate(data)
    except ValidationError as exc:
        return None, f"schema violation: {exc.errors()[0]['msg']}"
    letters = []
    for letter in answer.answer:
        letter = letter.strip().upper().rstrip(".)")
        if letter not in letters:
            letters.append(letter)
    invalid = [x for x in letters if x not in valid_letters]
    if invalid:
        return None, f"invalid option letters: {invalid}"
    if not letters:
        return None, "empty answer"
    if not multi_answer and len(letters) != 1:
        return None, f"expected exactly one option, got {letters}"
    answer.answer = sorted(letters)
    return answer, None


def parse_selection(
    text: str, valid_ids: list[str], max_docs: int
) -> tuple[list[str] | None, str | None]:
    """Parse the A3 document selection: 1 to `max_docs` distinct, known document IDs.

    Returns (doc_ids in the model's order, None) or (None, error message).
    """
    payload = extract_json_object(text or "")
    if payload is None:
        return None, "no JSON object found"
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    raw = data.get("doc_ids")
    if not isinstance(raw, list):
        return None, "schema violation: doc_ids must be a list"
    doc_ids: list[str] = []
    for item in raw:
        doc_id = str(item).strip()
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    unknown = [d for d in doc_ids if d not in valid_ids]
    if unknown:
        return None, f"unknown document IDs: {unknown}"
    if not doc_ids:
        return None, "no document selected"
    if len(doc_ids) > max_docs:
        return None, f"selected {len(doc_ids)} documents, at most {max_docs} allowed"
    return doc_ids, None
