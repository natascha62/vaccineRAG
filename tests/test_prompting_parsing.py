import json

import pytest
from conftest import REPO_ROOT, make_chunk, make_question

from vaxrag.output_parsing import extract_json_object, parse_model_answer
from vaxrag.prompting import (
    REQUIRED_PARTS,
    answer_schema,
    build_answer_prompt,
)
from vaxrag.prompts import load_prompt, load_prompt_parts, render

PROMPTS = REPO_ROOT / "prompts"


def _prompt(question, chunks):
    system_t, user_t = load_prompt(PROMPTS, "answer_v1")
    parts = load_prompt_parts(PROMPTS, "answer_v1")
    return build_answer_prompt(question, chunks, system_t, user_t, parts)


def test_render_placeholders():
    assert render("a {{x}} {not} {{ y }}", x="1", y="2") == "a 1 {not} 2"
    with pytest.raises(KeyError):
        render("{{missing}}")


def test_answer_v1_parts_complete():
    assert set(REQUIRED_PARTS) <= set(load_prompt_parts(PROMPTS, "answer_v1"))


def test_closed_book_prompt():
    system, user, mapping = _prompt(make_question(), None)
    assert mapping == {}
    assert "Reference material" not in user
    assert "from memory" in user
    assert "A. alpha" in user and "Select exactly one option" in user
    assert "{{" not in user and "JSON" in system


def test_context_prompt_numbering_and_multi_instruction():
    q = make_question(multi=True)
    chunks = [
        make_chunk("c1", text="Doc › Sec\n\nfirst"),
        make_chunk("c2", text="Doc › Sec2\n\nsecond"),
    ]
    _, user, mapping = _prompt(q, chunks)
    assert mapping == {"S1": "c1", "S2": "c2"}
    assert "[S1] Doc › Sec\n\nfirst" in user and "[S2] Doc › Sec2" in user
    assert user.index("[S2]") < user.index("Question:")
    assert "Select ALL options" in user
    with pytest.raises(ValueError):
        _prompt(q, [])


def test_answer_schema_shape():
    schema = answer_schema(["A", "B"])
    assert schema["properties"]["answer"]["items"]["enum"] == ["A", "B"]
    assert schema["required"] == ["answer", "references", "answer_basis", "justification"]
    assert schema["additionalProperties"] is False


def _payload(**kw):
    base = {
        "answer": ["B"],
        "references": [{"source_id": "S1", "quote": "q"}],
        "answer_basis": "provided_context",
        "justification": "j",
    }
    base.update(kw)
    return json.dumps(base)


def test_extract_json_object_handles_fences_and_braces_in_strings():
    text = 'Sure:\n```json\n{"a": "x}y", "b": {"c": 1}}\n```'
    assert json.loads(extract_json_object(text)) == {"a": "x}y", "b": {"c": 1}}
    assert extract_json_object("no json") is None


@pytest.mark.parametrize(
    ("text", "multi", "expected", "error"),
    [
        (_payload(), False, ["B"], None),
        (_payload(answer="b"), False, ["B"], None),
        (_payload(answer=["C", "a", "C"]), True, ["A", "C"], None),
        (_payload(answer=["A", "B"]), False, None, "exactly one"),
        (_payload(answer=["E"]), False, None, "invalid option"),
        (_payload(answer=[]), True, None, "empty"),
        (_payload(answer_basis="guess"), False, None, "schema"),
        ("not json", False, None, "no JSON"),
        ('{"answer": ["B"],}', False, None, "invalid JSON"),
    ],
)
def test_parse_model_answer(text, multi, expected, error):
    parsed, err = parse_model_answer(text, ["A", "B", "C", "D"], multi)
    if expected is None:
        assert parsed is None and error in err
    else:
        assert err is None and parsed.answer == expected
