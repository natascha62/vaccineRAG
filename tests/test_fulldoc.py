"""Tests for arm A3: model-selected full PDF documents (no retrieval)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from conftest import REPO_ROOT, FakeClient, make_chunk, make_question

from vaxrag.analysis import stats
from vaxrag.analysis.collect import build_analysis_table
from vaxrag.config import Config, ModelConfig
from vaxrag.data.manifest import ManifestEntry
from vaxrag.evaluation.citations import excerpt_around
from vaxrag.evaluation.judge import format_evidence
from vaxrag.io_utils import read_jsonl
from vaxrag.llm.base import LLMError, LLMRequest, PdfDocument
from vaxrag.llm.factory import create_client
from vaxrag.output_parsing import parse_selection
from vaxrag.prompting import build_answer_prompt, build_selection_prompt
from vaxrag.prompts import load_prompt, load_prompt_parts
from vaxrag.runner import load_run_context, plan_calls, run_experiment
from vaxrag.schemas import CallRecord, LLMResult

PROMPTS = REPO_ROOT / "prompts"


def _entry(doc_id, title, doc_type, version_date, filename):
    return ManifestEntry(
        doc_id=doc_id,
        publisher="CDC",
        title=title,
        doc_type=doc_type,
        version_date=version_date,
        filename=filename,
    )


def _entries():
    return [
        _entry("acip_mmr_2013", "MMR", "vaccine_recommendation", "2013-06-14", "mmr.pdf"),
        _entry("acip_hpv_2019", "HPV", "vaccine_recommendation", "2019-08-16", "hpv.pdf"),
        _entry("schedule_child_2026", "Child schedule", "schedule", "2026-01", "child.pdf"),
    ]


def _a3_config(project: Config) -> Config:
    project.experiment.arms = ["A3"]
    project.experiment.fulldoc.models = ["fake"]
    project.models["fake"].supports_pdf = True
    return project


def _context(project: Config, tmp_path: Path, questions):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for e in _entries():
        (raw / e.filename).write_bytes(f"%PDF {e.doc_id}".encode())
    return load_run_context(
        project, "a3", questions, [], None, {}, documents=_entries(), raw_dir=raw
    )


def _responder(selection: list[str], answer=("B",)):
    def respond(req: LLMRequest):
        if req.schema_name == "doc_selection":
            return json.dumps({"doc_ids": selection, "rationale": "r"})
        return json.dumps(
            {"answer": list(answer), "references": [{"source_id": "D1", "quote": "two doses"}],
             "answer_basis": "provided_context", "justification": ""}
        )  # fmt: skip

    return respond


# ----------------------------------------------------------------------------- prompts & parsing
def test_answer_prompt_lists_attached_documents():
    system_t, user_t = load_prompt(PROMPTS, "answer_v1")
    parts = load_prompt_parts(PROMPTS, "answer_v1")
    docs = [("acip_hpv_2019", "HPV (2019)"), ("acip_mmr_2013", "MMR (2013)")]
    _, user, mapping = build_answer_prompt(
        make_question(), None, system_t, user_t, parts, documents=docs
    )
    assert mapping == {"D1": "acip_hpv_2019", "D2": "acip_mmr_2013"}
    assert "[D1] HPV (2019)\n[D2] MMR (2013)" in user
    assert "attached PDF documents" in user and '"D2"' in user
    with pytest.raises(ValueError):
        build_answer_prompt(make_question(), None, system_t, user_t, parts, documents=[])
    with pytest.raises(KeyError, match="fulldoc"):
        build_answer_prompt(make_question(), None, system_t, user_t, {}, documents=docs)


def test_selection_prompt_shows_catalog_and_limit():
    system_t, user_t = load_prompt(PROMPTS, "doc_selection_v1")
    catalog = [("acip_hpv_2019", "HPV (2019)", "vaccine_recommendation")]
    system, user = build_selection_prompt(make_question(), catalog, 3, system_t, user_t)
    assert "acip_hpv_2019 | HPV (2019) | vaccine_recommendation" in user
    assert "at most 3" in system and "at most 3" in user
    assert "A. alpha" in user and "{{" not in user + system


@pytest.mark.parametrize(
    ("payload", "expected", "error"),
    [
        ({"doc_ids": ["b", "a", "b"]}, ["b", "a"], None),
        ({"doc_ids": ["a", "b", "c", "d"]}, None, "at most 3"),
        ({"doc_ids": []}, None, "no document"),
        ({"doc_ids": ["zzz"]}, None, "unknown"),
        ({"doc_ids": "a"}, None, "must be a list"),
    ],
)
def test_parse_selection(payload, expected, error):
    doc_ids, err = parse_selection(json.dumps(payload), ["a", "b", "c", "d"], max_docs=3)
    if expected is None:
        assert doc_ids is None and error in err
    else:
        assert doc_ids == expected and err is None
    assert parse_selection("nonsense", ["a"], 3)[0] is None


# ----------------------------------------------------------------------------- planning & config
def test_a3_planned_only_for_fulldoc_models_and_not_repeated(project: Config):
    project.experiment.arms = ["A0", "A3"]
    project.experiment.fulldoc.models = ["fake"]
    project.experiment.consistency.enabled = True
    project.experiment.consistency.n_questions = 1
    qs = [make_question("q1")]
    specs = {(s.model, s.arm, s.rep) for s in plan_calls(qs, ["fake", "fake2"], project)}
    assert ("fake", "A3", 1) in specs and ("fake2", "A3", 1) not in specs
    assert ("fake", "A3", 2) not in specs  # consistency.arms excludes A3
    assert ("fake2", "A0", 3) in specs


def test_config_requires_pdf_support_for_a3():
    with pytest.raises(ValueError, match="supports_pdf"):
        Config.model_validate(
            {
                "models": {
                    "m": {"provider": "vllm", "model_id": "x", "training_data_cutoff": "2025"}
                },
                "experiment": {"models": ["m"], "arms": ["A3"], "fulldoc": {"models": ["m"]}},
            }
        )


# ----------------------------------------------------------------------------- runner
def test_a3_selects_then_answers_with_sorted_pdfs(project: Config, tmp_path: Path):
    project = _a3_config(project)
    qs = [make_question("q1")]
    ctx = _context(project, tmp_path, qs)
    client = FakeClient(responder=_responder(["schedule_child_2026", "acip_hpv_2019"]))
    assert run_experiment(ctx, {"fake": client}, qs) == {"ok": 1}
    selection_req, answer_req = client.requests
    assert selection_req.schema_name == "doc_selection" and not selection_req.documents
    assert "acip_mmr_2013 | MMR (2013) | vaccine_recommendation" in selection_req.user
    # PDFs are attached in doc_id order for prompt caching, with year labels.
    assert [d.title for d in answer_req.documents] == ["HPV (2019)", "Child schedule (2026)"]
    assert answer_req.documents[0].data == b"%PDF acip_hpv_2019"
    assert answer_req.cache_documents is True
    (row,) = read_jsonl(ctx.calls_path)
    assert row["selected_doc_ids"] == ["schedule_child_2026", "acip_hpv_2019"]
    assert row["context_labels"] == {"D1": "acip_hpv_2019", "D2": "schedule_child_2026"}
    assert row["context_chunk_ids"] == []
    (sel,) = read_jsonl(ctx.selections_path)
    assert sel["status"] == "ok" and sel["cost_usd"] > 0


def test_a3_resume_reuses_selection(project: Config, tmp_path: Path):
    project = _a3_config(project)
    qs = [make_question("q1")]
    ctx = _context(project, tmp_path, qs)
    failing = FakeClient(
        responder=lambda req: (
            json.dumps({"doc_ids": ["acip_hpv_2019"], "rationale": ""})
            if req.schema_name == "doc_selection"
            else LLMError("503")
        )
    )
    assert run_experiment(ctx, {"fake": failing}, qs) == {"api_error": 1}
    ctx2 = _context(project, tmp_path, qs)
    client = FakeClient(responder=_responder(["acip_mmr_2013"]))  # would select differently
    assert run_experiment(ctx2, {"fake": client}, qs) == {"ok": 1}
    assert [r.schema_name for r in client.requests] == ["mcq_answer"]
    assert client.requests[0].documents[0].title == "HPV (2019)"


def test_a3_failed_selection_is_final(project: Config, tmp_path: Path):
    project = _a3_config(project)
    qs = [make_question("q1")]
    ctx = _context(project, tmp_path, qs)
    bad = FakeClient(responder=lambda req: json.dumps({"doc_ids": ["nope"], "rationale": ""}))
    assert run_experiment(ctx, {"fake": bad}, qs) == {"parse_error": 1}
    assert len(list(read_jsonl(ctx.selections_path))) == 2  # two selection attempts
    assert run_experiment(_context(project, tmp_path, qs), {"fake": bad}, qs) == {}


def test_a3_skipped_without_manifest(project: Config, tmp_path: Path):
    project = _a3_config(project)
    qs = [make_question("q1")]
    ctx = load_run_context(project, "a3", qs, [], None, {})
    assert run_experiment(ctx, {"fake": FakeClient()}, qs) == {}


# ----------------------------------------------------------------------------- clients
@pytest.fixture
def keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "test-key")


def _client(provider, **kw):
    cfg = Config.model_validate({"models": {"m": {"provider": provider, "model_id": "x", **kw}}})
    return create_client(cfg, "m")


PDFS = [PdfDocument(title="HPV (2019)", filename="hpv.pdf", data=b"%PDF-1")]


def test_anthropic_attaches_cached_documents(keys):
    params = _client("anthropic").build_params(
        LLMRequest(system="s", user="u", documents=PDFS, cache_documents=True)
    )
    doc, text = params["messages"][0]["content"]
    assert doc["type"] == "document" and doc["title"] == "HPV (2019)"
    assert doc["source"]["media_type"] == "application/pdf"
    assert doc["cache_control"] == {"type": "ephemeral"}
    assert text == {"type": "text", "text": "u"}
    uncached = _client("anthropic").build_params(LLMRequest(system="s", user="u", documents=PDFS))
    assert "cache_control" not in uncached["messages"][0]["content"][0]


def test_anthropic_usage_includes_cache_tokens(keys, monkeypatch):
    client = _client("anthropic", price_input_per_mtok=5, price_output_per_mtok=25,
                     price_cache_read_per_mtok=0.5, price_cache_write_per_mtok=6.25)  # fmt: skip
    usage = SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=1000,
                            cache_creation_input_tokens=2000)  # fmt: skip
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")], model="m",
                               stop_reason="end_turn", usage=usage)  # fmt: skip
    monkeypatch.setattr(client.client.messages, "create", lambda **kw: response)
    result = client.generate(LLMRequest(system="s", user="u"))
    assert (result.input_tokens, result.cache_read_tokens, result.cache_write_tokens) == (
        3100, 1000, 2000
    )  # fmt: skip
    expected = (100 * 5 + 1000 * 0.5 + 2000 * 6.25 + 10 * 25) / 1e6
    assert client.cost_usd(result) == pytest.approx(expected)


def test_openai_attaches_files_before_text(keys):
    params = _client("openai").build_params(LLMRequest(system="s", user="u", documents=PDFS))
    file_part, text_part = params["messages"][1]["content"]
    assert file_part["type"] == "file" and file_part["file"]["filename"] == "hpv.pdf"
    assert file_part["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert text_part == {"type": "text", "text": "u"}


def test_gemini_contents_start_with_pdfs(keys):
    contents = _client("google").build_contents(LLMRequest(system="s", user="u", documents=PDFS))
    assert contents[0].inline_data.mime_type == "application/pdf" and contents[-1] == "u"


def test_vllm_rejects_documents(keys):
    with pytest.raises(ValueError, match="cannot read PDF"):
        _client("vllm", base_url="http://x/v1").build_params(
            LLMRequest(system="s", user="u", documents=PDFS)
        )


def test_cost_without_cache_prices_uses_input_price():
    client = FakeClient()
    client.config = ModelConfig(provider="openai", model_id="x", price_input_per_mtok=2,
                                price_output_per_mtok=10)  # fmt: skip
    result = LLMResult(text="", input_tokens=1000, output_tokens=0, cache_read_tokens=400)
    assert client.cost_usd(result) == pytest.approx(1000 * 2 / 1e6)


# ----------------------------------------------------------------------------- evaluation
def test_excerpt_around_finds_quote():
    text = "x" * 3000 + " Two doses of MMR are recommended. " + "y" * 3000
    excerpt = excerpt_around("two doses of mmr", text, window=20)
    assert "Two doses of MMR" in excerpt and excerpt.startswith("... ") and excerpt.endswith(" ...")
    assert excerpt_around("", text) == ""


def _a3_record(**kw):
    base = dict(
        call_id="q1|m|A3|r1|v", attempt=1, question_id="q1", model="m", arm="A3", rep=1,
        prompt_version="v", git_commit="g", run_id="r", timestamp="t", prompt_system="",
        prompt_user="", context_labels={"D1": "acip_hpv_2019"}, status="ok",
        selected_doc_ids=["acip_hpv_2019"], answer=["B"],
        parsed={"answer": ["B"], "references": [{"source_id": "D1", "quote": "two doses"}],
                "answer_basis": "provided_context", "justification": ""},
        cost_usd=0.10,
    )  # fmt: skip
    base.update(kw)
    return CallRecord(**base)


def test_grounding_evidence_uses_document_excerpt():
    record = _a3_record()
    evidence = format_evidence(record, {}, {"acip_hpv_2019": "intro. Give two doses. end"})
    assert "Give two doses" in evidence and "cites D1" in evidence
    assert "does not exist" in format_evidence(record, {}, {})


def test_collect_a3_columns(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "calls.jsonl").write_text(_a3_record().model_dump_json() + "\n")
    (run_dir / "selections.jsonl").write_text(
        json.dumps({"call_id": "q1|m|A3|r1|v", "status": "ok", "cost_usd": 0.01}) + "\n"
    )
    chunk = make_chunk("acip_hpv_2019::0001", doc_id="acip_hpv_2019")
    table = build_analysis_table(
        [run_dir], [make_question("q1")], [chunk], {"q1": [chunk.chunk_id]}, {},
        doc_texts={"acip_hpv_2019": "Give two doses."},
    )  # fmt: skip
    row = table.iloc[0]
    assert row["gold_doc_selected"] and row["n_selected_docs"] == 1
    assert row["share_quotes_verbatim"] == 1 and row["cost_usd"] == pytest.approx(0.11)
    assert pd.isna(row["gold_in_context_strict"]) and "citation_recall_strict" not in table


def test_doc_selection_table():
    df = pd.DataFrame(
        {
            "model": ["m"] * 4 + ["m"],
            "arm": ["A3"] * 4 + ["A1"],
            "n_selected_docs": [1, 2, 3, 2, None],
            "gold_doc_selected": [True, True, False, None, None],
            "correct": [True, False, False, True, True],
        }
    )
    row = stats.doc_selection_table(df).iloc[0]
    assert row["n"] == 4 and row["mean_docs_selected"] == 2
    assert row["gold_doc_selected_rate"] == pytest.approx(2 / 3)
    assert row["accuracy_gold_selected"] == 0.5 and row["accuracy_gold_not_selected"] == 0
