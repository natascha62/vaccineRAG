"""Tests for gold-independent evidence judgments (chunk relevance, context sufficiency)."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from conftest import FakeClient, make_chunk, make_question

from vaxrag.config import Config
from vaxrag.evaluation.evidence import (
    EvidenceTask,
    build_evidence_request,
    export_evidence_sample,
    judged_citation_support,
    judged_membership,
    lenient_relevant_sets,
    load_evidence_judgments,
    load_evidence_verdicts,
    relevance_key,
    relevance_pool,
    run_evidence_judgments,
    sufficiency_key,
    sufficiency_tasks,
)
from vaxrag.evaluation.judge import cohens_kappa
from vaxrag.evaluation.quadrants import evidence_flags, quadrant_table
from vaxrag.io_utils import read_jsonl
from vaxrag.prompts import load_prompt
from vaxrag.retrieval.evaluate import evaluate_retrieval, summarize_retrieval
from vaxrag.schemas import CallRecord, RetrievalRecord, RetrievedChunk


def _retrieval(qid, ids):
    return RetrievalRecord(
        question_id=qid,
        retrieval_config_hash="h",
        query="",
        results=[RetrievedChunk(chunk_id=c, rank=i + 1, score=1.0) for i, c in enumerate(ids)],
    )


def _record(qid, arm, labels, cited):
    return CallRecord(
        call_id=f"{qid}|{arm}", attempt=1, question_id=qid, model="m", arm=arm, rep=1,
        prompt_version="v", git_commit="g", run_id="r", timestamp="t",
        prompt_system="", prompt_user="", context_labels=labels, status="ok",
        parsed={"answer": ["B"], "references": [{"source_id": s, "quote": "x"} for s in cited],
                "answer_basis": "provided_context", "justification": ""},
    )  # fmt: skip


# ----------------------------------------------------------------------------- keys & requests
def test_keys_are_order_invariant_for_sufficiency():
    assert sufficiency_key("q1", ["b", "a"]) == sufficiency_key("q1", ["a", "b"])
    assert sufficiency_key("q1", ["a"]) != sufficiency_key("q2", ["a"])
    assert relevance_key("q1", "d::0001") == "q1|d::0001"


def test_request_shows_correct_answer_and_numbered_passages(project: Config):
    q = make_question("q1", correct=["A", "C"], multi=True)
    chunks = {"c1": make_chunk("c1", text="one"), "c2": make_chunk("c2", text="two")}
    system_t, user_t = load_prompt(project.path("prompts_dir"), "context_sufficiency_v1")
    req = build_evidence_request(
        "context_sufficiency", EvidenceTask("q1", ("c1", "c2")), q, chunks, system_t, user_t
    )
    assert "Correct answer (all of the listed options):\nA. alpha\nC. gamma" in req.user
    assert "[P1] one" in req.user and "[P2] two" in req.user
    assert req.json_schema["properties"]["verdict"]["enum"] == [
        "sufficient", "partial", "insufficient"
    ]  # fmt: skip
    system_t, user_t = load_prompt(project.path("prompts_dir"), "chunk_relevance_v1")
    single = build_evidence_request(
        "chunk_relevance", EvidenceTask("q1", ("c1",)), make_question("q1"), chunks, system_t,
        user_t,
    )  # fmt: skip
    assert "Correct answer (this option):\nB. beta" in single.user
    assert "{{" not in single.user


# ----------------------------------------------------------------------------- running & caching
def test_run_evidence_judgments_caches_per_judge_and_version(project: Config, tmp_path: Path):
    qs = {"q1": make_question("q1"), "q2": make_question("q2")}
    chunks = {f"c{i}": make_chunk(f"c{i}") for i in range(3)}
    out = tmp_path / "chunk_relevance.jsonl"
    judge = FakeClient("judge", lambda req: '{"verdict": "supports", "rationale": "r"}')
    tasks = [
        EvidenceTask("q1", ("c0",)),
        EvidenceTask("q1", ("c0",)),  # duplicate
        EvidenceTask("q2", ("c1",)),
        EvidenceTask("q2", ("missing",)),  # unknown chunk -> skipped
    ]
    args = (qs, chunks, judge, project.path("prompts_dir"), "chunk_relevance_v1", out)
    assert run_evidence_judgments("chunk_relevance", tasks, *args) == 2
    assert run_evidence_judgments("chunk_relevance", tasks, *args) == 0
    verdicts = load_evidence_verdicts(out, "judge", "chunk_relevance_v1")
    assert verdicts == {"q1|c0": "supports", "q2|c1": "supports"}
    assert load_evidence_verdicts(out, "other-judge", "chunk_relevance_v1") == {}
    other = FakeClient("other-judge", lambda req: '{"verdict": "not_relevant", "rationale": ""}')
    n = run_evidence_judgments(
        "chunk_relevance", tasks, qs, chunks, other, project.path("prompts_dir"),
        "chunk_relevance_v1", out,
    )  # fmt: skip
    assert n == 2  # a different judge model is not served from the cache


def test_invalid_verdicts_are_parse_errors_and_retried(project: Config, tmp_path: Path):
    qs = {"q1": make_question("q1")}
    chunks = {"c0": make_chunk("c0")}
    out = tmp_path / "s.jsonl"
    bad = FakeClient("judge", lambda req: '{"verdict": "maybe", "rationale": ""}')
    task = [EvidenceTask("q1", ("c0",))]
    run_evidence_judgments(
        "context_sufficiency", task, qs, chunks, bad, project.path("prompts_dir"),
        "context_sufficiency_v1", out,
    )  # fmt: skip
    rows = list(read_jsonl(out))
    assert rows[0]["status"] == "parse_error"
    assert load_evidence_judgments(out, "judge", "context_sufficiency_v1") == {}
    good = FakeClient("judge", lambda req: '{"verdict": "partial", "rationale": ""}')
    n = run_evidence_judgments(
        "context_sufficiency", task, qs, chunks, good, project.path("prompts_dir"),
        "context_sufficiency_v1", out,
    )  # fmt: skip
    assert n == 1


# ----------------------------------------------------------------------------- task builders
def test_sufficiency_tasks_use_top_k_and_gold():
    retrieval = {"q1": _retrieval("q1", ["a", "b", "c"])}
    tasks = sufficiency_tasks(["q1", "q2"], retrieval, top_k=2, gold={"q1": ["z"], "q2": ["y"]})
    assert tasks == [
        EvidenceTask("q1", ("a", "b")),
        EvidenceTask("q1", ("z",)),
        EvidenceTask("q2", ("y",)),
    ]
    assert sufficiency_tasks(["q1"], retrieval, top_k=2) == [EvidenceTask("q1", ("a", "b"))]


def test_relevance_pool_unions_retrieval_gold_and_citations():
    retrieval = {"q1": _retrieval("q1", ["a", "b", "c"])}
    records = [
        _record("q1", "A1", {"S1": "b", "S2": "x"}, ["S2", "S1", "S9"]),
        _record("q3", "A1", {"S1": "zz"}, ["S1"]),  # question not selected
    ]
    tasks = relevance_pool(["q1"], retrieval, pool_k=2, gold={"q1": ["g", "a"]}, records=records)
    assert [t.chunk_ids[0] for t in tasks] == ["a", "b", "g", "x"]


# ----------------------------------------------------------------------------- derived sets
def test_lenient_sets_keep_gold_and_add_supported_chunks():
    sets = lenient_relevant_sets(
        {"q1": ["g"], "q2": []},
        {"q1|g": "not_relevant", "q1|a": "supports", "q2|b": "partial", "q3|c": "supports"},
    )
    assert sets == {"q1": {"g", "a"}, "q3": {"c"}}
    with_partial = lenient_relevant_sets({}, {"q2|b": "partial"}, positive=("supports", "partial"))
    assert with_partial == {"q2": {"b"}}


def test_judged_membership_is_unknown_when_unjudged_chunks_remain():
    relevance = {"q1|a": "not_relevant", "q1|b": "supports"}
    relevant = lenient_relevant_sets({}, relevance)
    assert judged_membership("q1", ["a", "b"], relevant, relevance) is True
    assert judged_membership("q1", ["a"], relevant, relevance) is False
    assert judged_membership("q1", ["a", "c"], relevant, relevance) is None


def test_judged_citation_support():
    relevance = {"q1|a": "not_relevant", "q1|b": "supports"}
    relevant = lenient_relevant_sets({"q1": ["g"]}, relevance)
    full = judged_citation_support("q1", ["a", "b", "g"], relevant, relevance)
    assert full["citation_support_any_judged"] is True
    assert full["citation_precision_judged"] == pytest.approx(2 / 3)
    none = judged_citation_support("q1", ["a"], relevant, relevance)
    assert none == {"citation_support_any_judged": False, "citation_precision_judged": 0.0}
    unknown = judged_citation_support("q1", ["a", "c"], relevant, relevance)
    assert unknown["citation_support_any_judged"] is None
    assert unknown["citation_precision_judged"] is None
    empty = judged_citation_support("q1", [], relevant, relevance)
    assert empty["citation_support_any_judged"] is False


# ----------------------------------------------------------------------------- quadrants
def test_evidence_flags_and_judged_quadrants():
    df = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "arm": ["A1"] * 5,
            "correct": [True, False, True, False, True],
            "context_sufficiency_judged": ["sufficient", "sufficient", "partial", "insufficient",
                                           None],
        }
    )  # fmt: skip
    flags = evidence_flags(df, "judged").tolist()
    assert flags[:4] == [True, True, False, False] and flags[4] is None
    lenient_partial = evidence_flags(df, "judged", ("sufficient", "partial")).tolist()
    assert lenient_partial[2] is True
    row = quadrant_table(df, "judged").iloc[0]
    assert row["n"] == 4 and row["evidence_definition"] == "judged"
    assert (row["grounded_success"], row["generation_failure"]) == (1, 1)
    assert (row["own_knowledge_or_guess"], row["retrieval_failure"]) == (1, 1)
    assert quadrant_table(df, "lenient").empty  # column absent -> no table


# ----------------------------------------------------------------------------- retrieval metrics
def test_strict_and_lenient_retrieval_summaries():
    questions = [make_question("q1"), make_question("q2")]
    records = [_retrieval("q1", ["a", "b", "c"]), _retrieval("q2", ["x", "y"])]
    chunk_doc = {c: "d" for c in "abcxy"}
    strict = evaluate_retrieval(records, {"q1": ["c"]}, chunk_doc, questions, ks=(1, 3))
    lenient_sets = lenient_relevant_sets({"q1": ["c"]}, {"q1|a": "supports", "q2|y": "supports"})
    lenient = evaluate_retrieval(
        records, {q: sorted(s) for q, s in lenient_sets.items()}, chunk_doc, questions,
        ks=(1, 3), relevance_label="lenient",
    )  # fmt: skip
    summary = summarize_retrieval(pd.concat([strict, lenient])).set_index(["relevance", "group"])
    assert summary.loc[("strict", "all"), "n"] == 1
    assert summary.loc[("strict", "all"), "hit@1"] == 0
    assert summary.loc[("lenient", "all"), "n"] == 2
    assert summary.loc[("lenient", "all"), "hit@1"] == 0.5
    assert summary.loc[("lenient", "all"), "MRR"] == pytest.approx((1 + 0.5) / 2)


# ----------------------------------------------------------------------------- human validation
def test_export_evidence_sample_is_blinded(project: Config, tmp_path: Path):
    qs = {"q1": make_question("q1")}
    chunks = {"c0": make_chunk("c0", text="passage zero")}
    out = tmp_path / "j.jsonl"
    judge = FakeClient("judge", lambda req: '{"verdict": "supports", "rationale": "secret"}')
    run_evidence_judgments(
        "chunk_relevance", [EvidenceTask("q1", ("c0",))], qs, chunks, judge,
        project.path("prompts_dir"), "chunk_relevance_v1", out,
    )  # fmt: skip
    judgments = load_evidence_judgments(out, "judge", "chunk_relevance_v1")
    df = export_evidence_sample("chunk_relevance", judgments, qs, chunks, 10, 1, tmp_path / "h.csv")
    assert len(df) == 1 and df.loc[0, "key"] == "q1|c0"
    assert "passage zero" in df.loc[0, "passages"] and "B. beta" in df.loc[0, "correct_answer"]
    assert "supports" not in json.dumps(df.drop(columns="allowed_verdicts").to_dict())
    assert df.loc[0, "allowed_verdicts"] == "supports/partial/not_relevant"


def test_kappa_infers_labels():
    a = ["sufficient", "partial", "insufficient", "sufficient"]
    assert cohens_kappa(a, a) == 1.0
    assert not math.isnan(cohens_kappa(a, ["sufficient", "sufficient", "insufficient", "partial"]))


# ----------------------------------------------------------------------------- config
def test_judge_settings_do_not_change_run_hash(project: Config):
    before = project.frozen_hash()
    project.judge.pool_k = 20
    project.judge.positive_relevance = ["supports", "partial"]
    project.models["judge"].model_id = "another-judge"
    project.prompt.sufficiency_version = "context_sufficiency_v2"
    project.analysis.bootstrap_iterations = 5000
    assert project.frozen_hash() == before
    project.prompt.version = "answer_v2"
    assert project.frozen_hash() != before


# ----------------------------------------------------------------------------- training cutoffs
def test_parse_partial_date_resolves_periods():
    from datetime import date

    from vaxrag.config import parse_partial_date

    assert parse_partial_date("2025-08") == date(2025, 8, 31)
    assert parse_partial_date("2024-02") == date(2024, 2, 29)
    assert parse_partial_date("2025") == date(2025, 12, 31)
    assert parse_partial_date("2025-08", end_of_period=False) == date(2025, 8, 1)
    assert parse_partial_date("2026-02-14") == date(2026, 2, 14)


def test_models_must_predate_question_bank(project: Config):
    project.check_training_cutoffs(["fake", "fake2"])  # both cutoffs before 2026-02-14
    project.models["fake2"].training_data_cutoff = "2026-02"  # month containing the release
    with pytest.raises(ValueError, match="fake2"):
        project.check_training_cutoffs(["fake2"])
    project.models["fake2"].training_data_cutoff = None
    with pytest.raises(ValueError, match="not set"):
        project.check_training_cutoffs(["fake2"])
    project.questions.release_date = None  # check disabled
    project.check_training_cutoffs(["fake2"])


def test_default_config_models_predate_question_bank():
    from conftest import REPO_ROOT

    from vaxrag.config import load_config

    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    config.check_training_cutoffs(config.experiment.models)
    assert all(not m.model_id.endswith("latest") for m in config.models.values())
