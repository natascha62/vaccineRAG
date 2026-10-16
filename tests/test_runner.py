import json

import pytest
from conftest import FakeClient, make_chunk, make_question

from vaxrag.config import Config
from vaxrag.io_utils import read_jsonl
from vaxrag.llm.base import LLMError, LLMRefusal
from vaxrag.runner import (
    CallSpec,
    completed_call_ids,
    load_run_context,
    plan_calls,
    prepare_run_dir,
    run_experiment,
    select_questions,
)
from vaxrag.schemas import RetrievalRecord, RetrievedChunk


def answer(letters, refs=()):
    return json.dumps(
        {
            "answer": letters,
            "references": [{"source_id": s, "quote": q} for s, q in refs],
            "answer_basis": "provided_context",
            "justification": "ok",
        }
    )


def _context(project: Config, questions, retrieval=True):
    chunks = [make_chunk(f"d::{i}") for i in range(6)]
    ctx = load_run_context(project, "test", questions, chunks, None, {"q1": ["d::5"]})
    if retrieval:
        ctx.retrieval = {
            q.question_id: RetrievalRecord(
                question_id=q.question_id,
                retrieval_config_hash="h",
                query="",
                results=[
                    RetrievedChunk(chunk_id=f"d::{i}", rank=i + 1, score=1.0) for i in range(6)
                ],
            )
            for q in questions
        }
    return ctx


def test_plan_calls_with_consistency(project: Config):
    project.experiment.consistency.enabled = True
    project.experiment.consistency.n_questions = 1
    qs = [make_question("q1"), make_question("q2")]
    specs = plan_calls(qs, ["fake"], project)
    # 2 questions x 3 arms main + 1 question x 3 arms x 2 extra reps
    assert len(specs) == 6 + 6
    assert len({s.call_id for s in specs}) == len(specs)
    assert CallSpec("q1", "m", "A1", 2, "v1").call_id == "q1|m|A1|r2|v1"


def test_select_questions_filters(project: Config):
    qs = [
        make_question("q1"),
        make_question("q2", multi=True).model_copy(update={"version_status": "outdated"}),
    ]
    project.experiment.question_filter.exclude_version_status = ["outdated"]
    assert [q.question_id for q in select_questions(qs, project)] == ["q1"]
    project.experiment.question_filter.exclude_version_status = []
    project.experiment.question_filter.subsets = ["S2"]
    assert [q.question_id for q in select_questions(qs, project)] == ["q2"]


def test_run_all_arms_and_resume(project: Config):
    qs = [make_question("q1"), make_question("q2")]
    ctx = _context(project, qs)
    client = FakeClient()
    counts = run_experiment(ctx, {"fake": client}, qs)
    # q2 has no gold chunks -> A2 skipped for q2
    assert counts == {"ok": 5}
    rows = list(read_jsonl(ctx.calls_path))
    by_arm = {(r["question_id"], r["arm"]): r for r in rows}
    assert by_arm[("q1", "A0")]["context_chunk_ids"] == []
    assert by_arm[("q1", "A1")]["context_chunk_ids"] == [f"d::{i}" for i in range(5)]
    assert by_arm[("q1", "A2")]["context_chunk_ids"] == ["d::5"]
    assert by_arm[("q1", "A1")]["context_labels"]["S1"] == "d::0"
    assert by_arm[("q1", "A0")]["answer"] == ["B"]
    assert by_arm[("q1", "A0")]["cost_usd"] == pytest.approx((100 * 1 + 10 * 2) / 1e6)
    assert (ctx.run_dir / "retrieval.jsonl").exists() and (
        ctx.run_dir / "skipped_calls.jsonl"
    ).exists()
    # Second invocation makes no new calls.
    n_requests = len(client.requests)
    assert run_experiment(ctx, {"fake": client}, qs) == {}
    assert len(client.requests) == n_requests


def test_parse_retry_then_final_failure(project: Config):
    project.experiment.arms = ["A0"]
    qs = [make_question("q1")]
    ctx = _context(project, qs, retrieval=False)
    client = FakeClient(responder=lambda req: "garbage")
    assert run_experiment(ctx, {"fake": client}, qs) == {"parse_error": 1}
    rows = list(read_jsonl(ctx.calls_path))
    assert [r["attempt"] for r in rows] == [1, 2]
    done, _ = completed_call_ids(ctx.calls_path, 2)
    assert len(done) == 1


def test_parse_retry_succeeds_on_second_attempt(project: Config):
    project.experiment.arms = ["A0"]
    qs = [make_question("q1")]
    ctx = _context(project, qs, retrieval=False)
    outputs = iter(["garbage", answer(["B"])])
    run_experiment(ctx, {"fake": FakeClient(responder=lambda req: next(outputs))}, qs)
    assert [r["status"] for r in read_jsonl(ctx.calls_path)] == ["parse_error", "ok"]


def test_api_errors_are_retried_on_next_invocation_and_refusals_are_final(project: Config):
    project.experiment.arms = ["A0"]
    qs = [make_question("q1"), make_question("q2")]
    ctx = _context(project, qs, retrieval=False)

    def flaky(req):
        return LLMError("503") if "q1" in req.user else LLMRefusal("no")

    assert run_experiment(ctx, {"fake": FakeClient(responder=flaky)}, qs) == {
        "api_error": 1,
        "refusal": 1,
    }
    good = FakeClient()
    assert run_experiment(ctx, {"fake": good}, qs) == {"ok": 1}
    assert len(good.requests) == 1 and "q1" in good.requests[0].user


def test_frozen_config_guard(project: Config):
    prepare_run_dir(project, "frozen")
    project.experiment.max_workers = 99  # execution-only: allowed
    prepare_run_dir(project, "frozen")
    project.retrieval.top_k = 9  # result-affecting: refused
    with pytest.raises(RuntimeError, match="different settings"):
        prepare_run_dir(project, "frozen")
    prepare_run_dir(project, "frozen", allow_config_change=True)


def test_dry_run_counts_only(project: Config):
    qs = [make_question("q1")]
    ctx = _context(project, qs)
    client = FakeClient()
    assert run_experiment(ctx, {"fake": client}, qs, dry_run=True) == {"planned": 3}
    assert client.requests == []
