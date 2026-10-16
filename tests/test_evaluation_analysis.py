import json

import numpy as np
import pandas as pd
import pytest
from conftest import FakeClient, make_chunk, make_question

from vaxrag.analysis import stats
from vaxrag.analysis.collect import build_analysis_table
from vaxrag.analysis.report import write_report
from vaxrag.config import Config
from vaxrag.evaluation.citations import (
    citation_correctness,
    citation_validity,
    cited_chunk_ids,
    match_memory_citation,
    quote_score,
)
from vaxrag.evaluation.evidence import relevance_key, sufficiency_key
from vaxrag.evaluation.judge import cohens_kappa, judge_one, run_judge
from vaxrag.evaluation.quadrants import assign_quadrant, quadrant_table
from vaxrag.evaluation.scoring import final_calls, flip_table, is_correct, score_calls
from vaxrag.io_utils import read_jsonl
from vaxrag.prompts import load_prompt
from vaxrag.runner import load_run_context, run_experiment
from vaxrag.schemas import CallRecord, Reference, RetrievalRecord, RetrievedChunk


# ----------------------------------------------------------------------------- scoring
def test_final_calls_prefers_success_and_drops_api_errors():
    calls = pd.DataFrame(
        [
            {"call_id": "a", "attempt": 1, "status": "parse_error", "answer": []},
            {"call_id": "a", "attempt": 2, "status": "ok", "answer": ["B"]},
            {"call_id": "b", "attempt": 1, "status": "api_error", "answer": []},
            {"call_id": "c", "attempt": 1, "status": "parse_error", "answer": []},
            {"call_id": "c", "attempt": 2, "status": "parse_error", "answer": []},
        ]
    )
    final = final_calls(calls)
    assert dict(zip(final["call_id"], final["status"], strict=True)) == {
        "a": "ok",
        "c": "parse_error",
    }
    assert final.loc[final["call_id"] == "c", "attempt"].item() == 2


def test_is_correct_exact_set_match():
    assert is_correct(["A", "C"], ["C", "A"])
    assert not is_correct(["A"], ["A", "C"])
    assert not is_correct([], ["A"])


def test_score_and_flip_table():
    q = {f"q{i}": make_question(f"q{i}", correct=["A"]) for i in range(4)}
    rows = []
    for i, (a0, a1) in enumerate([("A", "A"), ("B", "A"), ("A", "B"), ("B", "B")]):
        for arm, ans in (("A0", a0), ("A1", a1)):
            rows.append(
                {
                    "call_id": f"q{i}{arm}",
                    "question_id": f"q{i}",
                    "model": "m",
                    "arm": arm,
                    "rep": 1,
                    "status": "ok",
                    "answer": [ans],
                }
            )
    scored = score_calls(pd.DataFrame(rows), q)
    flips = flip_table(scored)
    assert flips.iloc[0][["both_correct", "rescued", "harmed", "both_wrong"]].tolist() == [
        1,
        1,
        1,
        1,
    ]


# ----------------------------------------------------------------------------- citations
def test_quote_score_tolerates_whitespace_and_case():
    assert quote_score("Minimum  interval is 4 weeks", "The minimum interval is 4 weeks.") == 100
    assert quote_score("completely different", "The minimum interval is 4 weeks.") < 60
    assert quote_score("", "x") == 0


def test_citation_validity_and_correctness():
    chunks = {
        "c1": make_chunk("c1", text="MMR dose 2 at 4-6 years"),
        "c2": make_chunk("c2", text="Hep B"),
    }
    refs = [
        Reference(source_id="S1", quote="dose 2 at 4-6 years"),
        Reference(source_id="[S2]", quote="nope"),
        Reference(source_id="S9", quote="x"),
    ]
    mapping = {"S1": "c1", "S2": "c2"}
    v = citation_validity(refs, mapping, {cid: c.text for cid, c in chunks.items()})
    assert (v["n_references"], v["n_known_source_ids"], v["n_verbatim_quotes"]) == (3, 2, 1)
    assert v["all_quotes_verbatim"] is False
    cited = cited_chunk_ids(refs, mapping)
    assert cited == ["c1", "c2"]
    assert citation_correctness(cited, ["c1"]) == {
        "citation_recall_strict": 1.0,
        "citation_precision_strict": 0.5,
    }
    assert np.isnan(citation_correctness(cited, [])["citation_recall_strict"])


def test_memory_citation_matching():
    titles = {"gbp": "ACIP General Best Practice Guidelines for Immunization"}
    sections = {
        "gbp": ["Timing and Spacing of Immunobiologics", "Contraindications and Precautions"]
    }
    good = Reference(
        source_id="ACIP General Best Practice Guidelines › Timing and Spacing of Immunobiologics",
        quote="",
    )
    doc_only = Reference(
        source_id="General Best Practice Guidelines for Immunization > Made-up section", quote=""
    )
    fake = Reference(source_id="WHO Vaccine Handbook 2031", quote="")
    assert match_memory_citation(good, titles, sections)["status"] == "doc_and_section"
    assert match_memory_citation(doc_only, titles, sections)["status"] == "doc_only"
    assert match_memory_citation(fake, titles, sections)["status"] == "unverified"


def test_quadrants():
    assert assign_quadrant(True, False) == "generation_failure"
    df = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "arm": ["A1"] * 4,
            "gold_in_context_strict": [True, True, False, None],
            "correct": [True, False, False, True],
        }
    )
    table = quadrant_table(df, "strict")
    row = table.iloc[0]
    assert (row["grounded_success"], row["generation_failure"], row["retrieval_failure"]) == (
        1,
        1,
        1,
    )
    assert row["own_knowledge_or_guess"] == 0


# ----------------------------------------------------------------------------- judge
def test_cohens_kappa():
    assert (
        cohens_kappa(["supports"] * 2 + ["partial"] * 2, ["supports"] * 2 + ["partial"] * 2) == 1.0
    )
    k = cohens_kappa(
        ["supports", "partial", "supports", "not_supports"],
        ["supports", "supports", "supports", "not_supports"],
    )
    assert 0 < k < 1
    with pytest.raises(ValueError):
        cohens_kappa(["a"], [])


def test_judge_one_and_resume(project: Config, tmp_path):
    q = make_question("q1")
    chunks = {"c1": make_chunk("c1", text="beta is right")}
    record = CallRecord(
        call_id="x",
        attempt=1,
        question_id="q1",
        model="m",
        arm="A1",
        rep=1,
        prompt_version="v",
        git_commit="g",
        run_id="r",
        timestamp="t",
        prompt_system="",
        prompt_user="",
        context_labels={"S1": "c1"},
        status="ok",
        parsed={
            "answer": ["B"],
            "references": [{"source_id": "S1", "quote": "beta"}],
            "answer_basis": "provided_context",
            "justification": "",
        },
    )
    system_t, user_t = load_prompt(project.path("prompts_dir"), "grounding_v1")
    judge = FakeClient("judge", responder=lambda req: '{"verdict": "supports", "rationale": "ok"}')
    j = judge_one(record, q, chunks, judge, system_t, user_t, "grounding_v1")
    assert j.verdict == "supports" and j.status == "ok"
    assert (
        "beta is right" in judge.requests[0].user and "Selected answer: B" in judge.requests[0].user
    )
    out = tmp_path / "judgments.jsonl"
    assert (
        run_judge(
            [record], {"q1": q}, chunks, judge, project.path("prompts_dir"), "grounding_v1", out
        )
        == 1
    )
    assert (
        run_judge(
            [record], {"q1": q}, chunks, judge, project.path("prompts_dir"), "grounding_v1", out
        )
        == 0
    )


# ----------------------------------------------------------------------------- statistics
def _synthetic(n_q=120, seed=0):
    """Two models x three arms; A1 and A2 raise accuracy, model m2 is weaker."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0, 1, n_q)
    rows = []
    for model, shift in (("m1", 0.8), ("m2", 0.0)):
        for arm, effect in (("A0", 0.0), ("A1", 0.9), ("A2", 1.6)):
            p = 1 / (1 + np.exp(-(shift + effect - difficulty)))
            correct = rng.random(n_q) < p
            for i in range(n_q):
                rows.append(
                    {
                        "call_id": f"{model}{arm}{i}",
                        "question_id": f"q{i}",
                        "model": model,
                        "arm": arm,
                        "rep": 1,
                        "correct": bool(correct[i]),
                        "answer": json.dumps(["A"]),
                        "parse_failed": False,
                        "refused": False,
                        "status": "ok",
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "latency_s": 1.0,
                        "cost_usd": 0.001,
                    }
                )
    return pd.DataFrame(rows)


def test_accuracy_table_and_pairwise_tests():
    df = _synthetic()
    acc = stats.accuracy_table(df, ["model", "arm"])
    assert len(acc) == 6 and acc["ci_low"].le(acc["accuracy"]).all()
    tests = stats.pairwise_arm_tests(df, iterations=200)
    assert len(tests) == 6
    a1 = tests[(tests["model"] == "m2") & (tests["contrast"] == "A1 - A0")].iloc[0]
    assert a1["diff"] > 0 and a1["diff_ci_low"] < a1["diff"] < a1["diff_ci_high"]
    assert (tests["p_holm"] >= tests["p_mcnemar"]).all()


def test_mcnemar_and_bootstrap_edge_cases():
    same = np.array([True, False, True])
    assert stats.mcnemar_exact(same, same) == 1.0
    lo, hi = stats.bootstrap_diff_ci(same, same, iterations=50)
    assert lo == hi == 0


def test_gee_and_bayes_glmm():
    df = _synthetic(n_q=150)
    gee = stats.gee_arm_effects(df)
    ors = gee["arm_odds_ratios"].set_index("term")
    assert ors.loc["C(arm)[T.A1]", "odds_ratio"] > 1 and ors.loc["C(arm)[T.A2]", "odds_ratio"] > 1
    assert 0 <= gee["interaction_p_value"] <= 1
    bayes = stats.bayes_mixed_arm_effects(df).set_index("term")
    assert bayes.loc["C(arm)[T.A1]", "post_mean"] > 0


def test_consistency():
    rows = []
    for rep, ans in ((1, "A"), (2, "A"), (3, "B")):
        rows.append(
            {
                "model": "m",
                "arm": "A0",
                "question_id": "q1",
                "rep": rep,
                "answer": ans,
                "correct": ans == "A",
            }
        )
    for rep in (1, 2, 3):
        rows.append(
            {
                "model": "m",
                "arm": "A0",
                "question_id": "q2",
                "rep": rep,
                "answer": "C",
                "correct": True,
            }
        )
    df = pd.DataFrame(rows)
    cons = stats.consistency_table(df)
    assert cons.iloc[0]["consistency_rate"] == 0.5 and cons.iloc[0]["n_questions"] == 2


def test_model_pair_tests():
    df = _synthetic()
    out = stats.model_pair_tests(df, "m2", "m1", ["A0", "A1"], iterations=100)
    assert list(out["arm"]) == ["A0", "A1"] and (out["diff"] > 0).all()


# ----------------------------------------------------------------------------- end to end
def test_collect_and_report_end_to_end(project: Config):
    project.experiment.models = ["fake", "fake2"]
    qs = [make_question(f"q{i}", correct=["B"] if i % 2 else ["A"]) for i in range(6)]
    chunks = [make_chunk(f"d::{i}", doc_id="d", text=f"passage {i} says beta") for i in range(6)]
    gold = {q.question_id: [f"d::{i}"] for i, q in enumerate(qs)}
    ctx = load_run_context(project, "e2e", qs, chunks, None, gold)
    ctx.retrieval = {
        q.question_id: RetrievalRecord(
            question_id=q.question_id,
            retrieval_config_hash="h",
            query="",
            results=[
                RetrievedChunk(chunk_id=f"d::{(i + j) % 6}", rank=j + 1, score=1.0)
                for j in range(6)
            ],
        )
        for i, q in enumerate(qs)
    }

    def respond(req):
        has_ctx = "[S1]" in req.user
        return json.dumps(
            {
                "answer": ["B"],
                "references": [{"source_id": "S1", "quote": "says beta"}]
                if has_ctx
                else [{"source_id": "Title d › Intro", "quote": "x"}],
                "answer_basis": "provided_context" if has_ctx else "own_knowledge",
                "justification": "",
            }
        )

    run_experiment(
        ctx, {"fake": FakeClient("fake", respond), "fake2": FakeClient("fake2", respond)}, qs
    )
    # Judge verdicts: A1 contexts of even questions are sufficient, odd ones insufficient.
    sufficiency = {
        sufficiency_key(q.question_id, [f"d::{(i + j) % 6}" for j in range(5)]): (
            "sufficient" if i % 2 == 0 else "insufficient"
        )
        for i, q in enumerate(qs)
    }
    relevance = {relevance_key("q0", "d::1"): "supports"}
    table = build_analysis_table(
        [ctx.run_dir], qs, chunks, gold, {"d": "Title d"}, relevance, sufficiency
    )
    assert len(table) == 2 * 3 * 6
    assert table["correct"].sum() == 2 * 3 * 3  # answers "B" are right for odd questions
    a1 = table[table["arm"] == "A1"]
    assert a1["gold_in_context_strict"].all() and (a1["share_quotes_verbatim"] == 1).all()
    assert a1["relevant_in_context_lenient"].all()
    assert (a1["citation_recall_strict"] == 1).all() and a1["citation_support_any_judged"].all()
    assert set(a1["context_sufficiency_judged"]) == {"sufficient", "insufficient"}
    # Sufficient contexts are on even questions, where "B" is wrong -> generation failures.
    judged = quadrant_table(a1, "judged")
    assert judged["generation_failure"].sum() == 2 * 3 and judged["grounded_success"].sum() == 0
    a0 = table[table["arm"] == "A0"]
    assert (a0["memory_citation_section_match_share"] == 1).all()
    written = write_report(table, project.analysis, project.path("analysis_dir"))
    names = {p.name for p in written}
    assert {
        "accuracy_by_model_arm.csv",
        "paired_arm_tests.csv",
        "quadrants_A1_strict.csv",
        "quadrants_A1_lenient.csv",
        "quadrants_A1_judged.csv",
        "sufficiency_by_correctness.csv",
        "citations.csv",
        "efficiency.csv",
        "gee_arm_odds_ratios.csv",
    } <= names
    assert len(list(read_jsonl(ctx.calls_path))) == 36


def test_primary_population_reports(project: Config, tmp_path):
    from vaxrag.analysis.report import primary_population, write_population_reports

    df = _synthetic(n_q=40)
    covered = {f"q{i}" for i in range(20)}
    df["kb_covered"] = df["question_id"].isin(covered)
    df["version_status"] = ["current" if i % 4 else "outdated" for i in range(len(df))]
    primary = primary_population(df, project.analysis)
    assert primary["kb_covered"].all() and (primary["version_status"] == "current").all()
    written = write_population_reports(df, project.analysis, tmp_path)
    assert written["primary"] and written["all_questions"]
    assert all(p.parent.name == "primary" for p in written["primary"])
    unannotated = df.assign(kb_covered=None, version_status=None)
    assert write_population_reports(unannotated, project.analysis, tmp_path / "x")["primary"] == []
