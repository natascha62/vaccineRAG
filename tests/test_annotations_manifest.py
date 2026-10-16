from pathlib import Path

import pandas as pd
import pytest
from conftest import make_chunk, make_question

from vaxrag.data.annotations import (
    apply_annotations,
    create_annotation_templates,
    resolve_gold_chunks,
    resolve_gold_row,
)
from vaxrag.data.manifest import (
    create_manifest_template,
    fill_hashes,
    load_manifest,
    verify_manifest,
)


def test_templates_do_not_overwrite_existing_annotations(tmp_path: Path):
    ann, gold = tmp_path / "ann.csv", tmp_path / "gold.csv"
    q1, q2 = make_question("q1"), make_question("q2")
    assert create_annotation_templates([q1], ann, gold) == (1, 1)
    df = pd.read_csv(ann, dtype=str, keep_default_na=False)
    df.loc[0, "question_type"] = "fact"
    df.to_csv(ann, index=False)
    added, created = create_annotation_templates([q1, q2], ann, gold)
    assert (added, created) == (1, 0)
    df = pd.read_csv(ann, dtype=str, keep_default_na=False)
    assert list(df["question_id"]) == ["q1", "q2"]
    assert df.loc[0, "question_type"] == "fact"


def test_apply_annotations_parses_and_validates(tmp_path: Path):
    ann, gold = tmp_path / "ann.csv", tmp_path / "gold.csv"
    q = make_question("q1")
    create_annotation_templates([q], ann, gold)
    df = pd.read_csv(ann, dtype=str, keep_default_na=False)
    df.loc[0, ["question_type", "answer_source", "kb_covered", "version_status"]] = [
        "rule_application",
        "table",
        "Yes",
        "current",
    ]
    df.to_csv(ann, index=False)
    (annotated,) = apply_annotations([q], ann)
    assert annotated.question_type == "rule_application"
    assert annotated.answer_source == "table"
    assert annotated.kb_covered is True
    df.loc[0, "answer_source"] = "figure"
    df.to_csv(ann, index=False)
    with pytest.raises(ValueError, match="answer_source"):
        apply_annotations([q], ann)


def test_resolve_gold_by_section_page_and_quote():
    chunks = [
        make_chunk(
            "d::0",
            section=["Timing and Spacing", "Intervals"],
            text="minimum interval is 4 weeks",
            pages=(3, 4),
        ),
        make_chunk(
            "d::1",
            section=["Timing and Spacing", "Intervals"],
            text="live vaccines spacing",
            pages=(5, 5),
        ),
        make_chunk(
            "d::2", section=["Contraindications"], text="severe allergic reaction", pages=(9, 9)
        ),
    ]
    for c in chunks:
        c.doc_id = "d"
    assert resolve_gold_row(
        {"doc_id": "d", "section": "timing and spacing > intervals"}, chunks
    ) == ["d::0", "d::1"]
    assert resolve_gold_row(
        {"doc_id": "d", "section": "Timing and Spacing", "page": "4"}, chunks
    ) == ["d::0"]
    assert resolve_gold_row({"doc_id": "d", "quote": "Minimum interval is 4 weeks"}, chunks) == [
        "d::0"
    ]
    assert resolve_gold_row({"chunk_id": "x::1; x::2"}, chunks) == ["x::1", "x::2"]
    assert resolve_gold_row({"doc_id": "other"}, chunks) == []


def test_resolve_gold_chunks_groups_and_flags(tmp_path: Path):
    chunks = [make_chunk(f"d::{i}", doc_id="d", section=["S"]) for i in range(5)]
    gold = tmp_path / "gold.csv"
    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "doc_id": "d",
                "section": "S",
                "page": "",
                "quote": "",
                "chunk_id": "",
                "notes": "",
            },
            {
                "question_id": "q2",
                "doc_id": "d",
                "section": "",
                "page": "",
                "quote": "",
                "chunk_id": "d::1",
                "notes": "",
            },
            {
                "question_id": "q2",
                "doc_id": "zzz",
                "section": "",
                "page": "",
                "quote": "",
                "chunk_id": "",
                "notes": "",
            },
        ]
    ).to_csv(gold, index=False)
    res = {r.question_id: r for r in resolve_gold_chunks(gold, chunks)}
    assert res["q1"].ambiguous and len(res["q1"].chunk_ids) == 5
    assert res["q2"].chunk_ids == ["d::1"] and len(res["q2"].unresolved) == 1


def test_manifest_hashing_and_verification(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.pdf").write_bytes(b"pdf-a")
    manifest = tmp_path / "manifest.csv"
    assert create_manifest_template(manifest)
    assert not create_manifest_template(manifest)
    pd.DataFrame(
        [
            {
                "doc_id": "a",
                "publisher": "CDC",
                "title": "Doc A",
                "doc_type": "schedule",
                "version_date": "2024-01-01",
                "download_date": "",
                "url": "",
                "filename": "a.pdf",
                "sha256": "",
            }
        ]
    ).to_csv(manifest, index=False)
    entries = load_manifest(manifest)
    assert "sha256 not recorded" in verify_manifest(entries, raw)[0]
    assert fill_hashes(manifest, raw) == 1
    assert verify_manifest(load_manifest(manifest), raw) == []
    (raw / "a.pdf").write_bytes(b"changed")
    assert "mismatch" in verify_manifest(load_manifest(manifest), raw)[0]


def test_manifest_rejects_bad_doc_type(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "doc_id": "a",
                "publisher": "",
                "title": "",
                "doc_type": "blog",
                "version_date": "2024-01-01",
                "download_date": "",
                "url": "",
                "filename": "a.pdf",
                "sha256": "",
            }
        ]
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="doc_type"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    "doc_id", ["ACIP_HPV", "acip hpv", "acip-hpv", "acip/hpv", "_acip", "acip__hpv"]
)
def test_manifest_rejects_bad_doc_ids(tmp_path: Path, doc_id: str):
    manifest = tmp_path / "manifest.csv"
    row = {
        "doc_id": doc_id,
        "publisher": "",
        "title": "",
        "doc_type": "other",
        "version_date": "2024-01-01",
        "download_date": "",
        "url": "",
        "filename": "a.pdf",
        "sha256": "",
    }
    pd.DataFrame([row]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="Invalid doc_id"):
        load_manifest(manifest)


def test_manifest_accepts_simple_doc_ids(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {
            "doc_id": d,
            "publisher": "",
            "title": "",
            "doc_type": "other",
            "version_date": "2024-01-01",
            "download_date": "",
            "url": "",
            "filename": f"{d}.pdf",
            "sha256": "",
        }
        for d in ("acip_hpv_2019", "gbp_timing_spacing", "schedule7")
    ]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    assert [e.doc_id for e in load_manifest(manifest)] == [
        "acip_hpv_2019",
        "gbp_timing_spacing",
        "schedule7",
    ]


def _manifest_row(**overrides):
    row = {
        "doc_id": "acip_hpv_2019",
        "publisher": "CDC/ACIP",
        "title": "HPV",
        "doc_type": "vaccine_recommendation",
        "version_date": "2019-08-16",
        "download_date": "2026-10-15",
        "url": "",
        "filename": "a.pdf",
        "sha256": "",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize("version_date", ["", "2019/8/16", "16-08-2019", "2019-13-01", "Aug 2019"])
def test_manifest_requires_valid_version_date(tmp_path: Path, version_date: str):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([_manifest_row(version_date=version_date)]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="version_date"):
        load_manifest(manifest)


def test_manifest_date_formats_and_year(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        _manifest_row(doc_id="a", version_date="2019-08-16"),
        _manifest_row(doc_id="b", version_date="2021-03", download_date=""),
        _manifest_row(doc_id="c", version_date="2024"),
    ]
    pd.DataFrame(rows).to_csv(manifest, index=False)
    assert [e.year for e in load_manifest(manifest)] == ["2019", "2021", "2024"]
    pd.DataFrame([_manifest_row(download_date="15/10/2026")]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="download_date"):
        load_manifest(manifest)
