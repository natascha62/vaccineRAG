"""Guideline document manifest (`manifest.csv`).

Every document in the knowledge base has one row recording its provenance:
publisher, title, version date, download date, URL, local filename and the
SHA-256 hash of the file. Hashes freeze the knowledge-base version: if a file
changes after freezing, `verify_manifest` reports it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from vaxrag.io_utils import sha256_file

MANIFEST_COLUMNS = [
    "doc_id",
    "publisher",
    "title",
    "doc_type",
    "version_date",
    "download_date",
    "url",
    "filename",
    "sha256",
]
# doc_id becomes part of chunk IDs and override file names, so keep it simple.
DOC_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
# Dates: YYYY-MM-DD (YYYY-MM or YYYY when the document gives no day).
DATE_RE = re.compile(r"^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$")
DOC_TYPES = {"general_principles", "vaccine_recommendation", "schedule", "other"}


class ManifestEntry(BaseModel):
    """One guideline document."""

    doc_id: str
    publisher: str
    title: str
    doc_type: str
    version_date: str = ""
    download_date: str = ""
    url: str = ""
    filename: str
    sha256: str = ""

    @property
    def year(self) -> str:
        """Publication year (first four characters of `version_date`)."""
        return self.version_date[:4]


def create_manifest_template(path: str | Path) -> bool:
    """Create an empty manifest with headers if none exists; return True if created."""
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(path, index=False)
    return True


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Load and validate the manifest."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    entries = [ManifestEntry(**row) for row in df[MANIFEST_COLUMNS].to_dict(orient="records")]
    ids = [e.doc_id for e in entries]
    bad_ids = [i for i in ids if not DOC_ID_RE.match(i)]
    if bad_ids:
        raise ValueError(
            f"Invalid doc_id {bad_ids}: use lowercase letters, digits and single underscores "
            "(e.g. 'acip_hpv_2019')"
        )
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id values in the manifest must be unique")
    for e in entries:
        if not DATE_RE.match(e.version_date):
            raise ValueError(
                f"{e.doc_id}: version_date '{e.version_date}' is required and must be "
                "YYYY-MM-DD (or YYYY-MM / YYYY); the year is shown to the models"
            )
        if e.download_date and not DATE_RE.match(e.download_date):
            raise ValueError(f"{e.doc_id}: download_date '{e.download_date}' must be YYYY-MM-DD")
    bad_types = [e.doc_id for e in entries if e.doc_type not in DOC_TYPES]
    if bad_types:
        raise ValueError(f"Invalid doc_type for {bad_types}; allowed: {sorted(DOC_TYPES)}")
    return entries


def fill_hashes(manifest_path: str | Path, raw_dir: str | Path) -> int:
    """Compute SHA-256 for rows whose hash is empty; return the number filled."""
    df = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    filled = 0
    for idx, row in df.iterrows():
        if row["sha256"]:
            continue
        file_path = Path(raw_dir) / row["filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"{row['doc_id']}: {file_path} not found")
        df.at[idx, "sha256"] = sha256_file(file_path)
        filled += 1
    df.to_csv(manifest_path, index=False)
    return filled


def verify_manifest(entries: list[ManifestEntry], raw_dir: str | Path) -> list[str]:
    """Return problems: missing files, missing hashes or hash mismatches."""
    problems: list[str] = []
    for e in entries:
        file_path = Path(raw_dir) / e.filename
        if not file_path.exists():
            problems.append(f"{e.doc_id}: file {e.filename} not found")
        elif not e.sha256:
            problems.append(f"{e.doc_id}: sha256 not recorded (run `vaxrag manifest-hash`)")
        elif sha256_file(file_path) != e.sha256:
            problems.append(f"{e.doc_id}: sha256 mismatch - file changed after freezing")
    return problems
