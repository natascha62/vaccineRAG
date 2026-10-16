"""Small I/O helpers: JSONL reading/writing, hashing and provenance metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_WRITE_LOCK = threading.Lock()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line of a JSONL file.

    A missing file yields nothing, which makes resumable runs simple. A
    truncated final line (e.g. after a crash mid-write) is skipped.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_models(path: str | Path, model: type[T]) -> list[T]:
    """Read a JSONL file into a list of pydantic models."""
    return [model.model_validate(row) for row in read_jsonl(path)]


def append_jsonl(path: str | Path, record: BaseModel | dict[str, Any]) -> None:
    """Append one record to a JSONL file in a thread-safe way and flush immediately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    line = json.dumps(payload, ensure_ascii=False)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict[str, Any]]) -> int:
    """Overwrite a JSONL file with the given records and return the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(obj: Any, length: int = 12) -> str:
    """Return a short deterministic hash of any JSON-serializable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def git_commit(repo_dir: str | Path = ".") -> str:
    """Return the current git commit hash (with '-dirty' if uncommitted changes exist)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
