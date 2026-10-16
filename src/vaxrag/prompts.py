"""Loading and rendering of versioned prompt templates.

Each prompt version consists of two files in `prompts/`:
`<version>.system.txt` and `<version>.user.txt`, plus an optional
`<version>.parts.yaml` with reusable instruction snippets. Placeholders are written as
`{{name}}`; braces used elsewhere (e.g. JSON examples) are left untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def load_prompt(prompts_dir: str | Path, version: str) -> tuple[str, str]:
    """Return (system, user) template strings for a prompt version."""
    base = Path(prompts_dir)
    system = (base / f"{version}.system.txt").read_text(encoding="utf-8").strip()
    user = (base / f"{version}.user.txt").read_text(encoding="utf-8").strip()
    return system, user


def load_prompt_parts(prompts_dir: str | Path, version: str) -> dict[str, str]:
    """Return the instruction snippets in `<version>.parts.yaml` (empty if absent)."""
    path = Path(prompts_dir) / f"{version}.parts.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {k: str(v).strip() for k, v in (yaml.safe_load(fh) or {}).items()}


def render(template: str, **values: str) -> str:
    """Fill `{{name}}` placeholders; raises KeyError if a placeholder has no value."""

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise KeyError(f"Missing value for placeholder '{name}'")
        return str(values[name])

    return _PLACEHOLDER_RE.sub(substitute, template)
