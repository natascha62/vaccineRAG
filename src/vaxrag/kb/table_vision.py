"""Convert hard tables to Markdown with a multimodal LLM.

Pages listed in `table_vision_pages.csv` (columns: doc_id, page, notes) are
rendered to PNG and sent to the configured vision model. The model's Markdown
is saved as `<doc_id>_p<page>.md` in the overrides folder, where it should be
checked manually before chunking (`apply_table_overrides` then uses it).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vaxrag.llm.base import LLMClient, LLMRequest
from vaxrag.prompts import load_prompt, render


def render_page_png(pdf_path: str | Path, page_no: int, scale: float = 2.0) -> bytes:
    """Render one 1-based PDF page to PNG bytes with pypdfium2."""
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        image = pdf[page_no - 1].render(scale=scale).to_pil()
    finally:
        pdf.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def convert_pages(
    pages_csv: str | Path,
    filenames: dict[str, str],
    raw_dir: str | Path,
    images_dir: str | Path,
    overrides_dir: str | Path,
    client: LLMClient,
    prompts_dir: str | Path,
    prompt_version: str,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every listed page; existing override files are kept unless `overwrite`."""
    pages = pd.read_csv(pages_csv, dtype=str, keep_default_na=False)
    system, user = load_prompt(prompts_dir, prompt_version)
    images_dir, overrides_dir = Path(images_dir), Path(overrides_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    overrides_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in pages.to_dict(orient="records"):
        doc_id, page = row["doc_id"].strip(), int(row["page"])
        out_path = overrides_dir / f"{doc_id}_p{page}.md"
        if out_path.exists() and not overwrite:
            continue
        if doc_id not in filenames:
            raise KeyError(f"{doc_id} is not in the manifest")
        png = render_page_png(Path(raw_dir) / filenames[doc_id], page)
        (images_dir / f"{doc_id}_p{page}.png").write_bytes(png)
        result = client.generate(
            LLMRequest(
                system=system, user=render(user, notes=row.get("notes", "")), images_png=[png]
            )
        )
        out_path.write_text(result.text.strip() + "\n", encoding="utf-8")
        written.append(out_path)
    return written
