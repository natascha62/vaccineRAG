# Guideline knowledge base

Put the frozen CDC/ACIP guideline PDFs in `raw/` and describe each one in `manifest.csv`.

Scope: every recommendation ACIP lists as current on the freeze date (not only vaccines in the
question bank), plus the General Best Practice Guidelines and the immunization schedules. The CDC
Pink Book is not included. When several documents are current for one vaccine, include all of them; skip
superseded documents. Each chunk label shows the document's publication year so the models can
tell newer from older guidance.

## manifest.csv

| column | meaning |
| --- | --- |
| `doc_id` | your own short, unique ID: lowercase letters, digits and single underscores, e.g. `acip_hpv_2019`; never change it once annotation has started |
| `publisher` | e.g. `CDC/ACIP` |
| `title` | full document title (shown to the model as the source label) |
| `doc_type` | `general_principles`, `vaccine_recommendation`, `schedule` or `other` |
| `version_date` | **required**: publication/version date printed in the document, `YYYY-MM-DD` (or `YYYY-MM` / `YYYY` if no day is given); its year is shown to the models |
| `download_date` | date you downloaded it (YYYY-MM-DD) |
| `url` | source URL |
| `filename` | file name inside `raw/` |
| `sha256` | leave empty; `vaxrag manifest-hash` fills it in |

Extra columns (e.g. `landing_page`, `archive_url`, `notes`) are allowed and ignored by the code.

After the knowledge base is frozen, do not change files in `raw/`; `vaxrag manifest-hash`
and `vaxrag parse` report any file whose hash no longer matches.

## Hard tables

Tables are not all checked by hand. Parsing problems are caught in two ways:

1. **Automatically flagged tables.** `vaxrag parse` writes `parsed/table_quality.csv`, which flags
   tables that Docling probably failed to parse (too few rows, mostly empty cells, no header).
   List those pages in `table_vision_pages.csv` (`doc_id,page,notes`; pages are 1-based) and run
   `vaxrag table-vision`, which writes `table_overrides/<doc_id>_p<page>.md`. These files replace
   Docling's tables on that page during `vaxrag chunk`.
2. **Targeted review after answering.** `vaxrag table-review` lists only the tables that were
   retrieved for wrong A1 answers (see `data/annotations/README.md`). During the pilot, faulty
   tables are fixed (edit or add the override file for that page, then re-run `chunk`,
   `resolve-gold` and `retrieve`) before the knowledge base is frozen. During the main
   experiment the knowledge base is not changed; reviews are used for error attribution only.
