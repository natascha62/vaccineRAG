# Manual annotations

Two infectious-disease physicians (one adult, one pediatric) annotate **all** questions
independently. The workflow:

1. `vaxrag prepare-questions` creates (or extends) one copy of both files per annotator in
   `annotators/<name>/` (`adult_id`, `pediatric_id`; configurable in `annotation.annotators`) and
   the consensus files in this folder. Existing annotations are never overwritten.
2. Each annotator fills `annotators/<name>/question_annotations.csv` and
   `annotators/<name>/gold_passages.csv` without seeing the other's work.
3. `vaxrag annotation-agreement` reports agreement (`results/analysis/annotation/`: percent
   agreement and Cohen's kappa per field, gold-document overlap per question) and writes:
   - `disagreements.csv`: every question and field on which the annotators differ,
   - `consensus_draft.csv`: agreed values pre-filled, disagreements blank with both values in `notes`,
   - `gold_passages_combined.csv`: both annotators' gold passages side by side.
4. In a consensus discussion the annotators settle the disagreements and write the final labels to
   `question_annotations.csv` and `gold_passages.csv` in this folder. Only these consensus files are
   used by the pipeline.

The formats below apply to the annotator copies and the consensus files alike.

## question_annotations.csv (one row per question)

| column | allowed values |
| --- | --- |
| `question_type` | `fact`, `rule_application`, `multi_condition` |
| `population` | free text (optional)  (e.g. `infants`, `pregnant`, `immunocompromised`) |
| `answer_source` | `table`, `text`, `both` |
| `kb_covered` | `yes` / `no` - is the answer supported by a document in the knowledge base? |
| `version_status` | `current`, `outdated`, `unmappable` (checked against the frozen guidelines) |
| `notes` | free text |

Blank cells mean "not annotated yet". `vaxrag` validates the values on every load.

## gold_passages.csv (one row per question x gold passage)

Used for the oracle arm (A2) and for the *strict* retrieval and citation metrics.

Annotate **one sufficient set of evidence** per question: the passage(s) from which an expert
can answer the question. You do not need to find every chunk or guideline that could also answer
it - other supporting chunks are found by the LLM judge (`vaxrag judge-evidence`), and the
analysis reports strict (gold-only), lenient (gold + judged) and judged metrics side by side.
If the answer needs information from two places (e.g. a table and its footnote in another
section), add both rows: together they form the sufficient set.

| column | meaning |
| --- | --- |
| `question_id` | from `question_annotations.csv` |
| `doc_id` | from `data/guidelines/manifest.csv` |
| `section` | section path (or its start), e.g. `Timing and Spacing > Table 3-1` |
| `page` | 1-based PDF page (optional) |
| `quote` | short verbatim text from the passage (optional, most precise) |
| `chunk_id` | optional explicit chunk ID(s) separated by `;` (overrides the other fields) |
| `notes` | free text |

Add several rows when the answer needs several passages. After chunking, run
`vaxrag resolve-gold` to map rows to chunk IDs (`gold_chunks.jsonl`); it lists rows that
could not be resolved or are ambiguous so you can add a page or quote.

## table_review.csv (targeted table check)

Created and extended by `vaxrag table-review` after `vaxrag collect`. It lists each table chunk
that was in the context of at least one wrong A1 answer, most suspicious first (cited by the
wrong answer, relevant to the question, number of models that got it wrong). Add `--images` to
render the PDF pages to `results/analysis/table_review/`.

| column | meaning |
| --- | --- |
| `priority` ... `page_images` | filled by the tool; do not edit |
| `parse_ok` | `yes` = the chunk matches the PDF table; `no` = parsing error |
| `error_description` | what is wrong (e.g. shifted rows, merged cells lost, footnote missing) |
| `reviewer`, `review_date` | who reviewed it and when |

Review from the top until the remaining tables are unlikely to matter. Existing entries are kept
when the queue is rebuilt. Reviews are tied to the knowledge-base version (`kb_hash`): if the
knowledge base is re-chunked, the affected tables appear again for review.

