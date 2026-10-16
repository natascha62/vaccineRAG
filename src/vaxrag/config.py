"""Experiment configuration.

All parameters live in YAML files under `configs/`. A base file can be combined
with one or more override files (deep-merged in order), e.g.::

    vaxrag run --config configs/default.yaml --config configs/pilot.yaml

The merged configuration is validated by the pydantic models below and copied
into each run folder so every result can be traced back to its exact settings.
"""

from __future__ import annotations

import calendar
import copy
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from vaxrag.io_utils import stable_hash


class PathsConfig(BaseModel):
    """Locations of input data, intermediate artifacts and results."""

    questions_dir: str = "MCQ_dataset"
    questions_jsonl: str = "data/processed/questions.jsonl"
    question_exclusions: str = "data/processed/question_exclusions.csv"
    annotations: str = "data/annotations/question_annotations.csv"
    gold_passages: str = "data/annotations/gold_passages.csv"
    annotators_dir: str = "data/annotations/annotators"
    gold_chunks: str = "data/annotations/gold_chunks.jsonl"
    manifest: str = "data/guidelines/manifest.csv"
    guidelines_raw: str = "data/guidelines/raw"
    parsed_dir: str = "data/guidelines/parsed"
    table_vision_pages: str = "data/guidelines/table_vision_pages.csv"
    table_images_dir: str = "data/guidelines/table_images"
    table_overrides_dir: str = "data/guidelines/table_overrides"
    chunks: str = "data/processed/chunks.jsonl"
    index_dir: str = "data/processed/index"
    retrieval_dir: str = "data/processed/retrieval"
    runs_dir: str = "results/runs"
    judgments_dir: str = "results/judgments"
    table_review: str = "data/annotations/table_review.csv"
    table_review_images: str = "results/analysis/table_review"
    analysis_dir: str = "results/analysis"
    prompts_dir: str = "prompts"


class QuestionsConfig(BaseModel):
    """How the raw question bank is loaded."""

    files: dict[str, str] = Field(
        default_factory=lambda: {
            "S1": "[final]English_MCQ_S1_labeled.xlsx",
            "S2": "[final]English_MCQ_S2_labeled.xlsx",
        }
    )
    multi_answer_subsets: list[str] = Field(default_factory=lambda: ["S2"])
    # First public release of the question bank. Every evaluated model must have
    # a training-data cutoff before this date (contamination control by design).
    release_date: str | None = "2026-02-14"


class AnnotationConfig(BaseModel):
    """Independent annotators; their consensus is written to the top-level annotation files."""

    annotators: list[str] = Field(default_factory=lambda: ["adult_id", "pediatric_id"])


class ChunkingConfig(BaseModel):
    """Section-based chunking parameters (token counts use `tokenizer`)."""

    tokenizer: str = "cl100k_base"
    min_tokens: int = 400
    max_tokens: int = 800
    overlap_ratio: float = 0.10
    table_max_tokens: int = 1200


class EmbedderConfig(BaseModel):
    """Dense retriever settings."""

    name: Literal["bge-m3", "medcpt"] = "bge-m3"
    model_id: str = "BAAI/bge-m3"
    query_model_id: str | None = None
    batch_size: int = 16
    max_length: int = 1024
    device: str = "auto"


class RerankerConfig(BaseModel):
    """Cross-encoder reranker settings."""

    model_id: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    max_length: int = 1024
    device: str = "auto"


class RetrievalConfig(BaseModel):
    """Hybrid retrieval pipeline: BM25 + dense -> RRF -> cross-encoder rerank."""

    query_mode: Literal["stem_and_options", "stem_only"] = "stem_and_options"
    use_bm25: bool = True
    use_dense: bool = True
    use_reranker: bool = True
    bm25_candidates: int = 30
    dense_candidates: int = 30
    rrf_k: int = 60
    top_k: int = 5
    rerank_field: Literal["retrieval_text", "text"] = "retrieval_text"
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    def config_hash(self) -> str:
        """Hash of every setting that changes retrieval output (device excluded)."""
        payload = self.model_dump()
        payload["embedder"].pop("device", None)
        payload["embedder"].pop("batch_size", None)
        payload["reranker"].pop("device", None)
        payload["reranker"].pop("batch_size", None)
        return stable_hash(payload)


class ModelConfig(BaseModel):
    """One LLM endpoint.

    `temperature=None` means the parameter is not sent (some reasoning models
    reject it); the provider default is then recorded in the run notes.
    """

    provider: Literal["openai", "anthropic", "google", "vllm"]
    model_id: str
    temperature: float | None = 0.0
    reasoning_effort: str | None = None
    max_output_tokens: int = 4096
    base_url: str | None = None
    api_key_env: str | None = None
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    timeout_s: float = 300.0
    max_workers: int | None = None
    supports_vision: bool = False
    supports_pdf: bool = False
    # Prompt-cache prices per 1M tokens; None means "same as the input price".
    price_cache_read_per_mtok: float | None = None
    price_cache_write_per_mtok: float | None = None
    # Provenance documented in the methods; `training_data_cutoff` is checked
    # against `questions.release_date` for every evaluated model.
    release_date: str | None = None
    training_data_cutoff: str | None = None
    cutoff_source: str | None = None


class PromptConfig(BaseModel):
    """Prompt template selection."""

    version: str = "answer_v1"
    grounding_version: str = "grounding_v1"
    relevance_version: str = "chunk_relevance_v1"
    sufficiency_version: str = "context_sufficiency_v1"
    selection_version: str = "doc_selection_v1"
    table_vision_version: str = "table_vision_v1"


class QuestionFilter(BaseModel):
    """Which questions enter a run."""

    subsets: list[str] = Field(default_factory=lambda: ["S1", "S2"])
    exclude_version_status: list[str] = Field(default_factory=list)
    only_kb_covered: bool = False
    sample_n: int | None = None
    seed: int = 2026


class ConsistencySpec(BaseModel):
    """Repeat runs on a seeded random subset of questions to measure answer consistency."""

    enabled: bool = False
    n_questions: int = 300
    reps: int = 3
    seed: int = 2026
    arms: list[str] = Field(default_factory=lambda: ["A0", "A1", "A2"])


class FullDocConfig(BaseModel):
    """Arm A3: the model selects guideline PDFs and reads them in full (no retrieval)."""

    models: list[str] = Field(default_factory=list)
    max_docs: int = 3
    cache_documents: bool = True


ARMS = ("A0", "A1", "A2", "A3")


class ExperimentConfig(BaseModel):
    """Which models, arms and repetitions are executed."""

    models: list[str] = Field(default_factory=list)
    arms: list[str] = Field(default_factory=lambda: ["A0", "A1", "A2"])
    question_filter: QuestionFilter = Field(default_factory=QuestionFilter)
    consistency: ConsistencySpec = Field(default_factory=ConsistencySpec)
    fulldoc: FullDocConfig = Field(default_factory=FullDocConfig)
    max_workers: int = 4
    max_parse_attempts: int = 2

    @model_validator(mode="after")
    def _check_arms(self) -> ExperimentConfig:
        unknown = (set(self.arms) | set(self.consistency.arms)) - set(ARMS)
        if unknown:
            raise ValueError(f"Unknown arms: {sorted(unknown)}")
        return self

    def arm_models(self, arm: str, models: list[str]) -> list[str]:
        """Models (from `models`) that run a given arm; A3 is limited to `fulldoc.models`."""
        if arm == "A3":
            return [m for m in models if m in self.fulldoc.models]
        return list(models)


class JudgeConfig(BaseModel):
    """LLM-as-judge settings.

    Three judgments use the same judge model:

    * grounding - do the model's quotes support the answer it selected?
    * chunk relevance - does a chunk support the correct answer? (citation
      support and pooled, "lenient" retrieval relevance)
    * context sufficiency - do the given chunks suffice to answer correctly?
      (error attribution without exhaustive gold annotation)
    """

    model: str = "judge"
    arms: list[str] = Field(default_factory=lambda: ["A1", "A2", "A3"])
    pool_k: int = 10
    judge_gold_sufficiency: bool = True
    positive_relevance: list[str] = Field(default_factory=lambda: ["supports"])
    positive_sufficiency: list[str] = Field(default_factory=lambda: ["sufficient"])
    human_sample_n: int = 200
    seed: int = 2026
    max_workers: int = 4


class PopulationFilter(BaseModel):
    """Annotation-based definition of the primary analysis population."""

    only_kb_covered: bool = True
    version_status: list[str] = Field(default_factory=lambda: ["current"])


class AnalysisConfig(BaseModel):
    """Settings used by the evaluation and statistics stages."""

    quote_match_threshold: float = 90.0
    citation_title_threshold: float = 80.0
    bootstrap_iterations: int = 1000
    seed: int = 2026
    reference_arm: str = "A0"
    # Primary analysis: questions covered by the knowledge base whose gold answer
    # agrees with the frozen guidelines. All questions are analysed as secondary.
    primary_population: PopulationFilter = Field(default_factory=PopulationFilter)
    # H5: (base model key, medically fine-tuned model key).
    paired_models: tuple[str, str] | None = ("gemma", "medgemma")
    run_glmm: bool = True
    # Arms of the primary regression models (H1/H2). A3 runs on a subset of
    # models, so it is compared with A1/A0 in separate analyses.
    primary_arms: list[str] = Field(default_factory=lambda: ["A0", "A1", "A2"])
    # Arms whose wrong answers put a retrieved table into the review queue.
    table_review_arms: list[str] = Field(default_factory=lambda: ["A1"])


class Config(BaseModel):
    """Top-level configuration object."""

    project_root: str = "."
    paths: PathsConfig = Field(default_factory=PathsConfig)
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    table_vision_model: str | None = None

    @model_validator(mode="after")
    def _check_model_keys(self) -> Config:
        missing = [m for m in self.experiment.models if m not in self.models]
        if missing:
            raise ValueError(f"experiment.models references undefined models: {missing}")
        if self.judge.model in self.experiment.models:
            raise ValueError("The judge model must not be one of the evaluated models.")
        if "A3" in self.experiment.arms:
            for key in self.experiment.fulldoc.models:
                if key not in self.models:
                    raise ValueError(f"experiment.fulldoc.models references undefined model {key}")
                if not self.models[key].supports_pdf:
                    raise ValueError(f"Model {key} runs arm A3 but supports_pdf is false")
        self.check_training_cutoffs(self.experiment.models)
        return self

    def check_training_cutoffs(self, model_keys: list[str]) -> None:
        """Raise if a model's training data could include the released question bank.

        Every model must declare `training_data_cutoff`, and the cutoff (read as
        the last day of the stated period) must precede `questions.release_date`.
        """
        if not self.questions.release_date:
            return
        release = parse_partial_date(self.questions.release_date, end_of_period=False)
        problems = []
        for key in model_keys:
            cutoff = self.models[key].training_data_cutoff
            if not cutoff:
                problems.append(f"{key}: training_data_cutoff is not set")
            elif parse_partial_date(cutoff) >= release:
                problems.append(f"{key}: training data cutoff {cutoff} is not before {release}")
        if problems:
            raise ValueError("Models may have seen the question bank:\n  " + "\n  ".join(problems))

    def path(self, name: str) -> Path:
        """Resolve a configured path relative to `project_root`."""
        return Path(self.project_root) / getattr(self.paths, name)

    def frozen_hash(self) -> str:
        """Hash of all settings that affect results.

        Execution-only settings (which models run in this invocation, worker
        counts, endpoints and timeouts) are excluded so that the same run can be
        continued on another machine or split by model.
        """
        payload = self.model_dump(mode="json")
        payload.pop("project_root", None)
        payload["experiment"].pop("models", None)
        payload["experiment"].pop("max_workers", None)
        # Judging, analysis and auxiliary prompts never change the LLM answers.
        for section in ("judge", "analysis", "table_vision_model"):
            payload.pop(section, None)
        payload["prompt"] = {"version": payload["prompt"]["version"]}
        payload["models"].pop(self.judge.model, None)
        for model in payload["models"].values():
            for key in ("base_url", "api_key_env", "timeout_s", "max_workers"):
                model.pop(key, None)
        payload["retrieval"] = self.retrieval.config_hash()
        return stable_hash(payload)


def parse_partial_date(value: str, end_of_period: bool = True) -> date:
    """Parse 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'.

    Partial dates resolve to the last day of the period when `end_of_period`
    is True (conservative for training cutoffs), otherwise to the first day.
    """
    parts = [int(p) for p in str(value).strip().split("-")]
    if len(parts) == 3:
        return date(*parts)
    year = parts[0]
    if len(parts) == 2:
        month = parts[1]
        day = calendar.monthrange(year, month)[1] if end_of_period else 1
        return date(year, month, day)
    return date(year, 12, 31) if end_of_period else date(year, 1, 1)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(paths: list[str | Path] | str | Path, project_root: str | Path = ".") -> Config:
    """Load and deep-merge one or more YAML config files into a validated `Config`."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    data: dict[str, Any] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as fh:
            data = deep_merge(data, yaml.safe_load(fh) or {})
    data.setdefault("project_root", str(project_root))
    return Config.model_validate(data)


def dump_config(config: Config, path: str | Path) -> None:
    """Write a configuration snapshot as YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)
