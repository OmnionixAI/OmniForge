from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    vendor: str = "Omnionix"
    app_name: str = "OmniForge"
    version: str = "0.2.5"


@dataclass
class CLIConfig:
    enabled: bool = True
    startup_banner: bool = True
    color: bool = True


@dataclass
class ProjectConfig:
    name: str = "omniforge"
    output_dir: str = "outputs/omniforge"
    seed: int = 42


@dataclass
class ModelConfig:
    model_name_or_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    use_flash_attention_2: bool = False
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"
    use_fast_tokenizer: bool = True


@dataclass
class AdapterConfig:
    mode: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = "all-linear"


@dataclass
class DataConfig:
    source: str = "local"
    train_path: str = "data/train.jsonl"
    eval_path: str | None = "data/eval.jsonl"
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str | None = "validation"
    text_field: str = "text"
    prompt_field: str = "prompt"
    response_field: str = "response"
    messages_field: str = "messages"
    max_length: int = 2048
    packing: bool = True
    packing_block_size: int = 2048
    num_proc: int = 1
    train_on_prompt: bool = False
    append_eos_token: bool = True
    cache_dir: str = "cache"
    validation_split_pct: int = 5


@dataclass
class OptimizationConfig:
    profile: str = "balanced"
    auto_profile: bool = True
    low_vram_mode: bool = True
    notebook_safe: bool = True
    torch_compile: bool = False
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    gradient_checkpointing_reentrant: bool = False
    auto_enable_tf32: bool = True
    max_memory_margin_gb: float = 1.5


@dataclass
class CurriculumConfig:
    enabled: bool = False
    warmup_fraction: float = 0.4
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0


@dataclass
class WeightedLossConfig:
    enabled: bool = True
    default_weight: float = 1.0


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"
    report_to: list[str] = field(default_factory=list)
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    weighted_loss: WeightedLossConfig = field(default_factory=WeightedLossConfig)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9


@dataclass
class ExportConfig:
    merge_adapter: bool = False
    output_dir: str = "outputs/export"


@dataclass
class HubConfig:
    enabled: bool = False
    repo_id: str | None = None
    private: bool = True
    path_in_repo: str = "."
    source: str = "export"
    commit_message: str = "Upload OmniForge artifacts"
    token_env_var: str = "HF_TOKEN"


@dataclass
class GgufConfig:
    enabled: bool = False
    output_dir: str = "outputs/gguf"
    converter_path: str | None = None
    quantize: bool = False
    quantization: str = "Q4_K_M"
    filename: str | None = None
    source: str = "export"


@dataclass
class OmniForgeConfig:
    app: AppConfig = field(default_factory=AppConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    gguf: GgufConfig = field(default_factory=GgufConfig)


AegisConfig = OmniForgeConfig


def _filtered_kwargs(values: dict[str, Any] | None, allowed: set[str]) -> dict[str, Any]:
    values = values or {}
    return {key: value for key, value in values.items() if key in allowed}


def load_config(path: str | Path) -> OmniForgeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    training_raw = raw.get("training") or {}
    return OmniForgeConfig(
        app=AppConfig(**_filtered_kwargs(raw.get("app"), {"vendor", "app_name", "version"})),
        cli=CLIConfig(**_filtered_kwargs(raw.get("cli"), {"enabled", "startup_banner", "color"})),
        project=ProjectConfig(**_filtered_kwargs(raw.get("project"), {"name", "output_dir", "seed"})),
        model=ModelConfig(
            **_filtered_kwargs(
                raw.get("model"),
                {
                    "model_name_or_path",
                    "revision",
                    "trust_remote_code",
                    "torch_dtype",
                    "use_flash_attention_2",
                    "load_in_4bit",
                    "attn_implementation",
                    "use_fast_tokenizer",
                },
            )
        ),
        adapter=AdapterConfig(**_filtered_kwargs(raw.get("adapter"), {"mode", "r", "alpha", "dropout", "target_modules"})),
        data=DataConfig(
            **_filtered_kwargs(
                raw.get("data"),
                {
                    "source",
                    "train_path",
                    "eval_path",
                    "dataset_name",
                    "dataset_config_name",
                    "train_split",
                    "eval_split",
                    "text_field",
                    "prompt_field",
                    "response_field",
                    "messages_field",
                    "max_length",
                    "packing",
                    "packing_block_size",
                    "num_proc",
                    "train_on_prompt",
                    "append_eos_token",
                    "cache_dir",
                    "validation_split_pct",
                },
            )
        ),
        optimization=OptimizationConfig(
            **_filtered_kwargs(
                raw.get("optimization"),
                {
                    "profile",
                    "low_vram_mode",
                    "auto_profile",
                    "notebook_safe",
                    "torch_compile",
                    "dataloader_pin_memory",
                    "dataloader_persistent_workers",
                    "gradient_checkpointing_reentrant",
                    "auto_enable_tf32",
                    "max_memory_margin_gb",
                },
            )
        ),
        training=TrainingConfig(
            **_filtered_kwargs(
                training_raw,
                {
                    "per_device_train_batch_size",
                    "per_device_eval_batch_size",
                    "gradient_accumulation_steps",
                    "learning_rate",
                    "weight_decay",
                    "warmup_ratio",
                    "num_train_epochs",
                    "max_steps",
                    "lr_scheduler_type",
                    "logging_steps",
                    "eval_steps",
                    "save_steps",
                    "bf16",
                    "fp16",
                    "gradient_checkpointing",
                    "max_grad_norm",
                    "optimizer",
                    "report_to",
                    "save_total_limit",
                    "dataloader_num_workers",
                    "resume_from_checkpoint",
                },
            ),
            curriculum=CurriculumConfig(**_filtered_kwargs(training_raw.get("curriculum"), {"enabled", "warmup_fraction", "min_difficulty", "max_difficulty"})),
            weighted_loss=WeightedLossConfig(**_filtered_kwargs(training_raw.get("weighted_loss"), {"enabled", "default_weight"})),
        ),
        generation=GenerationConfig(**_filtered_kwargs(raw.get("generation"), {"max_new_tokens", "temperature", "top_p"})),
        export=ExportConfig(**_filtered_kwargs(raw.get("export"), {"merge_adapter", "output_dir"})),
        hub=HubConfig(**_filtered_kwargs(raw.get("hub"), {"enabled", "repo_id", "private", "path_in_repo", "source", "commit_message", "token_env_var"})),
        gguf=GgufConfig(**_filtered_kwargs(raw.get("gguf"), {"enabled", "output_dir", "converter_path", "quantize", "quantization", "filename", "source"})),
    )
