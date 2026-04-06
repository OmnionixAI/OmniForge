from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import OmniForgeConfig
from .utils import ensure_dir


def _infer_local_loader(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".txt":
        return "text"
    raise ValueError(f"Unsupported local dataset format for {path}. Use json/jsonl, csv, parquet, or txt.")


def _format_messages(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase | None = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return "\n".join(f"{item.get('role', 'user').title()}: {item.get('content', '').strip()}" for item in messages)


def normalize_record(example: dict[str, Any], config: OmniForgeConfig, tokenizer: PreTrainedTokenizerBase | None = None) -> dict[str, Any]:
    if example.get(config.data.messages_field):
        text = _format_messages(example[config.data.messages_field], tokenizer=tokenizer)
        prompt_length = 0 if config.data.train_on_prompt else max(text.rfind("Assistant:"), 0)
    elif example.get(config.data.prompt_field) is not None and example.get(config.data.response_field) is not None:
        text = f"{example[config.data.prompt_field]}{example[config.data.response_field]}"
        prompt_length = 0 if config.data.train_on_prompt else len(str(example[config.data.prompt_field]))
    elif example.get(config.data.text_field) is not None:
        text = str(example[config.data.text_field])
        prompt_length = 0
    elif example.get("text") is not None:
        text = str(example["text"])
        prompt_length = 0
    else:
        raise ValueError(
            "Each example must contain either messages, prompt+response, or a text field matching the config."
        )

    return {
        "text": text,
        "loss_weight": float(example.get("weight", config.training.weighted_loss.default_weight)),
        "difficulty": float(example.get("difficulty", 0.5)),
        "prompt_length": int(prompt_length),
    }


def tokenize_examples(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    append_eos_token: bool,
) -> dict[str, list[Any]]:
    encoded = tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)
    if append_eos_token and tokenizer.eos_token_id is not None:
        for idx, input_ids in enumerate(encoded["input_ids"]):
            if not input_ids or input_ids[-1] != tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)
                encoded["attention_mask"][idx].append(1)
    encoded["text"] = examples["text"]
    encoded["loss_weight"] = examples["loss_weight"]
    encoded["difficulty"] = examples["difficulty"]
    encoded["prompt_length"] = examples["prompt_length"]
    return encoded


def build_labels(example: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    labels = list(example["input_ids"])
    if example["prompt_length"] > 0:
        prompt_ids = tokenizer(example["text"][: example["prompt_length"]], add_special_tokens=False)["input_ids"]
        for idx in range(min(len(prompt_ids), len(labels))):
            labels[idx] = -100
    example["labels"] = labels
    example["token_weight"] = [float(example["loss_weight"])] * len(labels)
    return example


def pack_dataset(dataset: Dataset, block_size: int) -> Dataset:
    packed_rows: list[dict[str, Any]] = []
    buffer_ids: list[int] = []
    buffer_mask: list[int] = []
    buffer_labels: list[int] = []
    buffer_weights: list[float] = []
    buffer_difficulty: list[float] = []

    for row in dataset:
        ids = list(row["input_ids"])
        mask = list(row["attention_mask"])
        labels = list(row["labels"])
        weights = list(row["token_weight"])
        difficulty = float(row["difficulty"])
        while ids:
            take = min(block_size - len(buffer_ids), len(ids))
            buffer_ids.extend(ids[:take])
            buffer_mask.extend(mask[:take])
            buffer_labels.extend(labels[:take])
            buffer_weights.extend(weights[:take])
            buffer_difficulty.extend([difficulty] * take)
            ids = ids[take:]
            mask = mask[take:]
            labels = labels[take:]
            weights = weights[take:]
            if len(buffer_ids) == block_size:
                packed_rows.append(
                    {
                        "input_ids": buffer_ids,
                        "attention_mask": buffer_mask,
                        "labels": buffer_labels,
                        "token_weight": buffer_weights,
                        "difficulty": sum(buffer_difficulty) / len(buffer_difficulty),
                    }
                )
                buffer_ids, buffer_mask, buffer_labels, buffer_weights, buffer_difficulty = [], [], [], [], []

    if buffer_ids:
        packed_rows.append(
            {
                "input_ids": buffer_ids,
                "attention_mask": buffer_mask,
                "labels": buffer_labels,
                "token_weight": buffer_weights,
                "difficulty": sum(buffer_difficulty) / len(buffer_difficulty),
            }
        )
    return Dataset.from_list(packed_rows)


def _load_local_datasets(config: OmniForgeConfig) -> dict[str, Dataset]:
    cache_dir = ensure_dir(config.data.cache_dir)
    loader = _infer_local_loader(config.data.train_path)
    raw: dict[str, Dataset] = {
        "train": load_dataset(loader, data_files=config.data.train_path, split="train", cache_dir=str(cache_dir))
    }
    if config.data.eval_path:
        eval_loader = _infer_local_loader(config.data.eval_path)
        raw["eval"] = load_dataset(eval_loader, data_files=config.data.eval_path, split="train", cache_dir=str(cache_dir))
    else:
        split = raw["train"].train_test_split(
            test_size=config.data.validation_split_pct / 100.0,
            seed=config.project.seed,
        )
        raw["train"] = split["train"]
        raw["eval"] = split["test"]
    return raw


def _load_hf_datasets(config: OmniForgeConfig) -> dict[str, Dataset]:
    if not config.data.dataset_name:
        raise ValueError("For Hugging Face datasets, set data.dataset_name in the config.")
    cache_dir = ensure_dir(config.data.cache_dir)
    dataset_kwargs = {
        "path": config.data.dataset_name,
        "name": config.data.dataset_config_name,
        "cache_dir": str(cache_dir),
    }
    raw: dict[str, Dataset] = {
        "train": load_dataset(**dataset_kwargs, split=config.data.train_split),
    }
    if config.data.eval_split:
        raw["eval"] = load_dataset(**dataset_kwargs, split=config.data.eval_split)
    else:
        split = raw["train"].train_test_split(
            test_size=config.data.validation_split_pct / 100.0,
            seed=config.project.seed,
        )
        raw["train"] = split["train"]
        raw["eval"] = split["test"]
    return raw


def build_tokenized_datasets(config: OmniForgeConfig, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    source = config.data.source.lower()
    if source == "hf":
        raw = _load_hf_datasets(config)
    elif source == "local":
        raw = _load_local_datasets(config)
    else:
        raise ValueError("data.source must be either 'local' or 'hf'.")

    normalized = DatasetDict()
    for split_name, dataset in raw.items():
        normalized[split_name] = dataset.map(
            lambda example: normalize_record(example, config=config, tokenizer=tokenizer),
            remove_columns=dataset.column_names,
            num_proc=config.data.num_proc,
            desc=f"Normalizing {split_name}",
        )

    tokenized = DatasetDict()
    for split_name, dataset in normalized.items():
        ds = dataset.map(
            lambda batch: tokenize_examples(batch, tokenizer, config.data.max_length, config.data.append_eos_token),
            batched=True,
            num_proc=config.data.num_proc,
            desc=f"Tokenizing {split_name}",
        )
        ds = ds.map(lambda example: build_labels(example, tokenizer), desc=f"Labeling {split_name}")
        keep = {"input_ids", "attention_mask", "labels", "token_weight", "difficulty"}
        ds = ds.remove_columns([col for col in ds.column_names if col not in keep])
        if config.data.packing:
            ds = pack_dataset(ds, config.data.packing_block_size)
        tokenized[split_name] = ds

    return tokenized


def save_dataset_manifest(config: OmniForgeConfig, datasets_dict: DatasetDict) -> Path:
    manifest_path = ensure_dir(config.project.output_dir) / "dataset_manifest.json"
    payload = {
        "app": asdict(config.app),
        "project": config.project.name,
        "splits": {name: len(split) for name, split in datasets_dict.items()},
        "data": asdict(config.data),
        "optimization": asdict(config.optimization),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
