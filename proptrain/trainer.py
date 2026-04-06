from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from .config import OmniForgeConfig


@dataclass
class WeightedCausalCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id

        batch = {"input_ids": [], "attention_mask": [], "labels": [], "token_weight": [], "difficulty": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)
            batch["token_weight"].append(item["token_weight"] + [0.0] * pad_len)
            batch["difficulty"].append(float(item["difficulty"]))

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "token_weight": torch.tensor(batch["token_weight"], dtype=torch.float32),
            "difficulty": torch.tensor(batch["difficulty"], dtype=torch.float32),
        }


class AegisTrainer(Trainer):
    def __init__(self, *args, aegis_config: OmniForgeConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.aegis_config = aegis_config

    def _difficulty_threshold(self) -> float:
        curriculum = self.aegis_config.training.curriculum
        if not curriculum.enabled:
            return curriculum.max_difficulty
        total_steps = max(self.state.max_steps, 1)
        progress = min(max(self.state.global_step / total_steps, 0.0), 1.0)
        if progress >= curriculum.warmup_fraction:
            return curriculum.max_difficulty
        fraction = progress / max(curriculum.warmup_fraction, 1e-6)
        span = curriculum.max_difficulty - curriculum.min_difficulty
        return curriculum.min_difficulty + span * fraction

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        token_weight = inputs.pop("token_weight", None)
        difficulty = inputs.pop("difficulty", None)

        if difficulty is not None and self.aegis_config.training.curriculum.enabled:
            threshold = self._difficulty_threshold()
            keep_mask = (difficulty <= threshold).float().unsqueeze(-1)
            inputs["attention_mask"] = inputs["attention_mask"] * keep_mask.long()
            if token_weight is not None:
                token_weight = token_weight * keep_mask

        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        valid_mask = (shift_labels != -100).float()
        if token_weight is not None and self.aegis_config.training.weighted_loss.enabled:
            shift_weight = token_weight[..., 1:].contiguous().float()
            token_loss = token_loss * shift_weight
            valid_mask = valid_mask * shift_weight.clamp_min(1e-6)

        final_loss = (token_loss * (shift_labels != -100)).sum() / valid_mask.sum().clamp_min(1.0)
        return (final_loss, outputs) if return_outputs else final_loss


def build_training_arguments(config: OmniForgeConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": config.project.output_dir,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "logging_steps": config.training.logging_steps,
        "eval_steps": config.training.eval_steps,
        "save_steps": config.training.save_steps,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "max_grad_norm": config.training.max_grad_norm,
        "optim": config.training.optimizer,
        "report_to": config.training.report_to,
        "save_total_limit": config.training.save_total_limit,
        "dataloader_num_workers": config.training.dataloader_num_workers,
        "dataloader_pin_memory": config.optimization.dataloader_pin_memory,
        "dataloader_persistent_workers": config.optimization.dataloader_persistent_workers,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "remove_unused_columns": False,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "seed": config.project.seed,
    }
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**kwargs)
