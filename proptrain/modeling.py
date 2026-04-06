from __future__ import annotations

import importlib.util
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import OmniForgeConfig


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, "auto")


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def runtime_hardware_summary() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    summary: dict[str, Any] = {
        "cuda": cuda,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
        "bitsandbytes_available": bitsandbytes_available(),
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        summary["gpu_name"] = props.name
        summary["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
    else:
        summary["gpu_name"] = None
        summary["total_vram_gb"] = 0.0
    return summary


def recommend_optimization_profile(config: OmniForgeConfig) -> dict[str, Any]:
    summary = runtime_hardware_summary()
    recommendation: dict[str, Any] = {
        "profile": config.optimization.profile,
        "torch_dtype": config.model.torch_dtype,
        "load_in_4bit": config.model.load_in_4bit,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "dataloader_pin_memory": config.optimization.dataloader_pin_memory,
        "dataloader_persistent_workers": config.optimization.dataloader_persistent_workers,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "attn_implementation": config.model.attn_implementation,
    }
    if not config.optimization.auto_profile:
        return recommendation

    if not summary["cuda"]:
        recommendation.update(
            {
                "profile": "cpu-safe",
                "torch_dtype": "float32",
                "load_in_4bit": False,
                "gradient_checkpointing": False,
                "dataloader_pin_memory": False,
                "dataloader_persistent_workers": False,
                "bf16": False,
                "fp16": False,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": max(config.training.gradient_accumulation_steps, 1),
                "attn_implementation": "eager",
            }
        )
        return recommendation

    vram = float(summary["total_vram_gb"])
    if vram <= 16:
        recommendation.update(
            {
                "profile": "turbo-low-vram",
                "load_in_4bit": True,
                "gradient_checkpointing": True,
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": config.training.dataloader_num_workers > 0,
                "bf16": bool(summary["bf16_supported"]),
                "fp16": not bool(summary["bf16_supported"]),
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": max(config.training.gradient_accumulation_steps, 16),
                "attn_implementation": "sdpa",
            }
        )
    else:
        recommendation.update(
            {
                "profile": "throughput",
                "load_in_4bit": config.model.load_in_4bit,
                "gradient_checkpointing": False if vram >= 40 else config.training.gradient_checkpointing,
                "dataloader_pin_memory": True,
                "dataloader_persistent_workers": config.training.dataloader_num_workers > 0,
                "bf16": bool(summary["bf16_supported"]),
                "fp16": not bool(summary["bf16_supported"]),
                "per_device_train_batch_size": max(config.training.per_device_train_batch_size, 2 if vram < 40 else 4),
                "gradient_accumulation_steps": max(1, min(config.training.gradient_accumulation_steps, 8 if vram < 40 else 4)),
                "attn_implementation": "sdpa",
            }
        )
    if recommendation["bf16"]:
        recommendation["torch_dtype"] = "bfloat16"
    elif recommendation["fp16"]:
        recommendation["torch_dtype"] = "float16"
    return recommendation


def apply_runtime_optimizations(config: OmniForgeConfig) -> dict[str, Any]:
    recommendation = recommend_optimization_profile(config)
    config.optimization.profile = recommendation["profile"]
    config.model.torch_dtype = recommendation["torch_dtype"]
    config.model.load_in_4bit = recommendation["load_in_4bit"]
    config.training.gradient_checkpointing = recommendation["gradient_checkpointing"]
    config.optimization.dataloader_pin_memory = recommendation["dataloader_pin_memory"]
    config.optimization.dataloader_persistent_workers = recommendation["dataloader_persistent_workers"]
    config.training.per_device_train_batch_size = recommendation["per_device_train_batch_size"]
    config.training.gradient_accumulation_steps = recommendation["gradient_accumulation_steps"]
    config.training.bf16 = recommendation["bf16"]
    config.training.fp16 = recommendation["fp16"]
    config.model.attn_implementation = recommendation["attn_implementation"]
    return recommendation


def load_tokenizer(config: OmniForgeConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=config.model.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _build_quantization_config(config: OmniForgeConfig):
    if not config.model.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        print("OmniForge notice: 4-bit loading was requested but CUDA is unavailable, so standard loading will be used.")
        return None
    if not bitsandbytes_available():
        print("OmniForge notice: bitsandbytes is not installed, so standard loading will be used.")
        return None

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def _maybe_enable_perf_flags(config: OmniForgeConfig) -> None:
    if torch.cuda.is_available() and config.optimization.auto_enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_model(config: OmniForgeConfig):
    _maybe_enable_perf_flags(config)
    quantization_config = _build_quantization_config(config)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "revision": config.model.revision,
        "torch_dtype": resolve_torch_dtype(config.model.torch_dtype),
        "attn_implementation": config.model.attn_implementation,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path, **model_kwargs)
    if config.training.gradient_checkpointing:
        gradient_kwargs = {"use_reentrant": config.optimization.gradient_checkpointing_reentrant}
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config.optimization.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"OmniForge notice: torch.compile was skipped: {exc}")

    if config.adapter.mode.lower() != "full":
        lora_config = LoraConfig(
            r=config.adapter.r,
            lora_alpha=config.adapter.alpha,
            lora_dropout=config.adapter.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config.adapter.target_modules,
        )
        model = get_peft_model(model, lora_config)

    return model


def maybe_merge_adapter(model):
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model
