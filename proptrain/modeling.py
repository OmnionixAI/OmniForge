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
