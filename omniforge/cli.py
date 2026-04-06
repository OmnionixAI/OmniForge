from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import platform
import textwrap

import torch
import yaml

from proptrain.config import load_config
from proptrain.modeling import apply_runtime_optimizations, bitsandbytes_available, recommend_optimization_profile, runtime_hardware_summary
from proptrain.pipeline import run_data_prep, run_evaluation, run_export, run_training
from proptrain.utils import detect_runtime

from . import __version__


def _color(enabled: bool, code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _show_banner(enabled: bool) -> None:
    if not enabled:
        return
    banner = """
   ____                 _ ______
  / __ \\____ ___  ____ (_)_/ __/__  _________ ____ 
 / / / / __ `__ \\/ __ `/ / / /_/ _ \\/ ___/ __ `/ _ \\
/ /_/ / / / / / / /_/ / / / __/  __/ /  / /_/ /  __/
\\____/_/ /_/ /_/\\__,_/_/_/_/  \\___/_/   \\__, /\\___/
                                       /____/
"""
    print(_color(enabled, "96", banner.rstrip()))
    print(_color(enabled, "1;97", f"OmniForge {__version__}"))  # bold white
    print(_color(enabled, "90", "Made by Omnionix"))
    print()


def _print_status(label: str, value: str, enabled: bool = True) -> None:
    print(f"{_color(enabled, '94', label + ':'):14} {value}")


def _default_template(model: str, source: str) -> dict:
    local_block = {
        "source": source,
        "train_path": "data/train.jsonl",
        "eval_path": "data/eval.jsonl",
        "dataset_name": None,
        "dataset_config_name": None,
        "train_split": "train",
        "eval_split": "validation",
        "text_field": "text",
        "prompt_field": "prompt",
        "response_field": "response",
        "messages_field": "messages",
        "max_length": 1024,
        "packing": True,
        "packing_block_size": 1024,
        "num_proc": 1,
        "train_on_prompt": False,
        "append_eos_token": True,
        "cache_dir": "cache",
        "validation_split_pct": 5,
    }
    if source == "hf":
        local_block["train_path"] = None
        local_block["eval_path"] = None
        local_block["dataset_name"] = "tatsu-lab/alpaca"
        local_block["dataset_config_name"] = None
        local_block["eval_split"] = None

    return {
        "app": {"vendor": "Omnionix", "app_name": "OmniForge", "version": __version__},
        "cli": {"enabled": True, "startup_banner": True, "color": True},
        "project": {"name": "omniforge-run", "output_dir": "outputs/omniforge-run", "seed": 42},
        "model": {
            "model_name_or_path": model,
            "revision": None,
            "trust_remote_code": False,
            "torch_dtype": "auto",
            "load_in_4bit": True,
            "attn_implementation": "sdpa",
            "use_fast_tokenizer": True,
        },
        "adapter": {"mode": "lora", "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "data": local_block,
        "optimization": {
            "profile": "turbo",
            "auto_profile": True,
            "low_vram_mode": True,
            "notebook_safe": True,
            "torch_compile": False,
            "dataloader_pin_memory": False,
            "gradient_checkpointing_reentrant": False,
            "auto_enable_tf32": True,
        },
        "training": {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.0e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "num_train_epochs": 1,
            "max_steps": -1,
            "lr_scheduler_type": "cosine",
            "logging_steps": 5,
            "eval_steps": 25,
            "save_steps": 25,
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": True,
            "max_grad_norm": 1.0,
            "optimizer": "adamw_torch",
            "report_to": [],
            "save_total_limit": 2,
            "dataloader_num_workers": 0,
            "resume_from_checkpoint": None,
            "curriculum": {"enabled": False, "warmup_fraction": 0.4, "min_difficulty": 0.0, "max_difficulty": 1.0},
            "weighted_loss": {"enabled": True, "default_weight": 1.0},
        },
        "generation": {"max_new_tokens": 128, "temperature": 0.2, "top_p": 0.9},
        "export": {"merge_adapter": False, "output_dir": "outputs/omniforge-run/export"},
    }


def command_init(args: argparse.Namespace) -> int:
    template = _default_template(args.model, args.dataset_source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    print(f"Wrote OmniForge config to {output}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    color = not args.no_color
    _show_banner(not args.no_banner)
    runtime = detect_runtime()
    hardware = runtime_hardware_summary()
    _print_status("Python", platform.python_version(), color)
    _print_status("Platform", platform.platform(), color)
    _print_status("Runtime", runtime, color)
    _print_status("CUDA", str(hardware["cuda"]), color)
    _print_status("GPU count", str(hardware["gpu_count"]), color)
    _print_status("GPU name", str(hardware["gpu_name"]), color)
    _print_status("VRAM (GB)", str(hardware["total_vram_gb"]), color)
    _print_status("bitsandbytes", str(bitsandbytes_available()), color)
    _print_status("Notebook safe", "Yes", color)
    print()
    print(
        textwrap.fill(
            "OmniForge is designed to run locally, on Kaggle, and on Colab. 4-bit loading requires CUDA and bitsandbytes; when unavailable, OmniForge falls back to standard loading instead of hard failing.",
            width=92,
        )
    )
    return 0


def _resolve_config(args: argparse.Namespace):
    config = load_config(args.config)
    if getattr(args, "model", None):
        config.model.model_name_or_path = args.model
    if getattr(args, "train_path", None):
        config.data.source = "local"
        config.data.train_path = args.train_path
    if getattr(args, "eval_path", None) is not None:
        config.data.eval_path = args.eval_path
    if getattr(args, "dataset_name", None):
        config.data.source = "hf"
        config.data.dataset_name = args.dataset_name
    if getattr(args, "dataset_config_name", None) is not None:
        config.data.dataset_config_name = args.dataset_config_name
    if getattr(args, "output_dir", None):
        config.project.output_dir = args.output_dir
    if getattr(args, "resume_from_checkpoint", None):
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
    cli_enabled = config.cli.enabled and os.environ.get("OMNIFORGE_CLI_ENABLED", "1") != "0"
    banner_enabled = cli_enabled and config.cli.startup_banner and not getattr(args, "no_banner", False)
    _show_banner(banner_enabled)
    return config


def _write_temp_config(config, source_path: str) -> str:
    temp_path = Path(config.project.output_dir) / ".omniforge.resolved.yaml"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    return str(temp_path)


def command_prepare(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    config_path = _write_temp_config(config, args.config)
    run_data_prep(config_path)
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    applied = apply_runtime_optimizations(config)
    print(f"OmniForge optimization profile: {applied['profile']}")
    config_path = _write_temp_config(config, args.config)
    run_training(config_path)
    return 0


def command_eval(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    config_path = _write_temp_config(config, args.config)
    run_evaluation(config_path)
    return 0


def command_export(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    config_path = _write_temp_config(config, args.config)
    run_export(config_path)
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    color = not args.no_color
    config = _resolve_config(args)
    recommendation = recommend_optimization_profile(config)
    _print_status("Model", config.model.model_name_or_path, color)
    _print_status("Data source", config.data.source, color)
    _print_status("Profile", recommendation["profile"], color)
    _print_status("DType", recommendation["torch_dtype"], color)
    _print_status("4-bit", str(recommendation["load_in_4bit"]), color)
    _print_status("Grad ckpt", str(recommendation["gradient_checkpointing"]), color)
    _print_status("bf16", str(recommendation["bf16"]), color)
    _print_status("fp16", str(recommendation["fp16"]), color)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniforge", description="OmniForge by Omnionix")
    parser.add_argument("--no-banner", action="store_true", help="Disable the OmniForge startup banner.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored console output.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect runtime compatibility.")
    doctor.set_defaults(func=command_doctor)

    init_cmd = subparsers.add_parser("init", help="Write a starter OmniForge config.")
    init_cmd.add_argument("--output", default="configs/omniforge.yaml", help="Where to write the starter YAML config.")
    init_cmd.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base Hugging Face model path.")
    init_cmd.add_argument("--dataset-source", choices=["local", "hf"], default="local", help="Training dataset source.")
    init_cmd.set_defaults(func=command_init)

    for name, func, help_text in [
        ("prepare", command_prepare, "Prepare and tokenize a dataset."),
        ("train", command_train, "Run fine-tuning."),
        ("eval", command_eval, "Generate a sample response from the trained model."),
        ("export", command_export, "Export merged or adapter artifacts."),
        ("inspect", command_inspect, "Show the resolved optimization plan for a config."),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True, help="Path to an OmniForge YAML config.")
        sub.add_argument("--model", help="Override the base model for this run.")
        sub.add_argument("--train-path", help="Override the local training dataset path.")
        sub.add_argument("--eval-path", help="Override the local evaluation dataset path.")
        sub.add_argument("--dataset-name", help="Override with a Hugging Face dataset name.")
        sub.add_argument("--dataset-config-name", help="Optional Hugging Face dataset config name.")
        sub.add_argument("--output-dir", help="Override the output directory for this run.")
        sub.add_argument("--resume-from-checkpoint", help="Resume training from an existing checkpoint.")
        sub.set_defaults(func=func)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))
