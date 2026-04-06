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
from proptrain.pipeline import export_gguf, run_data_prep, run_evaluation, run_export, run_training, upload_to_hub
from proptrain.utils import detect_runtime

from . import __version__


MODEL_PRESETS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "distilgpt2",
]

HF_DATASET_PRESETS = [
    "tatsu-lab/alpaca",
    "Abirate/english_quotes",
    "mlabonne/guanaco-llama2-1k",
]


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
    print(_color(enabled, "1;97", f"OmniForge {__version__}"))
    print(_color(enabled, "90", "Made by Omnionix"))
    print()


def _print_status(label: str, value: str, enabled: bool = True) -> None:
    print(f"{_color(enabled, '94', label + ':'):14} {value}")


def _pick_from_list(title: str, options: list[str], allow_custom: bool = True) -> str:
    print(title)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    if allow_custom:
        print(f"{len(options) + 1}. Custom")
    while True:
        choice = input("Select an option: ").strip()
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(options):
                return options[selected - 1]
            if allow_custom and selected == len(options) + 1:
                custom = input("Enter custom value: ").strip()
                if custom:
                    return custom
        print("Enter a valid selection.")


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
            "dataloader_persistent_workers": False,
            "gradient_checkpointing_reentrant": False,
            "auto_enable_tf32": True,
            "max_memory_margin_gb": 1.5,
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
        "hub": {
            "enabled": False,
            "repo_id": None,
            "private": True,
            "path_in_repo": ".",
            "source": "export",
            "commit_message": "Upload OmniForge artifacts",
            "token_env_var": "HF_TOKEN",
        },
        "gguf": {
            "enabled": False,
            "output_dir": "outputs/omniforge-run/gguf",
            "converter_path": None,
            "quantize": False,
            "quantization": "Q4_K_M",
            "filename": None,
            "source": "export",
        },
    }


def command_init(args: argparse.Namespace) -> int:
    template = _default_template(args.model, args.dataset_source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    print(f"Wrote OmniForge config to {output}")
    return 0


def command_select_model(args: argparse.Namespace) -> int:
    model = _pick_from_list("Model presets", MODEL_PRESETS, allow_custom=True)
    print(model)
    return 0


def command_select_dataset(args: argparse.Namespace) -> int:
    source = _pick_from_list("Dataset source", ["local", "hf"], allow_custom=False)
    if source == "local":
        train_path = input("Enter local training dataset path: ").strip()
        eval_path = input("Enter local evaluation dataset path (optional): ").strip()
        print(yaml.safe_dump({"source": "local", "train_path": train_path, "eval_path": eval_path or None}, sort_keys=False).strip())
        return 0
    dataset_name = _pick_from_list("Hugging Face dataset presets", HF_DATASET_PRESETS, allow_custom=True)
    train_split = input("Train split [train]: ").strip() or "train"
    eval_split = input("Eval split [validation]: ").strip() or "validation"
    print(
        yaml.safe_dump(
            {"source": "hf", "dataset_name": dataset_name, "train_split": train_split, "eval_split": eval_split},
            sort_keys=False,
        ).strip()
    )
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
    if getattr(args, "hub_repo_id", None):
        config.hub.repo_id = args.hub_repo_id
    if getattr(args, "hub_private", None) is not None:
        config.hub.private = args.hub_private
    if getattr(args, "hub_source", None):
        config.hub.source = args.hub_source
    if getattr(args, "hub_path_in_repo", None):
        config.hub.path_in_repo = args.hub_path_in_repo
    if getattr(args, "hub_commit_message", None):
        config.hub.commit_message = args.hub_commit_message
    if getattr(args, "hf_token_env_var", None):
        config.hub.token_env_var = args.hf_token_env_var
    if getattr(args, "gguf_output_dir", None):
        config.gguf.output_dir = args.gguf_output_dir
    if getattr(args, "gguf_converter_path", None):
        config.gguf.converter_path = args.gguf_converter_path
    if getattr(args, "gguf_quantize", None):
        config.gguf.quantize = args.gguf_quantize
    if getattr(args, "gguf_quantization", None):
        config.gguf.quantization = args.gguf_quantization
    if getattr(args, "gguf_filename", None):
        config.gguf.filename = args.gguf_filename
    if getattr(args, "gguf_source", None):
        config.gguf.source = args.gguf_source
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


def command_upload(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    config.hub.enabled = True
    config_path = _write_temp_config(config, args.config)
    upload_to_hub(config_path, token=args.hf_token)
    return 0


def command_gguf(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    config.gguf.enabled = True
    config_path = _write_temp_config(config, args.config)
    export_gguf(config_path)
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
    _print_status("Batch size", str(recommendation["per_device_train_batch_size"]), color)
    _print_status("Grad accum", str(recommendation["gradient_accumulation_steps"]), color)
    _print_status("bf16", str(recommendation["bf16"]), color)
    _print_status("fp16", str(recommendation["fp16"]), color)
    _print_status("HF visibility", "private" if config.hub.private else "public", color)
    _print_status("GGUF source", config.gguf.source, color)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniforge", description="OmniForge by Omnionix")
    parser.add_argument("--no-banner", action="store_true", help="Disable the OmniForge startup banner.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored console output.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect runtime compatibility.")
    doctor.set_defaults(func=command_doctor)

    select_model_cmd = subparsers.add_parser("select-model", help="Interactively choose a base model.")
    select_model_cmd.set_defaults(func=command_select_model)

    select_dataset_cmd = subparsers.add_parser("select-dataset", help="Interactively choose a dataset source.")
    select_dataset_cmd.set_defaults(func=command_select_dataset)

    init_cmd = subparsers.add_parser("init", help="Write a starter OmniForge config.")
    init_cmd.add_argument("--output", default="configs/omniforge.yaml", help="Where to write the starter YAML config.")
    init_cmd.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base Hugging Face model path.")
    init_cmd.add_argument("--dataset-source", choices=["local", "hf"], default="local", help="Training dataset source.")
    init_cmd.set_defaults(func=command_init)

    for name, func, help_text in [
        ("prepare", command_prepare, "Prepare and tokenize a dataset."),
        ("train", command_train, "Run fine-tuning."),
        ("eval", command_eval, "Generate a sample response from the trained model."),
        ("export", command_export, "Export merged or adapter artifacts locally."),
        ("upload", command_upload, "Upload local artifacts to Hugging Face Hub."),
        ("gguf", command_gguf, "Convert local/exported model artifacts into GGUF file(s)."),
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
        sub.add_argument("--hub-repo-id", help="Hugging Face repo id like username/repo-name.")
        sub.add_argument("--hub-source", help="Upload source: export, train, or an explicit local path.")
        sub.add_argument("--hub-path-in-repo", help="Target folder path inside the Hugging Face repo.")
        sub.add_argument("--hub-commit-message", help="Commit message for the Hugging Face upload.")
        sub.add_argument("--hf-token", help="Hugging Face write token. Prefer using an environment variable instead.")
        sub.add_argument("--hf-token-env-var", help="Environment variable name to read the Hugging Face token from.")
        sub.add_argument("--hub-private", dest="hub_private", action="store_true", help="Create/upload to a private Hugging Face repo.")
        sub.add_argument("--hub-public", dest="hub_private", action="store_false", help="Create/upload to a public Hugging Face repo.")
        sub.add_argument("--gguf-output-dir", help="Where GGUF files should be written.")
        sub.add_argument("--gguf-converter-path", help="Path to llama.cpp convert_hf_to_gguf.py.")
        sub.add_argument("--gguf-source", help="GGUF source: export, train, or an explicit local path.")
        sub.add_argument("--gguf-filename", help="Output GGUF filename, e.g. model.gguf.")
        sub.add_argument("--gguf-quantize", dest="gguf_quantize", action="store_true", help="Also quantize the GGUF output.")
        sub.add_argument("--no-gguf-quantize", dest="gguf_quantize", action="store_false", help="Disable GGUF quantization.")
        sub.add_argument("--gguf-quantization", help="Quantization preset such as Q4_K_M or Q8_0.")
        sub.set_defaults(func=func, hub_private=None, gguf_quantize=None)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))
