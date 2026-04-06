from __future__ import annotations

import inspect
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig as HFGenerationConfig

from .config import load_config
from .data import build_tokenized_datasets, save_dataset_manifest
from .modeling import apply_runtime_optimizations, load_model, load_tokenizer, maybe_merge_adapter, runtime_hardware_summary
from .trainer import AegisTrainer, WeightedCausalCollator, build_training_arguments
from .utils import detect_runtime, ensure_dir, save_json, set_seed


def _has_exportable_model_artifacts(source_dir: Path, adapter_mode: str) -> bool:
    if not source_dir.exists():
        return False
    mode = adapter_mode.lower()
    if mode == "full":
        return (source_dir / "config.json").exists() and any(
            (source_dir / filename).exists() for filename in ("model.safetensors", "pytorch_model.bin")
        )
    return (source_dir / "adapter_config.json").exists() and any(
        (source_dir / filename).exists() for filename in ("adapter_model.safetensors", "adapter_model.bin")
    )


def run_data_prep(config_path: str):
    config = load_config(config_path)
    applied = apply_runtime_optimizations(config)
    set_seed(config.project.seed)
    tokenizer = load_tokenizer(config)
    datasets = build_tokenized_datasets(config, tokenizer)
    manifest = save_dataset_manifest(config, datasets)
    save_json(
        Path(config.project.output_dir) / "runtime_profile.json",
        {"runtime": detect_runtime(), "hardware": runtime_hardware_summary(), "applied_optimizations": applied},
    )
    print(f"Prepared dataset manifest at {manifest}")
    for split_name, split in datasets.items():
        print(f"{split_name}: {len(split)} rows")
    return datasets


def run_training(config_path: str):
    config = load_config(config_path)
    applied = apply_runtime_optimizations(config)
    set_seed(config.project.seed)
    ensure_dir(config.project.output_dir)

    tokenizer = load_tokenizer(config)
    datasets = build_tokenized_datasets(config, tokenizer)
    model = load_model(config)
    collator = WeightedCausalCollator(tokenizer=tokenizer)
    trainer_kwargs = {
        "model": model,
        "args": build_training_arguments(config),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets.get("eval"),
        "data_collator": collator,
        "aegis_config": config,
    }
    if "processing_class" in inspect.signature(AegisTrainer.__mro__[1].__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = AegisTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_model(config.project.output_dir)
    tokenizer.save_pretrained(config.project.output_dir)
    metrics = trainer.evaluate() if datasets.get("eval") is not None else {}
    save_json(Path(config.project.output_dir) / "train_metrics.json", metrics)
    save_json(
        Path(config.project.output_dir) / "run_summary.json",
        {
            "runtime": detect_runtime(),
            "hardware": runtime_hardware_summary(),
            "applied_optimizations": applied,
            "project_output_dir": config.project.output_dir,
            "model_name_or_path": config.model.model_name_or_path,
        },
    )
    return trainer


def run_evaluation(config_path: str):
    config = load_config(config_path)
    apply_runtime_optimizations(config)
    tokenizer = load_tokenizer(config)
    source_dir = Path(config.project.output_dir)
    if _has_exportable_model_artifacts(source_dir, config.adapter.mode) and config.adapter.mode.lower() != "full":
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path)
        model = PeftModel.from_pretrained(base_model, str(source_dir))
    elif _has_exportable_model_artifacts(source_dir, config.adapter.mode):
        model = AutoModelForCausalLM.from_pretrained(str(source_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path)
    model.eval()

    prompt = "User: Explain why evaluation checkpoints matter during fine-tuning.\nAssistant:"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_config = HFGenerationConfig(
        max_new_tokens=config.generation.max_new_tokens,
        top_p=config.generation.top_p,
        do_sample=config.generation.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if config.generation.temperature > 0:
        generation_config.temperature = config.generation.temperature
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=generation_config)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    results = {"prompt": prompt, "generated_text": text}
    output_path = Path(config.project.output_dir) / "evaluation_samples.json"
    save_json(output_path, results)
    print(text)
    print(f"Saved evaluation sample to {output_path}")
    return results


def run_export(config_path: str):
    config = load_config(config_path)
    apply_runtime_optimizations(config)
    export_dir = ensure_dir(config.export.output_dir)
    tokenizer = load_tokenizer(config)
    source_dir = Path(config.project.output_dir)
    if not _has_exportable_model_artifacts(source_dir, config.adapter.mode):
        raise FileNotFoundError(
            f"No trained model artifacts were found in {source_dir}. Run training first or point the config at a completed output directory."
        )

    if config.adapter.mode.lower() == "full":
        model = AutoModelForCausalLM.from_pretrained(str(source_dir))
    else:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path)
        model = PeftModel.from_pretrained(base_model, str(source_dir))

    if config.export.merge_adapter:
        model = maybe_merge_adapter(model)

    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"Exported model artifacts to {export_dir}")
    return export_dir
