# OmniForge

OmniForge is a proprietary-oriented CLI for fine-tuning Hugging Face language models on either:

- local datasets
- Hugging Face datasets

Branding:

- App: `OmniForge`
- Vendor: `Omnionix`
- Version: `0.2.5`

It is designed to run:

- locally
- in Google Colab
- on Kaggle

## What OmniForge does

- trains any Hugging Face causal LM you can load through `transformers`
- supports `full`, `lora`, and optional 4-bit LoRA-style loading
- accepts local JSONL/JSON/CSV/Parquet/TXT datasets
- accepts Hugging Face datasets by name and split
- supports prompt masking, weighted loss, curriculum gating, and sequence packing
- exports adapters or merged artifacts
- provides a branded CLI with optional startup banner and colors
- auto-recommends runtime optimization profiles based on available hardware
- supports CLI overrides for model, dataset, output directory, and checkpoint resume

## Important honesty note

This repository includes stronger optimization defaults, but it does **not** prove universal `90% less VRAM` and `5x faster` results across every model, GPU, and dataset. Those numbers depend on model size, precision, adapter mode, hardware, sequence length, and batch shape. OmniForge does include the practical building blocks usually used to pursue those gains:

- LoRA adapters
- optional 4-bit loading when `bitsandbytes` is available
- gradient checkpointing
- sequence packing
- TF32 enablement where available
- notebook-safe fallback behavior instead of hard crashes

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## CLI

```bash
python -m omniforge doctor
python -m omniforge inspect --config configs/example_sft.yaml
python -m omniforge init --output configs/omniforge.yaml --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset-source local
python -m omniforge prepare --config configs/example_sft.yaml
python -m omniforge train --config configs/example_sft.yaml
python -m omniforge eval --config configs/example_sft.yaml
python -m omniforge export --config configs/example_sft.yaml
```

After `pip install -e .`, you can also use:

```bash
omniforge doctor
omniforge train --config configs/example_sft.yaml
```

Example one-off overrides without editing YAML:

```bash
python -m omniforge train --config configs/example_sft.yaml --model Qwen/Qwen2.5-0.5B-Instruct --train-path data/train.jsonl --output-dir outputs/qwen-run
python -m omniforge train --config configs/hf_dataset_example.yaml --dataset-name tatsu-lab/alpaca
```

## Startup banner

You can activate or deactivate the CLI presentation at startup:

- config: `cli.enabled: true|false`
- config: `cli.startup_banner: true|false`
- env override: `OMNIFORGE_CLI_ENABLED=0`
- runtime flag: `--no-banner`

## Config examples

- [configs/example_sft.yaml](C:\JV\Other (Coding)\Proprietary Training Script\configs\example_sft.yaml): local dataset example
- [configs/hf_dataset_example.yaml](C:\JV\Other (Coding)\Proprietary Training Script\configs\hf_dataset_example.yaml): Hugging Face dataset example
- [configs/smoke_test.yaml](C:\JV\Other (Coding)\Proprietary Training Script\configs\smoke_test.yaml): tiny end-to-end validation config

## Local dataset format

Each row can use one of these shapes:

```json
{"messages":[{"role":"system","content":"You are precise."},{"role":"user","content":"Explain LoRA."},{"role":"assistant","content":"LoRA trains low-rank adapters instead of updating every base weight."}],"weight":1.0,"difficulty":0.2}
```

```json
{"prompt":"User: Explain gradient accumulation.\nAssistant:","response":" It simulates a larger batch by summing gradients across smaller steps.","weight":1.0}
```

```json
{"text":"Question: What is sequence packing?\nAnswer: It reduces padding waste by concatenating shorter samples into fixed blocks."}
```

## Kaggle and Colab

OmniForge is built to degrade gracefully:

- if CUDA is unavailable, it falls back to normal loading
- if `bitsandbytes` is unavailable, 4-bit loading is skipped instead of crashing
- notebook-safe defaults avoid assuming desktop-only behavior
- runtime-aware optimization profiles adapt dtype, 4-bit loading, and memory settings automatically

Typical notebook flow:

```python
!pip install -r requirements.txt
!pip install -e .
!python -m omniforge doctor
!python -m omniforge train --config configs/example_sft.yaml
```

## Proprietary posture

This repository includes an `All Rights Reserved` notice in [PROPRIETARY_NOTICE.txt](C:\JV\Other (Coding)\Proprietary Training Script\PROPRIETARY_NOTICE.txt). Third-party packages, datasets, and base model weights remain subject to their own licenses.

