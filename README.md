# OmniForge

OmniForge is a CLI for fine-tuning Hugging Face language models on either:

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
- exports adapters or merged artifacts locally
- uploads finished artifacts to Hugging Face Hub when you provide a write token
- supports public or private Hugging Face repos
- converts finished models into GGUF file(s) through a local `llama.cpp` converter path
- provides a branded CLI with optional startup banner and colors
- auto-recommends runtime optimization profiles based on available hardware
- supports CLI overrides for model, dataset, output directory, and checkpoint resume
- includes interactive model and dataset selection helpers

## Production Focus

OmniForge is built around production-ready workflows:

- deterministic local artifact storage
- explicit export, upload, and GGUF conversion steps
- runtime-aware optimization presets for low-VRAM and throughput-oriented hardware
- notebook-friendly execution for Kaggle and Colab
- optional Hugging Face Hub publishing with public or private visibility

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## CLI

```bash
python -m omniforge doctor
python -m omniforge select-model
python -m omniforge select-dataset
python -m omniforge inspect --config configs/example_sft.yaml
python -m omniforge init --output configs/omniforge.yaml --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset-source local
python -m omniforge prepare --config configs/example_sft.yaml
python -m omniforge train --config configs/example_sft.yaml
python -m omniforge eval --config configs/example_sft.yaml
python -m omniforge export --config configs/example_sft.yaml
python -m omniforge upload --config configs/example_sft.yaml --hub-repo-id yourname/omniforge-model --hub-private
python -m omniforge gguf --config configs/example_sft.yaml --gguf-converter-path C:\path\to\llama.cpp\convert_hf_to_gguf.py
```

## Local vs Hub storage

OmniForge still stores artifacts locally by default.

- local training outputs go to `project.output_dir`
- local exported model artifacts go to `export.output_dir`
- local GGUF files go to `gguf.output_dir`
- Hugging Face upload only happens when you explicitly run `upload` or set up a config for that workflow

## Hugging Face upload

Set a write token in an environment variable such as `HF_TOKEN`, then run:

```bash
python -m omniforge upload --config configs/example_sft.yaml --hub-repo-id yourname/your-repo --hub-private
python -m omniforge upload --config configs/example_sft.yaml --hub-repo-id yourname/your-repo --hub-public
```

You can also choose what gets uploaded:

- `--hub-source export`
- `--hub-source train`
- `--hub-source C:\some\local\folder`

## GGUF export

GGUF export needs a local `llama.cpp` converter path.

GGUF is optional. If you do not run the `gguf` command, OmniForge keeps your outputs in the normal local Hugging Face-style format and can still upload those artifacts to Hugging Face Hub.

Example:

```bash
python -m omniforge gguf --config configs/example_sft.yaml --gguf-converter-path C:\llama.cpp\convert_hf_to_gguf.py
python -m omniforge gguf --config configs/example_sft.yaml --gguf-converter-path C:\llama.cpp\convert_hf_to_gguf.py --gguf-quantize --gguf-quantization Q4_K_M
```

Notes:

- if your model is adapter-based, OmniForge will prepare a merged Hugging Face model directory for GGUF conversion when needed
- quantization also expects `llama-quantize` next to the converter

## Kaggle and Colab

Typical notebook flow:

```python
!pip install -r requirements.txt
!pip install -e .
!python -m omniforge doctor
!python -m omniforge train --config configs/example_sft.yaml
!python -m omniforge export --config configs/example_sft.yaml
```

## Proprietary posture

This repository includes an `All Rights Reserved` notice in `PROPRIETARY_NOTICE.txt`. Third-party packages, datasets, base model weights, and any uploaded Hub repos remain subject to their own licenses and platform rules.

