from __future__ import annotations

import argparse

from proptrain.pipeline import run_data_prep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for OmniForge.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_data_prep(args.config)


if __name__ == "__main__":
    main()
