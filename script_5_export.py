from __future__ import annotations

import argparse

from proptrain.pipeline import run_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OmniForge artifacts.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_export(args.config)


if __name__ == "__main__":
    main()
