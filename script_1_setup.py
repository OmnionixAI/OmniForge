from __future__ import annotations

import platform
import sys


def main() -> None:
    print("OmniForge setup")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print()
    print("Install dependencies with:")
    print("python -m pip install -r requirements.txt")
    print("python -m pip install -e .")
    print()
    print("CLI flow:")
    print("python -m omniforge doctor")
    print("python -m omniforge train --config configs/example_sft.yaml")


if __name__ == "__main__":
    main()
