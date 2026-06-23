#!/usr/bin/env python3
"""Download Hugging Face models needed by the demo.

This avoids the deprecated `huggingface-cli` command and uses the
huggingface_hub Python API directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "3rdParty" / "models"

WHISPER_REPO = "Systran/faster-whisper-small"
QWEN_REPOS = {
    "4b": "Qwen/Qwen3-4B-Instruct-2507",
    "0.6b": "Qwen/Qwen3-0.6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download faster-whisper and Qwen models for the demo."
    )
    parser.add_argument(
        "--qwen",
        choices=sorted(QWEN_REPOS),
        default="4b",
        help="Qwen model size to download.",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Do not download faster-whisper-small.",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Do not download Qwen.",
    )
    return parser.parse_args()


def download_repo(repo_id: str, local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Activate the conda environment "
            "or install requirements with `pip install -r requirements.txt`."
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}")
    print(f"Target: {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
    )


def main() -> None:
    args = parse_args()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_whisper:
        download_repo(WHISPER_REPO, MODEL_ROOT / "faster-whisper-small")

    if not args.skip_qwen:
        qwen_repo = QWEN_REPOS[args.qwen]
        qwen_dir = qwen_repo.split("/", 1)[1]
        download_repo(qwen_repo, MODEL_ROOT / qwen_dir)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
