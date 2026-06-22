#!/usr/bin/env python3
"""MeloTTS English example.

Usage:
    python examples/tts_example.py
    python examples/tts_example.py --text "The robot picks up the cup and places it on the table."
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "3rdParty" / "models" / "MeloTTS-English"
DEFAULT_OUTPUT = ROOT / "examples" / "out" / "tts_en.wav"
CACHE_DIR = ROOT / "3rdParty" / "models" / ".cache" / "numba"

os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR))

from melo.api import TTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English speech with MeloTTS.")
    parser.add_argument(
        "--text",
        default="Hello. This is a local text to speech test.",
        help="English text to synthesize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output wav path.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument(
        "--speaker",
        default="EN-US",
        choices=("EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"),
        help="English MeloTTS speaker.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda:0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = TTS(
        language="EN",
        device=args.device,
        config_path=str(MODEL_DIR / "config.json"),
        ckpt_path=str(MODEL_DIR / "checkpoint.pth"),
    )
    speaker_id = model.hps.data.spk2id[args.speaker]
    model.tts_to_file(args.text, speaker_id, str(args.output), speed=args.speed)

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
