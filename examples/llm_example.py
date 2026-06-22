#!/usr/bin/env python3
"""Small local LLM example with Qwen3-0.6B.

Usage:
    python examples/llm_example.py
    python examples/llm_example.py --prompt "Who are you?"
    python examples/llm_example.py --prompt-file examples/out/stt.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "3rdParty" / "models" / "Qwen3-0.6B"
DEFAULT_MODEL_DIR = ROOT / "3rdParty" / "models" / "Qwen3-4B-Instruct-2507"
CACHE_DIR = ROOT / "3rdParty" / "models" / ".cache" / "numba"

os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer


def build_system_prompt(persona: str) -> str:
    return (
        f"You are {persona}. The user is speaking to you directly.\n"
        "Respond as that persona, not as an assistant describing the persona.\n"
        "Return only valid JSON with exactly these keys: "
        '"talk" and "motion_instruction".\n'
        "talk: reply like a disciplined field soldier. Keep it brief, direct, alert, "
        "mission-focused, and respectful. Use military-style wording when natural, "
        "such as sir, ma'am, copy, understood, standing by, ready, report, command, "
        "secure, or move out. Do not sound like a chatbot or explain the roleplay.\n"
        "motion_instruction: generate a concise third-person motion-generation prompt "
        "that fits the user's message and your in-character reply. The motion_instruction "
        "must be one plain English sentence that starts exactly with 'a person is '. "
        "Describe only visible full-body motion, gesture, or posture. Use generic human "
        "motion words such as standing, walking, nodding, pointing, waving, turning, "
        "stepping forward, stepping back, crouching, looking around, raising arms, "
        "lowering arms, or placing hands behind the back. Pick a different motion "
        "depending on the user's message and your reply. Do not always salute. Use "
        "saluting only when the user greets you, asks who you are, or gives a formal "
        "command. Do not mention the soldier persona, speech, emotions, weapons, "
        "combat, locations, scenery, objects, or camera. Never include location phrases "
        "such as in the field, at the front line, on the ground, near a door, or inside "
        "a room. Focus only on the person's body movement."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona talk and a MoMask motion prompt with Qwen3-0.6B."
    )
    parser.add_argument(
        "--prompt",
        default="Hello, who are you?",
        help="What the user said to the persona.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Text file containing an STT transcript to use as the user prompt.",
    )
    parser.add_argument(
        "--persona",
        default="a soldier",
        help="Persona for the talk field.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map. Use cpu to force CPU.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else MODEL_DIR,
        help="Local Qwen model directory.",
    )
    return parser.parse_args()


def read_user_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return args.prompt


def main() -> None:
    args = parse_args()
    user_prompt = read_user_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map=args.device_map,
    )

    messages = [
        {"role": "system", "content": build_system_prompt(args.persona)},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        outputs = model.generate(**inputs, **generation_kwargs)

    new_tokens = outputs[0][inputs.input_ids.shape[-1] :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
