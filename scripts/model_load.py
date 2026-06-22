#!/usr/bin/env python3
"""Preload STT, LLM, and TTS models for the interactive demo.

The intent is to import and instantiate heavy models once at demo startup, then
reuse the same Python objects for every user turn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "3rdParty" / "models"
NUMBA_CACHE_DIR = MODEL_ROOT / ".cache" / "numba"

os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))


@dataclass
class STTConfig:
    model_dir: Path = MODEL_ROOT / "faster-whisper-small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"


@dataclass
class LLMConfig:
    model_dir: Path = (
        MODEL_ROOT / "Qwen3-4B-Instruct-2507"
        if (MODEL_ROOT / "Qwen3-4B-Instruct-2507").exists()
        else MODEL_ROOT / "Qwen3-0.6B"
    )
    persona: str = "a soldier"
    device_map: str = "auto"
    max_new_tokens: int = 96
    temperature: float = 0.0


@dataclass
class TTSConfig:
    engine: str = "piper"
    model_dir: Path = MODEL_ROOT / "MeloTTS-English"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    speaker: str = "EN-BR"
    speed: float = 1.0
    piper_model: Path = MODEL_ROOT / "Piper" / "en_US-ryan-medium.onnx"
    piper_config: Path = MODEL_ROOT / "Piper" / "en_US-ryan-medium.onnx.json"
    piper_executable: str = "piper"


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
        "Describe only visible full-body motion, gesture, or posture. Pick a different "
        "motion depending on the user's message and your reply. Do not always salute. "
        "Do not mention the soldier persona, speech, emotions, weapons, combat, "
        "locations, scenery, objects, or camera. Focus only on body movement."
    )


def parse_llm_json(text: str) -> dict[str, str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        data = json.loads(match.group(0))

    talk = str(data["talk"]).strip()
    motion_instruction = str(data["motion_instruction"]).strip()
    return {"talk": talk, "motion_instruction": motion_instruction}


class STTRuntime:
    def __init__(self, config: STTConfig = STTConfig()) -> None:
        from faster_whisper import WhisperModel

        self.config = config
        self.model = WhisperModel(
            str(config.model_dir),
            device=config.device,
            compute_type=config.compute_type,
        )

    def transcribe(self, audio_path: str | Path) -> str:
        segments, _info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class LLMRuntime:
    def __init__(self, config: LLMConfig = LLMConfig()) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_dir,
            torch_dtype="auto",
            device_map=config.device_map,
        )

    def generate_text(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": build_system_prompt(self.config.persona)},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.temperature > 0,
            }
            if self.config.temperature > 0:
                generation_kwargs["temperature"] = self.config.temperature
            outputs = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = outputs[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_json(self, user_prompt: str) -> dict[str, str]:
        return parse_llm_json(self.generate_text(user_prompt))


class TTSRuntime:
    def __init__(self, config: TTSConfig = TTSConfig()) -> None:
        self.config = config
        self.engine = config.engine.lower()
        if self.engine == "piper":
            env_piper = Path(sys.executable).resolve().parent / config.piper_executable
            self.piper_executable = shutil.which(config.piper_executable)
            if self.piper_executable is None and env_piper.exists():
                self.piper_executable = str(env_piper)
            if self.piper_executable is None:
                raise FileNotFoundError(
                    "Piper executable not found. Install it with "
                    "`python -m pip install piper-tts` in the demo environment."
                )
            if not config.piper_model.exists():
                raise FileNotFoundError(f"Piper model not found: {config.piper_model}")
            if not config.piper_config.exists():
                raise FileNotFoundError(f"Piper config not found: {config.piper_config}")
            return

        if self.engine == "melo":
            from melo.api import TTS

            self.model = TTS(
                language="EN",
                device=config.device,
                config_path=str(config.model_dir / "config.json"),
                ckpt_path=str(config.model_dir / "checkpoint.pth"),
            )
            self.speaker_id = self.model.hps.data.spk2id[config.speaker]
            return

        raise ValueError(f"Unsupported TTS engine: {config.engine}")

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.engine == "piper":
            subprocess.run(
                [
                    self.piper_executable,
                    "--model",
                    str(self.config.piper_model),
                    "--config",
                    str(self.config.piper_config),
                    "--output_file",
                    str(output),
                ],
                input=text,
                text=True,
                check=True,
            )
        else:
            self.model.tts_to_file(
                text,
                self.speaker_id,
                str(output),
                speed=self.config.speed,
            )
        return output


class DemoModels:
    def __init__(
        self,
        load_stt: bool = True,
        load_llm: bool = True,
        load_tts: bool = True,
        stt_config: STTConfig = STTConfig(),
        llm_config: LLMConfig = LLMConfig(),
        tts_config: TTSConfig = TTSConfig(),
    ) -> None:
        self.stt = STTRuntime(stt_config) if load_stt else None
        self.llm = LLMRuntime(llm_config) if load_llm else None
        self.tts = TTSRuntime(tts_config) if load_tts else None

    def run_text_turn(self, user_text: str, tts_output: str | Path) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded")
        result = self.llm.generate_json(user_text)

        wav_path = None
        if self.tts is not None:
            wav_path = self.tts.synthesize(result["talk"], tts_output)

        return {
            "user_text": user_text,
            "talk": result["talk"],
            "motion_instruction": result["motion_instruction"],
            "tts_output": str(wav_path) if wav_path else None,
        }

    def run_audio_turn(self, audio_path: str | Path, tts_output: str | Path) -> dict[str, Any]:
        if self.stt is None:
            raise RuntimeError("STT is not loaded")
        user_text = self.stt.transcribe(audio_path)
        return self.run_text_turn(user_text, tts_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload demo models and run a smoke turn.")
    parser.add_argument("--no-stt", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--text", default="Report your status.")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--tts-output", type=Path, default=ROOT / "outputs" / "demo" / "talk.wav")
    parser.add_argument("--llm-device-map", default="auto")
    parser.add_argument("--stt-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--stt-compute-type", default="int8")
    parser.add_argument("--tts-device", default=TTSConfig.device)
    parser.add_argument("--tts-engine", default=TTSConfig.engine, choices=("piper", "melo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = DemoModels(
        load_stt=not args.no_stt,
        load_llm=not args.no_llm,
        load_tts=not args.no_tts,
        stt_config=STTConfig(device=args.stt_device, compute_type=args.stt_compute_type),
        llm_config=LLMConfig(device_map=args.llm_device_map),
        tts_config=TTSConfig(engine=args.tts_engine, device=args.tts_device),
    )

    if args.no_llm:
        print("Loaded requested models. No smoke turn because --no-llm was set.")
        return

    if args.audio:
        result = models.run_audio_turn(args.audio, args.tts_output)
    else:
        result = models.run_text_turn(args.text, args.tts_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
