import argparse
import os
import queue
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from model_load import *


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from momask_runtime import MomaskRuntime


CHANNELS = 1
MIC_WAV = ROOT / "scripts" / "mic_input.wav"
TTS_WAV = ROOT / "outputs" / "talk.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record voice, generate MoMask motion, and preview it in Blender."
    )
    parser.add_argument(
        "--blender",
        default=None,
        help="Path to the Blender executable. If omitted, uses $BLENDER or PATH.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=-1,
        help="GPU id for MoMask. Use -1 for CPU.",
    )
    parser.add_argument(
        "--momask-ext",
        default="demo_run",
        help="MoMask output run name under 3rdParty/momask-codes/generation/.",
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help="Sounddevice input device id or name. If omitted, uses the default input.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=None,
        help="Microphone sample rate. If omitted, uses the input device default.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    return parser.parse_args()


def resolve_blender_executable(blender_arg: str | None) -> str:
    explicit_candidates = [
        ("--blender", blender_arg),
        ("BLENDER", os.environ.get("BLENDER")),
    ]

    for source, candidate in explicit_candidates:
        if not candidate:
            continue

        expanded = Path(candidate).expanduser()
        if expanded.exists():
            if not expanded.is_file():
                raise FileNotFoundError(f"{source} is not a file: {expanded}")
            if not os.access(expanded, os.X_OK):
                raise PermissionError(f"{source} is not executable: {expanded}")
            return str(expanded.resolve())

        resolved = shutil.which(candidate)
        if resolved:
            return resolved

        raise FileNotFoundError(f"{source} does not point to Blender: {candidate}")

    resolved = shutil.which("blender")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "Blender executable not found. Pass --blender /path/to/blender, "
        "set BLENDER=/path/to/blender, or add blender to PATH."
    )


def resolve_input_device(device_arg: str | None) -> int | str | None:
    if device_arg is None:
        return None
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def resolve_samplerate(input_device: int | str | None, samplerate_arg: int | None) -> int:
    if samplerate_arg is not None:
        return samplerate_arg

    device_info = sd.query_devices(input_device, "input")
    return int(device_info["default_samplerate"])


def record_audio(
    output_path: Path,
    samplerate: int,
    input_device: int | str | None,
) -> bool:
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    input("Press Enter to start recording...")
    print(f"Recording at {samplerate} Hz... Press Enter to stop.")

    recorded_chunks = []
    try:
        with sd.InputStream(
            samplerate=samplerate,
            device=input_device,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
        ):
            input()
            print("Stopping...")

            while not audio_queue.empty():
                recorded_chunks.append(audio_queue.get())
    except sd.PortAudioError as exc:
        print(f"Could not open microphone input stream: {exc}")
        print("Run with --list-audio-devices, then retry with --input-device DEVICE_ID.")
        return False

    if not recorded_chunks:
        print("No audio recorded.")
        return False

    audio = np.concatenate(recorded_chunks, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, samplerate)
    print(f"Saved to {output_path}")
    return True


def preview_in_blender(
    blender_executable: str,
    ik_bvh: Path,
    tts_audio: str | Path,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            blender_executable,
            "--factory-startup",
            "--python",
            str(ROOT / "scripts" / "blender_keemap_retarget_preview.py"),
            "--",
            "--addon-dir",
            str(ROOT / "3rdParty" / "KeeMapAnimRetarget"),
            "--bvh",
            str(ik_bvh),
            "--fbx",
            str(ROOT / "asset" / "Ch15_nonPBR.fbx"),
            "--mapping",
            str(ROOT / "3rdParty" / "momask-codes" / "assets" / "mapping.json"),
            "--tts-audio",
            str(tts_audio),
            "--viewport-shading",
            "MATERIAL",
        ]
    )


def run_turn(
    models: DemoModels,
    momask: MomaskRuntime,
    blender_executable: str,
    samplerate: int,
    input_device: int | str | None,
) -> subprocess.Popen | None:
    if not record_audio(MIC_WAV, samplerate, input_device):
        return None

    result = models.run_audio_turn(MIC_WAV, TTS_WAV)
    motion_prompt = result["motion_instruction"]
    print("User text:", result["user_text"])
    print("Talk:", result["talk"])
    print("Motion:", motion_prompt)

    motion = momask.generate(motion_prompt)
    print(f"Opening Blender preview: {motion['ik_bvh']}")
    return preview_in_blender(
        blender_executable,
        Path(motion["ik_bvh"]),
        result["tts_output"],
    )


def main() -> None:
    args = parse_args()
    if args.list_audio_devices:
        print(sd.query_devices())
        return

    input_device = resolve_input_device(args.input_device)
    samplerate = resolve_samplerate(input_device, args.samplerate)
    blender_executable = resolve_blender_executable(args.blender)

    print("Loading demo models...")
    models = DemoModels(tts_config=TTSConfig(engine="piper"))
    momask = MomaskRuntime(
        gpu_id=args.gpu_id,
        ext=args.momask_ext,
        write_preview_mp4=False,
    )
    print(f"Blender: {blender_executable}")
    print(f"Audio input device: {input_device if input_device is not None else 'default'}")
    print(f"Audio sample rate: {samplerate}")
    print("Ready.")

    blender_proc: subprocess.Popen | None = None
    while True:
        command = input("Press r to record/open Blender preview, or q to quit: ").strip().lower()
        if command == "q":
            break
        if command != "r":
            continue

        if blender_proc is not None and blender_proc.poll() is None:
            blender_proc.terminate()
        blender_proc = run_turn(models, momask, blender_executable, samplerate, input_device)


if __name__ == "__main__":
    main()
