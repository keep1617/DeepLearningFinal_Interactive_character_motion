import queue
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


BLENDER = "/home/adfa5456/Downloads/blender-5.0.1-linux-x64/blender"
SAMPLERATE = 16000
CHANNELS = 1
MIC_WAV = ROOT / "scripts" / "mic_input.wav"
TTS_WAV = ROOT / "outputs" / "talk.wav"


def record_audio(output_path: Path) -> bool:
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    input("Press Enter to start recording...")
    print("Recording... Press Enter to stop.")

    recorded_chunks = []
    with sd.InputStream(
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    ):
        input()
        print("Stopping...")

        while not audio_queue.empty():
            recorded_chunks.append(audio_queue.get())

    if not recorded_chunks:
        print("No audio recorded.")
        return False

    audio = np.concatenate(recorded_chunks, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLERATE)
    print(f"Saved to {output_path}")
    return True


def preview_in_blender(ik_bvh: Path, tts_audio: str | Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            BLENDER,
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


def run_turn(models: DemoModels, momask: MomaskRuntime) -> subprocess.Popen | None:
    if not record_audio(MIC_WAV):
        return None

    result = models.run_audio_turn(MIC_WAV, TTS_WAV)
    motion_prompt = result["motion_instruction"]
    print("User text:", result["user_text"])
    print("Talk:", result["talk"])
    print("Motion:", motion_prompt)

    motion = momask.generate(motion_prompt)
    print(f"Opening Blender preview: {motion['ik_bvh']}")
    return preview_in_blender(Path(motion["ik_bvh"]), result["tts_output"])


def main() -> None:
    print("Loading demo models...")
    models = DemoModels(tts_config=TTSConfig(engine="piper"))
    momask = MomaskRuntime(gpu_id=-1, ext="demo_run", write_preview_mp4=False)
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
        blender_proc = run_turn(models, momask)


if __name__ == "__main__":
    main()
