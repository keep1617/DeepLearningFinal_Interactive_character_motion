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


SAMPLERATE = 16000
CHANNELS = 1
MIC_WAV = ROOT / "scripts" / "mic_input.wav"
TTS_WAV = ROOT / "outputs" / "talk.wav"
FRAMES_PREFIX = ROOT / "outputs" / "demo_frames" / "frame_"
FRAME_PATTERN = ROOT / "outputs" / "demo_frames" / "frame_%04d.png"
VIDEO_OUTPUT = ROOT / "outputs" / "demo" / "character_with_voice.mp4"


def record_audio(output_path: Path) -> bool:
    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
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


def clear_old_frames() -> None:
    FRAMES_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    for frame in FRAMES_PREFIX.parent.glob("frame_*.png"):
        frame.unlink()


def render_blender(ik_bvh: Path) -> None:
    clear_old_frames()
    subprocess.run(
        [
            "/home/adfa5456/Downloads/blender-5.0.1-linux-x64/blender",
            "-b",
            "--factory-startup",
            "--python",
            str(ROOT / "scripts" / "blender_keemap_retarget_render.py"),
            "--",
            "--addon-dir",
            str(ROOT / "3rdParty" / "KeeMapAnimRetarget"),
            "--bvh",
            str(ik_bvh),
            "--fbx",
            str(ROOT / "asset" / "Ch15_nonPBR.fbx"),
            "--mapping",
            str(ROOT / "3rdParty" / "momask-codes" / "assets" / "mapping.json"),
            "--output",
            str(FRAMES_PREFIX),
        ],
        check=True,
    )


def mux_video_with_audio(tts_output_path: str | Path) -> Path:
    VIDEO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "20",
            "-start_number",
            "0",
            "-i",
            str(FRAME_PATTERN),
            "-i",
            str(tts_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(VIDEO_OUTPUT),
        ],
        check=True,
    )
    return VIDEO_OUTPUT


def open_video(video_path: Path) -> None:
    try:
        subprocess.Popen(
            ["xdg-open", str(video_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"Could not open video automatically: {exc}")


def run_turn(models: DemoModels, momask: MomaskRuntime) -> None:
    if not record_audio(MIC_WAV):
        return

    result = models.run_audio_turn(MIC_WAV, TTS_WAV)
    motion_prompt = result["motion_instruction"]
    print("User text:", result["user_text"])
    print("Talk:", result["talk"])
    print("Motion:", motion_prompt)

    motion = momask.generate(motion_prompt)
    render_blender(Path(motion["ik_bvh"]))
    video_path = mux_video_with_audio(result["tts_output"])
    print(f"Video saved to {video_path}")
    open_video(video_path)


def main() -> None:
    print("Loading demo models...")
    models = DemoModels(tts_config=TTSConfig(engine="piper"))
    momask = MomaskRuntime(gpu_id=-1, ext="demo_run", write_preview_mp4=False)
    print("Ready.")

    while True:
        command = input("Press r to record/render again, or q to quit: ").strip().lower()
        if command == "q":
            break
        if command != "r":
            continue
        run_turn(models, momask)


if __name__ == "__main__":
    main()
