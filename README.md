# DeepLearning Final: Voice-to-Motion Character Demo

마이크 음성을 입력받아 STT로 텍스트를 만들고, 로컬 LLM이 캐릭터 대사와 motion prompt를 생성한 뒤, TTS 음성과 MoMask motion을 Blender 캐릭터에 적용하는 인터랙티브 데모입니다.

## Pipeline

```text
microphone
  -> faster-whisper STT
  -> Qwen LLM
       -> talk
       -> motion_instruction
  -> Piper TTS
       -> talk.wav
  -> MoMask text-to-motion
       -> joints .npy
       -> _ik.bvh
  -> Blender + KeeMapRig
       -> Mixamo FBX retarget preview
```

## Repository Layout

```text
scripts/model_load.py                    # STT/LLM/TTS runtime loader
src/momask_runtime.py                    # reusable MoMask runtime
scripts/run_demo_once_blender.py         # main Blender preview demo
scripts/blender_keemap_retarget_preview.py
3rdParty/momask-codes/                   # MoMask code/checkpoints
3rdParty/models/                         # local STT/LLM/TTS models
3rdParty/KeeMapAnimRetarget/             # Blender retarget addon
asset/Ch15_nonPBR.fbx                    # Mixamo character
```

## 1. System Dependencies

Install basic audio/video dependencies:

```bash
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev libsndfile1 curl git unzip
```

Download Blender with the helper script:

```bash
bash scripts/download_blender.sh
```

Default install path:

```text
/home/adfa5456/Downloads/blender-5.0.1-linux-x64/blender
```

To change the version or install location:

```bash
BLENDER_VERSION=5.0.1 BLENDER_INSTALL_ROOT="$HOME/Downloads" bash scripts/download_blender.sh
```

The path is currently hardcoded in:

```text
scripts/run_demo_once_blender.py
```

## 2. Conda Environment

Create and activate the demo environment:

```bash
conda env create -f environment.yml
conda activate momask5080
```

If you prefer manual installation:

```bash
conda create -n momask5080 python=3.10 -y
conda activate momask5080
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

If MoMask reports a missing package, also install the bundled MoMask requirements:

```bash
pip install -r 3rdParty/momask-codes/requirements_5080.txt
```

## 3. Download Models

Create model folders:

```bash
mkdir -p 3rdParty/models
```

### STT: faster-whisper-small

```bash
huggingface-cli download Systran/faster-whisper-small \
  --local-dir 3rdParty/models/faster-whisper-small \
  --local-dir-use-symlinks False
```

Expected path:

```text
3rdParty/models/faster-whisper-small/model.bin
```

### LLM: Qwen

Recommended:

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir 3rdParty/models/Qwen3-4B-Instruct-2507 \
  --local-dir-use-symlinks False
```

Smaller fallback:

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir 3rdParty/models/Qwen3-0.6B \
  --local-dir-use-symlinks False
```

`scripts/model_load.py` automatically uses `Qwen3-4B-Instruct-2507` if present, otherwise `Qwen3-0.6B`.

### TTS: Piper Male Voice

```bash
mkdir -p 3rdParty/models/Piper

curl -L -o 3rdParty/models/Piper/en_US-ryan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx

curl -L -o 3rdParty/models/Piper/en_US-ryan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
```

Expected paths:

```text
3rdParty/models/Piper/en_US-ryan-medium.onnx
3rdParty/models/Piper/en_US-ryan-medium.onnx.json
```

### MoMask Checkpoints

From the MoMask directory:

```bash
cd 3rdParty/momask-codes
bash prepare/download_models.sh
cd ../..
```

Expected paths include:

```text
3rdParty/momask-codes/checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar
3rdParty/momask-codes/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar
3rdParty/momask-codes/checkpoints/t2m/tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw/model/net_best_fid.tar
3rdParty/momask-codes/checkpoints/t2m/length_estimator/model/finest.tar
```

## 4. Verify TTS

```bash
python - <<'PY'
from pathlib import Path
from scripts.model_load import TTSRuntime, TTSConfig

out = TTSRuntime(TTSConfig(engine="piper")).synthesize(
    "Copy that. Standing by for orders.",
    Path("outputs/piper_smoke.wav"),
)
print(out)
PY
```

Expected output:

```text
outputs/piper_smoke.wav
```

## 5. Run Blender Preview Demo

Activate the environment:

```bash
conda activate momask5080
```

Run:

```bash
python scripts/run_demo_once_blender.py
```

Controls:

```text
r  -> record one utterance and open Blender preview
q  -> quit
```

When prompted:

```text
Press Enter to start recording...
Recording... Press Enter to stop.
```

Press Enter once to start recording, speak, then press Enter again to stop.

The script will:

1. Save microphone audio to `scripts/mic_input.wav`.
2. Transcribe it with faster-whisper.
3. Generate `talk` and `motion_instruction` with Qwen.
4. Synthesize `talk` with Piper male TTS to `outputs/talk.wav`.
5. Generate motion with MoMask.
6. Open Blender and retarget `_ik.bvh` to `asset/Ch15_nonPBR.fbx`.
7. Preview the animation in a procedural trench scene.

## Notes

- `run_demo_once_blender.py` is the main demo entry point.
- The demo previews directly in Blender UI and does not render mp4.
- MoMask generation can be slow on CPU.
- The Blender path is machine-specific; edit `BLENDER` in the scripts if Blender is installed elsewhere.
- KeeMap retarget uses `3rdParty/momask-codes/assets/mapping.json`, which matches the current Mixamo skeleton prefix `mixamorig:`.
