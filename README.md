# DeepLearning Final: Interactive Character Motion

마이크로 녹음한 말을 STT로 텍스트화하고, 로컬 LLM이 캐릭터 대사와 motion prompt를 만든 뒤, TTS 음성과 MoMask motion을 Blender 캐릭터에 적용하는 인터랙티브 데모입니다.

## Pipeline

```text
microphone
  -> faster-whisper STT
  -> Qwen LLM
       -> talk
       -> motion_instruction
  -> Piper TTS
       -> outputs/talk.wav
  -> MoMask text-to-motion
       -> joints .npy
       -> _ik.bvh
  -> Blender + KeeMap
       -> Mixamo FBX retarget preview
```

## Repository Layout

```text
scripts/model_load.py                    # STT / LLM / TTS loader
src/momask_runtime.py                    # reusable MoMask runtime
scripts/run_demo_once_blender.py         # main demo entry point
scripts/blender_keemap_retarget_preview.py
scripts/download_blender.sh              # Blender downloader
scripts/download_hf_models.py            # faster-whisper / Qwen downloader
requirements.txt                         # pip dependencies
environment.yml                          # conda environment
project_explanation.md                   # project/model/pipeline summary

3rdParty/KeeMapAnimRetarget/             # Blender retarget addon
3rdParty/momask-codes/                   # cloned during setup, not committed
3rdParty/models/                         # downloaded local models, not committed
asset/Ch15_nonPBR.fbx                    # Mixamo character, not committed by default
```

## 1. System Dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev libsndfile1 curl git unzip
```

Download Blender:

```bash
bash scripts/download_blender.sh
```

The script prints the Blender executable path on the last line. You will pass that path to the demo with `--blender`.

To use a different Blender version or install directory:

```bash
BLENDER_VERSION=5.0.1 BLENDER_INSTALL_ROOT="$HOME/Downloads" bash scripts/download_blender.sh
```

## 2. Conda Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate momask5080
```

This environment pins `numpy==1.23.5`. Do not upgrade NumPy: the original MoMask code uses `np.float`, which was removed in NumPy 1.24.

For a clean reinstall:

```bash
conda env remove -n momask5080
conda env create -f environment.yml
conda activate momask5080
```

If you already created the environment and see `AttributeError: module 'numpy' has no attribute 'float'`, fix it with:

```bash
conda activate momask5080
conda install -c conda-forge numpy=1.23.5 -y
```

Verify:

```bash
python - <<'PY'
import numpy as np
print(np.__version__)
print(hasattr(np, "float"))
PY
```

Expected output:

```text
1.23.5
True
```

Manual fallback:

```bash
conda create -n momask5080 python=3.10 -y
conda activate momask5080
conda install -c conda-forge numpy=1.23.5 ffmpeg portaudio libsndfile -y
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

If MoMask reports a missing package after setup:

```bash
pip install -r 3rdParty/momask-codes/requirements_5080.txt
conda install -c conda-forge numpy=1.23.5 -y
```

Run the NumPy verification command above again after installing extra MoMask packages.

## 3. Download Models And Assets

Create the local model directory:

```bash
mkdir -p 3rdParty/models
```

The old `huggingface-cli` command is deprecated. Use the included Python downloader or the current `hf download` command shown below.

To download faster-whisper and the recommended Qwen model together:

```bash
python scripts/download_hf_models.py --qwen 4b
```

For a smaller Qwen model:

```bash
python scripts/download_hf_models.py --qwen 0.6b
```

### MoMask Source

Clone MoMask into the path expected by the runtime:

```bash
mkdir -p 3rdParty
git clone https://github.com/EricGuo5513/momask-codes.git 3rdParty/momask-codes
```

`3rdParty/momask-codes` is intentionally not committed to this repository. Clone it during setup.

### MoMask Checkpoints

MoMask checkpoint download script is not reliable for this project setup. Download the checkpoint zip manually from Google Drive:

[MoMask humanml3d_models.zip Drive folder](https://drive.google.com/drive/folders/1sHajltuE2xgHh91H9pFpMAYAkHaX9o57)

Download:

```text
humanml3d_models.zip
```

Then place the zip file inside `3rdParty/momask-codes` and extract it there:

```bash
cd 3rdParty/momask-codes
mkdir -p checkpoints/t2m
cp ~/Downloads/humanml3d_models.zip checkpoints/t2m/
cd checkpoints/t2m/
unzip -o humanml3d_models.zip
cd ../..
```

Expected checkpoint files include:

```text
3rdParty/momask-codes/checkpoints/t2m/t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar
3rdParty/momask-codes/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar
3rdParty/momask-codes/checkpoints/t2m/tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw/model/net_best_fid.tar
3rdParty/momask-codes/checkpoints/t2m/length_estimator/model/finest.tar
```

### If you did step 3 ```python scripts/download_hf_models.py --qwen 4b``` you don't need to do under processes. You should do Piper!
### STT: faster-whisper-small

Preferred:

```bash
python scripts/download_hf_models.py --skip-qwen
```

Equivalent current Hugging Face CLI command:

```bash
hf download Systran/faster-whisper-small \
  --local-dir 3rdParty/models/faster-whisper-small
```

Expected path:

```text
3rdParty/models/faster-whisper-small/model.bin
```

### LLM: Qwen

Recommended model:

```bash
python scripts/download_hf_models.py --skip-whisper --qwen 4b
```

Smaller fallback model:

```bash
python scripts/download_hf_models.py --skip-whisper --qwen 0.6b
```

Equivalent current Hugging Face CLI commands:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir 3rdParty/models/Qwen3-4B-Instruct-2507

hf download Qwen/Qwen3-0.6B \
  --local-dir 3rdParty/models/Qwen3-0.6B
```

`scripts/model_load.py` uses `Qwen3-4B-Instruct-2507` if it exists, otherwise `Qwen3-0.6B`.

### TTS: Piper Male Voice

```bash

cd ../..
mkdir 3rdParty/models/Piper

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

### KeeMap Blender Addon

The demo expects KeeMap at:

```text
3rdParty/KeeMapAnimRetarget
```

You do not need to install the addon manually from Blender Preferences. `scripts/blender_keemap_retarget_preview.py` registers the local addon automatically when the demo starts.

### Character FBX

The demo expects a Mixamo-style character at:

```text
asset/Ch15_nonPBR.fbx
```

This FBX is not tracked by default because it is larger than GitHub's normal 100 MB file limit. Copy it into `asset/` manually, or use Git LFS if you want to commit it.

If the repository uses Git LFS for the FBX file, install Git LFS and pull the real asset after cloning:

```bash
sudo apt install -y git-lfs
git lfs install
git lfs pull
```

After pulling, `asset/Ch15_nonPBR.fbx` should be a large binary file, not a small text pointer file.

## 4. Verify Setup

```bash
test -f 3rdParty/models/faster-whisper-small/model.bin
test -f 3rdParty/models/Piper/en_US-ryan-medium.onnx
test -f 3rdParty/models/Piper/en_US-ryan-medium.onnx.json
test -f 3rdParty/momask-codes/checkpoints/t2m/length_estimator/model/finest.tar
test -d 3rdParty/KeeMapAnimRetarget
test -f asset/Ch15_nonPBR.fbx
```

Verify Piper TTS:

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

## 5. Run The Demo

Activate the environment:

```bash
conda activate momask5080
```

Use the Blender path printed by the downloader:

```bash
BLENDER_BIN="$(bash scripts/download_blender.sh | tail -n 1)"
python scripts/run_demo_once_blender.py --blender "$BLENDER_BIN"
```

Or pass your own Blender executable directly:

```bash
python scripts/run_demo_once_blender.py --blender /path/to/blender
```

Example:

```bash
python scripts/run_demo_once_blender.py --blender ~/Downloads/blender-5.0.1-linux-x64/blender
```

You can also set `BLENDER` once and omit the argument:

```bash
export BLENDER=/path/to/blender
python scripts/run_demo_once_blender.py
```

Audio device options:

```bash
python scripts/run_demo_once_blender.py --list-audio-devices
```

If the default microphone fails, choose an input device id from the list:

```bash
python scripts/run_demo_once_blender.py --blender ~/Downloads/blender-5.0.1-linux-x64/blender --input-device 2
```

If that PC needs a specific microphone sample rate:

```bash
python scripts/run_demo_once_blender.py --blender ~/Downloads/blender-5.0.1-linux-x64/blender --input-device 2 --samplerate 48000
```

Controls:

```text
r  -> record one utterance and open Blender preview
q  -> quit
```

Recording flow:

```text
Press Enter to start recording...
Recording at <device sample rate> Hz... Press Enter to stop.
```

Press Enter once, speak, then press Enter again to stop. The demo then transcribes your voice, generates the character response, creates motion with MoMask, and opens a Blender preview in the trench scene.

## Outputs

```text
scripts/mic_input.wav
outputs/talk.wav
3rdParty/momask-codes/generation/demo_run/
```

The current main demo previews directly in Blender UI. It does not render an mp4 by default.

## Troubleshooting

### Invalid microphone sample rate

If another PC shows:

```text
sounddevice.PortAudioError: Error opening InputStream: Invalid sample rate
```

the microphone probably does not support the old fixed `16000 Hz` input rate. `run_demo_once_blender.py` now uses the input device's default sample rate automatically, but you can still select the device and sample rate manually:

```bash
python scripts/run_demo_once_blender.py --list-audio-devices
python scripts/run_demo_once_blender.py --blender ~/Downloads/blender-5.0.1-linux-x64/blender --input-device 2 --samplerate 48000
```

## Notes

- `scripts/run_demo_once_blender.py` is the only main demo entry point.
- Blender path is configurable. Pass it with `--blender`, set `BLENDER`, or make `blender` available on PATH.
- MoMask generation can be slow on CPU. Use `--gpu-id 0` if your CUDA setup is ready.
- KeeMap uses `3rdParty/momask-codes/assets/mapping.json` for the current Mixamo skeleton mapping.
- Downloaded models, checkpoints, generated outputs, and large media files are ignored by git.

## GitHub Upload Notes

Before pushing, make sure generated outputs and downloaded models are not staged:

```bash
git status --short
git add .gitignore README.md requirements.txt environment.yml project_explanation.md scripts src 3rdParty/KeeMapAnimRetarget
git status --short
```

`3rdParty/momask-codes` should not be committed. If it was accidentally added before, remove it from the git index:

```bash
git rm -r --cached 3rdParty/momask-codes
```

Users clone MoMask during setup with the command above.

If `git add 3rdParty/KeeMapAnimRetarget` warns about an embedded git repository, either commit it as a proper submodule or remove the nested `.git` directory before adding its source code.

If you want to include the FBX character file, use Git LFS:

```bash
git lfs install
git lfs track "asset/*.fbx"
git add .gitattributes asset/Ch15_nonPBR.fbx
```

Otherwise, keep `asset/*.fbx` ignored and copy the character file manually after cloning.
