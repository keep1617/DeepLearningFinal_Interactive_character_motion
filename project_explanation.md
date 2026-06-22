# Project Explanation

## 프로젝트 개요

이 프로젝트는 사용자의 마이크 음성을 입력으로 받아 텍스트로 변환하고, 로컬 LLM이 캐릭터의 대사와 동작 지시문을 생성한 뒤, 대사는 TTS로 음성화하고 동작 지시문은 MoMask text-to-motion 모델로 3D 모션으로 변환하여 Blender 캐릭터에 retarget하는 실시간 인터랙티브 데모이다.

## 사용 모델 및 특징

| 단계 | 모델/도구 | 특징 및 방법론 |
| --- | --- | --- |
| STT | faster-whisper-small | Whisper 계열 음성 인식 모델을 CTranslate2 기반으로 경량화하여 로컬 CPU/GPU에서 빠르게 wav 음성을 텍스트로 변환한다. |
| LLM | Qwen3-4B-Instruct-2507 또는 Qwen3-0.6B | 입력 텍스트를 바탕으로 JSON 형식의 `talk`와 `motion_instruction`을 생성하는 instruction-following causal language model이다. |
| TTS | Piper `en_US-ryan-medium` | ONNX 기반 로컬 neural TTS로, 생성된 `talk` 문장을 남성 영어 음성 wav로 빠르게 합성한다. |
| TTS 대체 | MeloTTS English/Korean | 다국어 multi-speaker TTS 모델로, 기존에는 English speaker를 사용해 대사 음성을 생성했다. |
| Text-to-Motion | MoMask | VQ-VAE, masked transformer, residual transformer, length estimator를 사용해 자연어 motion prompt를 22-joint human motion으로 생성한다. |
| Retarget/Visualization | Blender + KeeMapRig | MoMask BVH 모션을 Mixamo FBX 캐릭터 rig에 bone mapping 기반으로 retarget하고 viewport preview 또는 frame rendering을 수행한다. |

## 주요 파일 구성

| 파일/폴더 | 역할 |
| --- | --- |
| `scripts/model_load.py` | STT, LLM, TTS 모델을 한 번 로드하고 `DemoModels` 런타임으로 재사용한다. |
| `src/momask_runtime.py` | MoMask 모델을 한 번 로드하고 여러 motion prompt에 대해 `.npy`, `.bvh`, `_ik.bvh`를 생성한다. |
| `scripts/run_demo_once.py` | 마이크 녹음부터 STT, LLM, TTS, MoMask, Blender render, mp4 생성까지 한 번 또는 반복 실행하는 데모 스크립트이다. |
| `scripts/run_demo_once_blender.py` | mp4 렌더링 대신 Blender UI에서 retarget 결과를 바로 preview하는 데모 스크립트이다. |
| `scripts/blender_keemap_retarget_render.py` | Blender background mode에서 BVH와 FBX를 불러와 KeeMap retarget 후 PNG frame sequence를 렌더링한다. |
| `scripts/blender_keemap_retarget_preview.py` | Blender UI에서 KeeMap retarget 결과를 즉시 재생하고 TTS 오디오를 animation loop에 맞춰 재생한다. |
| `src/stream_motion.py` | MoMask `.npy` joint sequence를 UDP frame 단위로 전송하는 실험용 streaming 모듈이다. |
| `3rdParty/momask-codes/` | MoMask 원본 코드, checkpoint, generation script, BVH 변환 로직이 들어 있다. |
| `3rdParty/models/` | faster-whisper, Qwen, Piper, MeloTTS 등 로컬 모델 파일이 저장되어 있다. |
| `asset/Ch15_nonPBR.fbx` | Blender retarget 대상 Mixamo 캐릭터 FBX asset이다. |

## 전체 파이프라인

```text
Microphone input
  -> record wav
  -> faster-whisper STT
  -> user_text
  -> Qwen LLM
       -> talk
       -> motion_instruction
  -> Piper TTS
       -> talk.wav
  -> MoMask text-to-motion
       -> joints .npy
       -> BVH
       -> foot-IK corrected _ik.bvh
  -> Blender
       -> import MoMask _ik.bvh
       -> import Mixamo FBX character
       -> KeeMapRig bone mapping retarget
       -> viewport preview or PNG frame render
  -> optional ffmpeg mux
       -> character animation mp4 with TTS audio
```

## 동작 흐름

1. 사용자가 `r`을 누르고 마이크로 말하면 `sounddevice`가 음성을 녹음해 `mic_input.wav`로 저장한다.
2. `faster-whisper-small`이 녹음된 wav를 영어 텍스트로 변환한다.
3. Qwen LLM이 텍스트 입력을 받아 soldier persona의 대사인 `talk`와 MoMask용 동작 문장인 `motion_instruction`을 JSON으로 생성한다.
4. Piper TTS가 `talk`를 남성 음성 wav로 합성한다.
5. `MomaskRuntime`이 `motion_instruction`을 입력으로 받아 22-joint motion을 생성하고 `_ik.bvh`를 만든다.
6. Blender가 `_ik.bvh`와 Mixamo 캐릭터 FBX를 불러온 뒤 KeeMapRig mapping으로 캐릭터에 animation을 transfer한다.
7. preview 모드에서는 Blender viewport에서 바로 animation을 재생하고, render 모드에서는 PNG frame sequence를 만든 뒤 ffmpeg로 TTS audio와 합쳐 mp4를 생성한다.

## 핵심 아이디어

이 프로젝트의 핵심은 음성 대화형 캐릭터 제어를 `speech-to-text -> language reasoning -> text-to-motion -> character retargeting`으로 분리하고, 각 모델을 로컬 런타임에 미리 로드하여 한 번의 사용자 발화가 대사와 몸동작을 동시에 생성하도록 구성한 점이다.

