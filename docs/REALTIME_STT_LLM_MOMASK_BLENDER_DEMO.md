# Realtime STT/TTS/LLM/MoMask/Blender 데모 설계

## 목표

사용자가 마이크로 말하면 STT가 텍스트로 바꾸고, LLM이 soldier persona의 `talk`와 `motion_instruction`을 만든다. `talk`는 TTS로 말하고, `motion_instruction`은 MoMask로 사람 모션을 생성한다. 생성된 MoMask BVH를 Blender에서 Mixamo 캐릭터에 KeeMapRig로 retarget하고, 캐릭터 animation을 렌더하거나 preview한다.

목표 파이프라인:

```text
microphone
  -> faster-whisper STT
  -> Qwen LLM
       -> talk
       -> motion_instruction
  -> MeloTTS English TTS for talk
  -> MoMask text-to-motion for motion_instruction
  -> MoMask _ik.bvh
  -> Blender + KeeMapRig + Mixamo FBX
  -> character animation mp4 / preview
```

## 지금까지 만든 구성

### STT

파일:

```text
examples/stt_example.py
```

역할:

- faster-whisper로 wav 파일을 transcript로 변환한다.
- 현재 기본 입력은 `examples/out/tts_en.wav`지만, 실시간 데모에서는 microphone 녹음 wav를 넘기면 된다.

실행 예:

```bash
/home/adfa5456/anaconda3/envs/momask5080/bin/python examples/stt_example.py --audio input.wav --text-only
```

### LLM

파일:

```text
examples/llm_example.py
```

역할:

- Qwen 로컬 모델로 JSON을 만든다.
- 출력 키:
  - `talk`: soldier persona로 말할 문장
  - `motion_instruction`: MoMask에 넣을 `a person is ...` motion prompt

실행 예:

```bash
/home/adfa5456/anaconda3/envs/momask5080/bin/python examples/llm_example.py \
  --prompt "Can you report your status?"
```

### TTS

파일:

```text
examples/tts_example.py
```

역할:

- MeloTTS English 모델로 `talk`를 wav로 만든다.
- 한국어 TTS 대신 영어 TTS로 전환해둔 상태다.

실행 예:

```bash
/home/adfa5456/anaconda3/envs/momask5080/bin/python examples/tts_example.py \
  --text "Copy that. Standing by for orders."
```

### MoMask

기존 생성 스크립트:

```text
3rdParty/momask-codes/gen_t2m.py
```

역할:

- `motion_instruction`에서 22-joint human motion을 만든다.
- 출력 위치는 `3rdParty/momask-codes/generation/<ext>/` 아래다.
- 중요한 결과물:
  - `joints/.../*.npy`
  - `animations/.../*.bvh`
  - `animations/.../*_ik.bvh`
  - skeleton mp4

실행 예:

```bash
cd 3rdParty/momask-codes
/home/adfa5456/anaconda3/envs/momask5080/bin/python gen_t2m.py \
  --gpu_id -1 \
  --ext demo_run \
  --text_prompt "a person is waving both hands."
```

MoMask README 기준으로 `_ik.bvh`가 foot IK 보정이 들어간 버전이라 Blender character retarget에는 보통 `_ik.bvh`를 우선 사용한다.

### Blender Character Retarget

Blender 실행 파일:

```text
/home/adfa5456/Downloads/blender-5.0.1-linux-x64/blender
```

Mixamo character:

```text
asset/Ch15_nonPBR.fbx
```

KeeMapRig addon:

```text
3rdParty/KeeMapAnimRetarget
```

MoMask bone mapping:

```text
3rdParty/momask-codes/assets/mapping.json
3rdParty/momask-codes/assets/mapping6.json
```

현재 FBX의 bone prefix는 `mixamorig:`라서 `mapping.json`이 맞다.

새로 만든 Blender scripts:

```text
scripts/blender_inspect_fbx.py
scripts/blender_retarget_momask_to_mixamo.py
scripts/blender_keemap_retarget_render.py
```

가장 중요한 것은 `scripts/blender_keemap_retarget_render.py`다. 이 스크립트는 Blender background mode에서 KeeMapRig addon을 등록하고, BVH와 FBX를 import한 뒤, mapping file을 읽어서 animation transfer를 실행하고 PNG frame sequence를 렌더한다.

실행 예:

```bash
/home/adfa5456/Downloads/blender-5.0.1-linux-x64/blender -b --factory-startup \
  --python scripts/blender_keemap_retarget_render.py -- \
  --bvh 3rdParty/momask-codes/generation/ue_stream/animations/0/sample0_repeat0_len196_ik.bvh \
  --fbx asset/Ch15_nonPBR.fbx \
  --mapping 3rdParty/momask-codes/assets/mapping.json \
  --output outputs/momask_keemap_full_frames/frame_
```

Blender 5.0.1 빌드에서 FFmpeg file output enum이 없어서 Blender 안에서 바로 mp4를 만들지 않고 PNG frame을 만든 뒤 ffmpeg로 묶는다.

```bash
ffmpeg -y -framerate 20 -start_number 0 \
  -i outputs/momask_keemap_full_frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p outputs/momask_keemap_full.mp4
```

## Blender 쪽 개발 순서

### 1. Blender 실행 파일 찾기

처음에는 `which blender`로 PATH를 확인했지만 잡히지 않았다. 이후 Downloads 안에서 Blender binary를 찾았다.

```text
/home/adfa5456/Downloads/blender-5.0.1-linux-x64/blender
```

### 2. Mixamo FBX 확인

workspace 안에서 FBX를 찾았다.

```text
asset/Ch15_nonPBR.fbx
```

### 3. FBX bone 구조 inspection

`scripts/blender_inspect_fbx.py`를 만들어 Blender background mode에서 FBX를 import하고 armature/bone 이름을 출력했다.

확인 결과:

```text
Armature: Armature
Mesh: Ch15
Bone prefix: mixamorig:
Example bones:
mixamorig:Hips
mixamorig:Spine
mixamorig:LeftArm
mixamorig:RightArm
```

이 때문에 `mapping6.json`이 아니라 `mapping.json`을 사용했다.

### 4. fallback retarget script 작성

처음에는 KeeMapRig을 자동으로 부르는 방법을 보기 전에 `scripts/blender_retarget_momask_to_mixamo.py`를 만들었다. 이 스크립트는 BVH source pose bone의 local transform을 Mixamo target bone에 단순 복사한다.

이 방식은 캐릭터가 움직이는지 빠르게 확인하는 용도였다. 하지만 KeeMap의 보정값을 충분히 쓰지 못하므로 최종 방식으로는 부족하다.

결과:

```text
outputs/momask_character_full.mp4
```

### 5. KeeMapRig addon 찾기

사용자가 `3rdParty`에 받아둔 KeeMapRig을 확인했다.

```text
3rdParty/KeeMapAnimRetarget
```

addon 내부에서 operator 이름을 확인했다.

```text
wm.keemap_read_file
wm.perform_animation_transfer
```

### 6. KeeMapRig 자동화 script 작성

`scripts/blender_keemap_retarget_render.py`를 추가했다.

이 스크립트가 하는 일:

1. Blender scene 초기화
2. `3rdParty/KeeMapAnimRetarget` addon 등록
3. MoMask `_ik.bvh` import
4. Mixamo `.fbx` import
5. `mapping.json` 읽기
6. source rig, destination rig 이름을 실제 import 이름으로 설정
7. `wm.perform_animation_transfer` 실행
8. source BVH skeleton 숨김
9. camera/light/ground 생성
10. Workbench renderer로 PNG frame sequence 출력

결과:

```text
outputs/momask_keemap_preview.mp4
outputs/momask_keemap_full.mp4
```

## 실시간 데모 구현 방식

완전한 의미의 실시간은 두 레벨로 나눠야 한다.

### Level 1: 반실시간 데모

이 방식이 지금 가장 현실적이다.

```text
user speech
  -> STT
  -> LLM JSON
  -> TTS starts immediately
  -> MoMask generation runs in background
  -> Blender retarget/render runs in background
  -> character mp4 appears after generation
```

장점:

- 지금 만든 파일들을 거의 그대로 연결할 수 있다.
- 음성 대답은 빠르게 나온다.
- motion은 몇 초 뒤에 렌더 결과로 보여줄 수 있다.

단점:

- MoMask와 Blender 렌더가 끝나야 캐릭터 animation이 나온다.
- “프레임 단위 실시간 avatar”는 아니다.

### Level 2: 캐시 기반 실시간 데모

자주 나오는 motion prompt를 미리 만들어둔다.

예:

```text
a person is standing at attention
a person is saluting
a person is waving both hands
a person is pointing forward
a person is walking forward
a person is nodding
a person is placing hands behind the back
```

미리 할 일:

1. prompt별 MoMask 생성
2. prompt별 `_ik.bvh` 저장
3. prompt별 Blender KeeMap 렌더 mp4 저장

실시간 때는 LLM의 `motion_instruction`을 nearest cached prompt로 매칭해서 즉시 mp4를 재생한다.

장점:

- 데모 반응성이 매우 좋다.
- 품질 검수된 motion만 보여줄 수 있다.
- MoMask가 느려도 문제가 적다.

단점:

- 완전한 open-ended generation은 아니다.
- prompt coverage를 미리 준비해야 한다.

### Level 3: 진짜 streaming character

이건 별도 프로젝트에 가깝다.

필요한 것:

- Blender에서 매번 mp4 렌더하지 않고 viewport/engine 안에서 armature pose를 실시간 update
- 또는 Unity/Unreal에서 BVH/animation clip을 runtime load
- MoMask generation latency를 줄이거나 cache/retrieval과 섞기

현재 프로젝트에서는 Level 1 또는 Level 2가 맞다.

## 추천 데모 구조

가장 안정적인 구조:

```text
main_demo.py
  1. record microphone to wav
  2. run STT
  3. run LLM
  4. parse JSON
  5. start TTS playback
  6. choose motion mode
       - cached animation exists: play immediately
       - cache miss: run MoMask + Blender render
  7. show resulting character mp4
```

### Process 분리

하나의 Python process에서 전부 import하면 dependency 충돌이 날 수 있다. 지금 환경은 MoMask, MeloTTS, transformers, Blender가 모두 무겁기 때문에 subprocess로 나누는 것이 안전하다.

권장:

```text
orchestrator process
  -> subprocess: STT
  -> subprocess: LLM
  -> subprocess: TTS
  -> subprocess: MoMask generation
  -> subprocess: Blender render
  -> subprocess: ffmpeg encode
```

### 파일 기반 인터페이스

처음에는 HTTP server보다 파일 기반이 단순하다.

```text
outputs/demo/latest_input.wav
outputs/demo/latest_stt.txt
outputs/demo/latest_llm.json
outputs/demo/latest_talk.wav
outputs/demo/latest_motion_prompt.txt
outputs/demo/latest_momask/
outputs/demo/latest_frames/
outputs/demo/latest_character.mp4
```

## 다음 구현 순서

### 1. 단일 command demo script 만들기

목표 파일:

```text
scripts/run_demo_once.py
```

기능:

1. `--audio input.wav` 또는 `--text "..."` 입력
2. LLM 실행
3. JSON 저장
4. TTS wav 생성
5. MoMask prompt file 생성
6. MoMask 실행
7. 생성된 `_ik.bvh` 자동 탐색
8. Blender KeeMap 렌더
9. ffmpeg mp4 생성

### 2. cache 디렉터리 만들기

목표:

```text
outputs/demo_cache/
```

key는 motion prompt를 slug/hash로 만든다.

```text
outputs/demo_cache/a_person_is_waving_both_hands/
  motion_prompt.txt
  momask/
  frames/
  character.mp4
```

### 3. 자주 쓰는 soldier motion preset 생성

추천 preset:

```text
a person is standing at attention
a person is saluting with one hand
a person is waving both hands
a person is pointing forward
a person is stepping forward
a person is stepping back
a person is nodding
a person is turning around
a person is placing hands behind the back
a person is looking around
```

### 4. 간단 UI

처음은 Gradio 또는 Streamlit보다 단순 CLI가 안정적이다.

1차:

```bash
python scripts/run_demo_once.py --text "What is your status?"
```

2차:

```bash
python scripts/run_demo_loop.py
```

3차:

웹 UI:

```text
record button
transcript
talk text
motion_instruction
audio player
video player
```

## 주의점

- MoMask는 motion generation이므로 매 요청마다 몇 초 이상 걸릴 수 있다.
- Blender full render도 196프레임 기준 약 1분 안팎이 걸릴 수 있다.
- 따라서 발표/demo는 cache 기반으로 준비해두고, cache miss일 때만 background generation을 돌리는 구조가 좋다.
- KeeMapRig은 Blender 3.x용 addon이지만 Blender 5.0.1 background mode에서 이번 테스트는 성공했다.
- Blender 5.0.1의 현재 빌드에서는 mp4 직접 렌더가 안 되어 PNG frame sequence 후 ffmpeg 인코딩을 사용한다.
