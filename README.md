# sam3d_asset_extractor

RGB(-D) 이미지에서 객체 단위 메시를 추출해 시뮬레이터 자산으로 저장하기 위한 전처리 파이프라인입니다. VLA 파인튜닝 데이터 증강 연구의 프리프로세싱으로 사용합니다.

## 파이프라인

1. **SAM2 마스킹** — 자동(`--sam2-mode auto`) 또는 포인트 기반 대화형(`--sam2-mode manual`)
2. **SAM3D 추론** — 각 마스크에 대해 가우시안 PLY + 메시(glb/ply/obj) + pose(json) 생성.
   Depth + 카메라 intrinsics 기반 pointmap을 **항상** SAM3D 입력으로 사용
   (`--sam3d-input full` 기본: 전체 pointmap, `--sam3d-input cropped`: 마스크 영역만)
3. **(선택) 메시 Decimate** — 생성된 메시의 face 수 축소

> **필수 입력**: RGB(`--image`) + depth(`--depth-image`) + intrinsics(`--cam-k`)
> 세 개가 모두 있어야 실행됩니다. 메트릭 스케일이 보장된 simulator asset을 만드는
> 것이 목적이므로 depth 없이 돌리는 경로는 지원하지 않습니다.
>
> ICP/VGICP 등 정합 단계는 이 레포에서 완전히 제거되었습니다. 필요하면 상위
> 스텝에서 별도 처리하세요.

## Quick Start

레포에 포함된 샘플(`datas/move_ham_onto_box/`)로 바로 실행해볼 수 있습니다.

```bash
# 1) 환경 준비 (자세한 내용은 INSTALL.md)
export HF_TOKEN=hf_...            # SAM3D 체크포인트 다운로드에 필요
export SAM2_ROOT=/path/to/sam2
export SAM3D_ROOT=/path/to/sam-3d-objects

# 2) 기본 실행 (전체 pointmap + 자동 마스킹 + decimate on)
sam3d-asset-extractor \
  --image       datas/move_ham_onto_box/rgb/000000.png \
  --depth-image datas/move_ham_onto_box/depth/000000.png \
  --cam-k       datas/move_ham_onto_box/cam_K.txt \
  --output-dir  outputs/demo \
  --overwrite

# 3) 마스크 영역만 pointmap으로 넣고 싶을 때
sam3d-asset-extractor \
  --image       datas/move_ham_onto_box/rgb/000000.png \
  --depth-image datas/move_ham_onto_box/depth/000000.png \
  --cam-k       datas/move_ham_onto_box/cam_K.txt \
  --sam3d-input cropped \
  --output-dir  outputs/demo --overwrite
```

설치 없이 저장소 안에서 바로 실행하려면:
```bash
./run_pipeline.sh \
  --image       datas/move_ham_onto_box/rgb/000000.png \
  --depth-image datas/move_ham_onto_box/depth/000000.png \
  --cam-k       datas/move_ham_onto_box/cam_K.txt \
  --output-dir  outputs/demo --overwrite
```

### 포함된 샘플 데이터

| 경로 | 내용 |
|---|---|
| `datas/move_ham_onto_box/rgb/000000.png` | 848×480 RGB |
| `datas/move_ham_onto_box/depth/000000.png` | 848×480 16-bit depth (mm 단위, `--depth-scale auto`가 0.001로 해석) |
| `datas/move_ham_onto_box/cam_K.txt` | 3×3 카메라 intrinsics |

## 주요 CLI 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--image` | (필수) | 입력 RGB 이미지 |
| `--depth-image` | (필수) | depth 이미지 (RGB와 정렬된 상태) |
| `--cam-k` | (필수) | 카메라 intrinsics (3x3 또는 fx fy cx cy) 텍스트 |
| `--output-dir` | (필수) | 실행 결과 루트 |
| `--depth-scale` | `auto` | depth → meter 스케일 (`auto` 또는 숫자) |
| `--sam2-mode` | `auto` | `auto`(`SAM2AutomaticMaskGenerator`) 또는 `manual`(대화형 UI) |
| `--sam3d-input` | `full` | SAM3D에 넘길 pointmap 범위. `full`: 전체, `cropped`: 마스크 영역만 |
| `--mesh-format` | `all` | `glb`/`ply`/`obj`/`all` |
| `--decimate / --no-decimate` | on | 메시 decimate 수행 여부 |
| `--decimate-method` | `auto` | `auto`/`open3d`/`trimesh`/`cluster` |
| `--decimate-target-faces` | 20000 | 목표 face 수 (0 이하면 ratio 사용) |
| `--decimate-ratio` | 0.02 | 목표 face 비율 |
| `--decimate-min-faces` | 200 | 최소 face 수 하한 |
| `--sam2-env` / `--sam3d-env` | `sam2` / `sam3d-objects` | conda env 이름 |
| `--sam2-checkpoint` | None | SAM2 체크포인트 경로 (기본은 `$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt`) |
| `--sam3d-config` | None | SAM3D pipeline.yaml (기본은 `$SAM3D_ROOT/checkpoints/hf/pipeline.yaml`) |
| `--sam3d-seed` | 42 | SAM3D 재현 시드 |
| `--overwrite` | off | `output-dir`가 존재해도 덮어씀 |
| `--latest-only` | off | 생성된 여러 마스크 중 가장 최신 1개만 처리 |
| `--dry-run` | off | preflight만 수행하고 종료 |
| `--log-level` | INFO | DEBUG/INFO/WARNING/ERROR |
| `--log-file` | None | stderr와 별개 로그 파일 |

## 출력 구조

```
<output-dir>/
  manifest.json                      # 실행 요약 (입력/옵션/산출물 목록)
  sam2_masks/
    vis_<image_stem>.png             # 마스크 오버레이 (auto 모드)
    <image_stem>_000.png             # 마스크 (1개당 PNG)
    ...
  sam3d/
    <image_stem>_<idx>.ply           # 가우시안 PLY
    <image_stem>_<idx>_mesh.{glb|ply|obj}
    <image_stem>_<idx>_pointmap_full.{npz|ply}  # depth 입력 시
    <image_stem>_<idx>_pose.json
    <image_stem>_<idx>_pose.ply
    <image_stem>_<idx>_pose_mesh.{glb|ply|obj}
  decimated/                         # --decimate 시
    <image_stem>_<idx>_pose_mesh_decimated.{glb|ply|obj}
    ...
```

## 패키지 구조

```
sam3d_asset_extractor/
  src/sam3d_asset_extractor/
    cli.py               # 진입점 (sam3d-asset-extractor)
    config.py            # PipelineConfig / DecimateOptions
    preflight.py         # conda env / HF_TOKEN / 입력 검증
    logging_setup.py     # 패키지 로거
    paths.py             # SAM2_ROOT / SAM3D_ROOT 해석
    common/
      camera.py          # intrinsics 파싱
      depth.py           # depth 이미지 로드 + 스케일 해석
      geometry.py        # backprojection / MAD 필터 / pointmap
      ply_io.py          # PLY 읽기/쓰기 (중복 제거된 통합 구현)
    sam2_mask/
      auto.py            # 자동 마스킹 (sam2 env에서 실행)
      manual.py          # 포인트 기반 UI (sam2 env에서 실행)
      runner.py          # orchestrator에서 conda subprocess로 호출
    sam3d/
      inference.py       # SAM3D 추론 + pose + mesh 저장 (sam3d-objects env)
      pointmap.py        # depth→pointmap, Pytorch3D frame 변환
      pose.py            # pose 파싱/적용 (quat/6d/matrix 대응)
      export.py          # mesh 저장 + world Z-up 회전
      runner.py          # orchestrator에서 subprocess 호출
    mesh/
      decimate.py        # quadric/vertex-cluster decimate
  configs/               # (선택) 프로젝트별 설정 YAML 자리
  tests/                 # pytest
  datas/
    move_ham_onto_box/
      rgb/000000.png
      depth/000000.png
      cam_K.txt
  scripts/
    setup_externals.sh   # sam2 + sam-3d-objects clone + 체크포인트
    setup_envs.sh        # conda env 생성 자동화
  sam2/                  # ← git clone (gitignored)
  sam-3d-objects/        # ← git clone (gitignored)
  requirements-sam3d-runtime.txt  # sam-3d-objects deps (bpy 제외)
  run_pipeline.sh        # python -m 래퍼
  pyproject.toml
  requirements.txt
  README.md
  INSTALL.md
```

## 외부 의존성

SAM2와 SAM3D Objects를 **본 레포 내부에 `git clone`**하여 사용합니다.
`pip install -e`로 외부 레포를 설치하지 않으며(sam2의 C++ 빌드만 예외), `sys.path`로
직접 참조합니다. `.gitignore`에 의해 커밋되지 않습니다.

```bash
# 레포 clone 후 한 번만
bash scripts/setup_externals.sh   # sam2, sam-3d-objects clone + 체크포인트
bash scripts/setup_envs.sh        # conda env 2개 생성 + deps 설치
```

단계별 수동 설치는 [INSTALL.md](INSTALL.md)를 참고하세요.

### 입력 요구사항
RGB + depth + intrinsics 세 개가 **항상** 필요합니다. 하나라도 누락되면
`argparse` 단계에서 실행이 차단됩니다. depth는 RGB와 동일 해상도로 정렬되어
있어야 하며(필요 시 내부에서 nearest-neighbor resize), scale은 `--depth-scale`로
직접 지정하거나 `auto` 힌트에 맡길 수 있습니다 (uint16/큰 정수 depth는 mm로 간주).

## 트러블슈팅

자주 만나는 이슈들. 전체 표는 [INSTALL.md 트러블슈팅](INSTALL.md#트러블슈팅) 참조.

- `undefined symbol: iJIT_NotifyEvent` — MKL 2025 이슈. `conda install -n <env> -c conda-forge "mkl<2025"` 로 다운그레이드
- `Could not find a version that satisfies the requirement bpy==4.3.0` — Python 3.10에 bpy 4.3 wheel이 없음. `pip install ... || true` 로 무시 가능 (본 파이프라인은 bpy 미사용)
- `No module named pytest` — dev extras 미설치. `pip install -e ".[dev]"` 로 재설치
- `conda not found in PATH` — `source ~/anaconda3/etc/profile.d/conda.sh` 또는 환경변수 점검
- `sam2 / sam3d_objects import 실패` — 각 conda env에서 `pip install -e ./sam2` / `pip install -e ./sam-3d-objects[inference]`
- `sam2._C` 관련 오류 — SAM2 C++/CUDA 확장 재빌드 (`cd $SAM2_ROOT && pip install -e .`)
- `HF_TOKEN not set` — `export HF_TOKEN=hf_...` 또는 `--skip-hf-check`
- `cv2.error … waitKey` (manual 모드) — 원격 세션에서 실행 중이면 X 포워딩 또는 VNC 필요

## 제거된 기능 (이전 `sam3d_metric_scale_org/`에서 옮겨오지 않음)

- ICP/VGICP 기반 스케일 정합 (`sam3d_scale.py`, `sam3d_scale_icp.py`, `sam3d_scale_vgicp.py`)
- VGICP pose refinement 훅 (`--pose-refine-*`)
- `real_depth_scale.py`의 metric target 생성 (통합 X, 필요하면 수동 후처리)
- `fast_gicp/` 전체 (미사용 C++ 라이브러리)
- `preflight_check.py`의 scale 관련 체크
- 레포에 커밋돼 있던 `hugging_face_token.txt`

이전 버전의 기능이 필요하면 `sam3d_metric_scale_org/`를 참고하세요.
