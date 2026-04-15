# INSTALL

이 파이프라인은 서로 격리된 두 개의 conda 환경을 사용합니다.

| 환경 | 용도 | 비고 |
|---|---|---|
| `sam2` | SAM2 마스킹 | `sam2` 레포 + PyTorch + CUDA |
| `sam3d-objects` | SAM3D 추론 + mesh 후처리 | `sam-3d-objects` 레포 + PyTorch3D + kaolin + trimesh |

## 1. 외부 레포 가져오기

```bash
cd /path/to/parent
git clone https://github.com/facebookresearch/sam2.git
git clone <sam-3d-objects-repo-url> sam-3d-objects
```

환경변수로 위치를 알려줍니다 (또는 본 레포와 같은 상위 디렉토리에 두면 자동 탐색됨):
```bash
export SAM2_ROOT=/path/to/sam2
export SAM3D_ROOT=/path/to/sam-3d-objects
```

## 2. conda env 생성

```bash
# SAM2 환경
conda create -n sam2 python=3.10 -y
conda activate sam2
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
cd "$SAM2_ROOT" && pip install -e .
pip install numpy opencv-python Pillow

# SAM3D 환경
conda create -n sam3d-objects python=3.10 -y
conda activate sam3d-objects
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
cd "$SAM3D_ROOT" && pip install -e ".[inference]"
pip install numpy opencv-python Pillow plyfile trimesh open3d
```

> PyTorch / CUDA 버전은 GPU에 맞게 조정하세요. PyTorch3D / kaolin은 CUDA 버전에 매우
> 민감하므로 공식 빌드 차트를 참고합니다.

## 3. HuggingFace 토큰

SAM3D 체크포인트는 HF hub에서 자동 다운로드됩니다. 토큰을 환경변수로 등록하세요
(레포에 파일로 커밋하지 않습니다):

```bash
export HF_TOKEN=hf_xxx_your_token
# 또는 HUGGING_FACE_HUB_TOKEN=hf_xxx
```

토큰 체크를 일시적으로 비활성화하려면 CLI에 `--skip-hf-check`를 추가하세요.

## 4. 체크포인트 준비

### SAM2
```bash
cd "$SAM2_ROOT"
bash checkpoints/download_ckpts.sh   # 기본 sam2.1_hiera_large.pt 포함
```
`--sam2-checkpoint` 로 다른 체크포인트 경로를 지정할 수 있습니다.

### SAM3D
`$SAM3D_ROOT/checkpoints/hf/pipeline.yaml` 위치를 확인하세요. 체크포인트 파일은
첫 실행 시 자동으로 다운로드됩니다.

## 5. 본 레포 설치

editable install을 권장합니다:
```bash
cd /path/to/sam3d_asset_extractor
pip install -e .
```
설치하지 않고 실행하려면 [run_pipeline.sh](run_pipeline.sh)에서 `PYTHONPATH=src`를
자동으로 세팅합니다.

## 6. Smoke test

레포에 포함된 샘플 데이터 (`datas/move_fruits_into_bowl/`)를 그대로 사용합니다.
RGB 1장 + 정렬된 depth + 카메라 intrinsics 세 개가 모두 같이 들어있습니다.

```bash
# 1) CLI help
sam3d-asset-extractor --help

# 2) dry-run (의존성 체크 없이 입력/레이아웃만 확인)
sam3d-asset-extractor \
  --image       datas/move_fruits_into_bowl/rgb/000000.png \
  --depth-image datas/move_fruits_into_bowl/depth/000000.png \
  --cam-k       datas/move_fruits_into_bowl/cam_K.txt \
  --output-dir  /tmp/sae_smoke \
  --dry-run

# 3) 실제 실행 (GPU + conda env 갖춰진 경우)
sam3d-asset-extractor \
  --image       datas/move_fruits_into_bowl/rgb/000000.png \
  --depth-image datas/move_fruits_into_bowl/depth/000000.png \
  --cam-k       datas/move_fruits_into_bowl/cam_K.txt \
  --output-dir  /tmp/sae_smoke \
  --latest-only --overwrite
```

## 트러블슈팅

- `HF_TOKEN not set` → 환경변수 등록 또는 `--skip-hf-check`
- `sam2._C` / `undefined symbol` → SAM2 C++ extension 재빌드
- `numpy dtype size changed` → 해당 env에서 binary 패키지(kaolin 등) 재설치
- `conda: command not found` → `source $CONDA_ROOT/etc/profile.d/conda.sh`
- `cv2.error ... waitKey` (manual 모드) → 원격 세션이면 X 포워딩 / VNC 필요
- `--sam3d-input cropped ... requires --depth-image and --cam-k` → depth와 intrinsics 모두 전달

## 제거된 의존성

이전 버전에서 요구하던 아래 패키지는 이제 **불필요**합니다:
- `small_gicp` (VGICP)
- Open3D의 ICP 모듈 (mesh decimation 용 Open3D는 선택 사항)

이전 파이프라인이 필요하면 `sam3d_metric_scale_org/`를 참고하세요.
