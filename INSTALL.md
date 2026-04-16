# INSTALL

## 개요

이 파이프라인은 두 개의 외부 레포(SAM2, SAM3D Objects)를 **레포 내부에 clone**하여
`sys.path`로 직접 참조합니다. `pip install`로 외부 레포를 설치하지 않으므로 sam-3d-objects의
`bpy` 의존성 문제를 회피합니다.

| 환경 | 용도 |
|---|---|
| `sam2` | SAM2 마스킹 (C++ extension 빌드 필요) |
| `sam3d-objects` | SAM3D 추론 + mesh 후처리 |

### 검증된 구성

| 항목 | 버전 |
|---|---|
| GPU | RTX 4090 (24 GB) |
| NVIDIA driver | CUDA 13.0 (forward-compat) |
| PyTorch | 2.5.1 + cu121 (sam3d) / cu124 (sam2) |
| Python | 3.10 (sam2) / 3.11 (sam3d, 공식 env yml) |
| MKL | `<2025` 필수 (2025는 PyTorch 2.5.1과 심볼 충돌) |

---

## 빠른 설치 (스크립트 사용)

```bash
git clone https://github.com/swoonge/sam3d_asset_extractor.git
cd sam3d_asset_extractor

# 1) 외부 레포 clone + 체크포인트 다운로드 (~14 GB)
export HF_TOKEN=hf_xxx_your_token
bash scripts/setup_externals.sh

# 2) conda env 생성 + 패키지 설치
bash scripts/setup_envs.sh

# 3) 검증
conda run -n sam3d-objects python -m pytest tests/
conda run -n sam3d-objects sam3d-asset-extractor \
  --image datas/move_ham_onto_box/rgb/000000.png \
  --depth-image datas/move_ham_onto_box/depth/000000.png \
  --cam-k datas/move_ham_onto_box/cam_K.txt \
  --output-dir outputs/smoke --latest-only --overwrite
```

---

## 수동 설치 (단계별)

### Step 1: 레포 clone

```bash
git clone https://github.com/swoonge/sam3d_asset_extractor.git
cd sam3d_asset_extractor
```

### Step 2: 외부 레포를 **레포 내부에** clone

```bash
# SAM2
git clone https://github.com/facebookresearch/sam2.git

# SAM3D Objects
git clone https://github.com/facebookresearch/sam-3d-objects.git
```

결과 디렉토리 구조:
```
sam3d_asset_extractor/
  sam2/                    ← .gitignore에 의해 커밋되지 않음
  sam-3d-objects/          ← 마찬가지
  src/
  ...
```

### Step 3: 체크포인트 다운로드

**SAM2** (~1.5 GB):
```bash
cd sam2/checkpoints && bash download_ckpts.sh && cd ../..
```

**SAM3D Objects** (~12 GB, HuggingFace):

⚠️ [HF 레포](https://huggingface.co/facebook/sam-3d-objects)에서 **Request access**를
먼저 해야 합니다.

```bash
export HF_TOKEN=hf_xxx_your_token
pip install 'huggingface-hub[cli]<1.0'

cd sam-3d-objects
TAG=hf
hf download --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
cd ..

# 확인
ls sam-3d-objects/checkpoints/hf/pipeline.yaml
```

### Step 4: `sam2` conda env 생성

```bash
conda create -y -n sam2 python=3.10
conda install -y -n sam2 \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.4 "mkl<2025"

# SAM2 패키지 editable 설치 (C++ extension 빌드)
conda run --no-capture-output -n sam2 pip install -e ./sam2
conda run --no-capture-output -n sam2 pip install numpy opencv-python Pillow

# 검증
conda run -n sam2 python -c "
import torch; from sam2.build_sam import build_sam2
print('sam2 OK, cuda:', torch.cuda.is_available())"
```

### Step 5: `sam3d-objects` conda env 생성

두 가지 방법 중 택 1:

#### 방법 A: 공식 env yml 사용 (권장)

```bash
# sam-3d-objects 공식 conda env (Python 3.11 + CUDA 12.1 toolkit)
conda env create -y -f sam-3d-objects/environments/default.yml

# PyTorch (공식 env가 깔아주지 않으므로 직접)
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
conda install -y -n sam3d-objects \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.1 "mkl<2025"
```

#### 방법 B: 최소 env + 직접 deps

```bash
conda create -y -n sam3d-objects python=3.10
conda install -y -n sam3d-objects \
  -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision pytorch-cuda=12.4 "mkl<2025"
```

#### 이어서 (A/B 공통): sam-3d-objects deps 설치

```bash
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"

# base deps (bpy 제외한 curated 목록)
conda run --no-capture-output -n sam3d-objects \
  pip install -r requirements-sam3d-runtime.txt

# pytorch3d + flash_attn
conda run --no-capture-output -n sam3d-objects \
  pip install -r sam-3d-objects/requirements.p3d.txt || true

# kaolin + gsplat + seaborn + gradio
conda run --no-capture-output -n sam3d-objects \
  pip install -r sam-3d-objects/requirements.inference.txt || true
```

> **NOTE**: `pytorch3d`, `kaolin`, `gsplat`는 CUDA 버전/torch 빌드에 민감해 실패할 수
> 있습니다. `|| true`로 넘기고 실제 실행 시 에러가 나면 그때 해당 패키지만 수동 재설치
> 또는 `--no-build-isolation` 등으로 대응합니다.

#### sam3d_asset_extractor 설치

```bash
conda run --no-capture-output -n sam3d-objects \
  pip install -e ".[dev]" open3d
```

#### 검증

```bash
conda run -n sam3d-objects python -c "
import os; os.environ['LIDRA_SKIP_INIT'] = 'true'
import torch, trimesh, plyfile, open3d
print('sam3d-objects env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

### Step 6: Smoke test

```bash
# 단위/통합 테스트 (GPU 불필요)
conda run -n sam3d-objects python -m pytest tests/

# dry-run
conda run -n sam3d-objects sam3d-asset-extractor \
  --image datas/move_ham_onto_box/rgb/000000.png \
  --depth-image datas/move_ham_onto_box/depth/000000.png \
  --cam-k datas/move_ham_onto_box/cam_K.txt \
  --output-dir /tmp/sae_smoke --dry-run

# 실제 실행 (GPU + HF_TOKEN 필요)
export HF_TOKEN=hf_xxx
conda run --no-capture-output -n sam3d-objects sam3d-asset-extractor \
  --image datas/move_ham_onto_box/rgb/000000.png \
  --depth-image datas/move_ham_onto_box/depth/000000.png \
  --cam-k datas/move_ham_onto_box/cam_K.txt \
  --output-dir outputs/smoke --latest-only --overwrite
```

---

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `undefined symbol: iJIT_NotifyEvent` | `conda install -n <env> -c conda-forge "mkl<2025"` |
| `bpy==4.3.0` 실패 | 무시 가능 — `requirements-sam3d-runtime.txt`에서 이미 제거됨 |
| `No module named pytorch3d` | `pip install -r sam-3d-objects/requirements.p3d.txt` |
| `No module named kaolin` | `PIP_FIND_LINKS=... pip install kaolin==0.17.0` |
| `No module named 'inference'` | `SAM3D_ROOT` 경로 확인 (레포 내부 `sam-3d-objects/` 가리켜야 함) |
| `Missing SAM3D config: .../pipeline.yaml` | 체크포인트 미다운로드. Step 3 재실행 |
| `HF_TOKEN not set` | `export HF_TOKEN=hf_...` 또는 `--skip-hf-check` |
| `sam2._C undefined symbol` | SAM2 C++ 후처리 호환성 경고. 결과에 영향 없음 (무시 가능) |

---

## 레포 내부 구조 (설치 후)

```
sam3d_asset_extractor/
  sam2/                          ← git clone (1.7 GB + checkpoints 1.5 GB)
  sam-3d-objects/                ← git clone (0.5 GB + checkpoints 12 GB)
  src/sam3d_asset_extractor/     ← 파이프라인 코드
  scripts/
    setup_externals.sh           ← clone + checkpoint 자동화
    setup_envs.sh                ← conda env 생성 자동화
  requirements-sam3d-runtime.txt ← sam-3d-objects base deps (bpy 제외)
  datas/move_ham_onto_box/       ← 샘플 RGB-D
  tests/                         ← pytest
  ...
```

`.gitignore`에 의해 `sam2/`, `sam-3d-objects/`, `outputs/` 등은 커밋되지 않습니다.
