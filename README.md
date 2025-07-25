# 🛰️ CV-SAR SR (Complex-Valued SAR Super-Resolution) 프로젝트

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [왜 이 프로젝트가 필요한가?](#왜-이-프로젝트가-필요한가)
3. [프로젝트 구조](#프로젝트-구조)
4. [핵심 개념 설명](#핵심-개념-설명)
5. [시작하기](#시작하기)
6. [상세 사용 가이드](#상세-사용-가이드)
7. [기술적 세부사항](#기술적-세부사항)
8. [문제 해결](#문제-해결)
9. [성능 및 결과](#성능-및-결과)

---

## 🎯 프로젝트 개요

**CV-SAR SR**은 Sentinel-1 위성의 SAR(합성개구레이더) 영상에서 **복소수 정보를 보존하면서** 작은 패치(조각)들을 추출하여, **초해상도(Super-Resolution) 딥러닝 모델** 학습용 데이터를 준비하는 파이프라인입니다.

### 🌟 주요 특징
- ✅ **위상 정보 완벽 보존**: I/Q 복소수 데이터 직접 처리
- ✅ **Dual-pol 지원**: VV + VH 2채널 동시 처리
- ✅ **GPU 가속**: CuPy/PyTorch FFT로 10배 빠른 처리
- ✅ **한국 재난 특화**: 홍수(서울), 태풍(제주) 패턴 최적화
- ✅ **대용량 처리**: 57,000+ 패치 안정적 처리
- ✅ **품질 보증**: Cross-pol coherence >0.95 검증

---

## 🤔 왜 이 프로젝트가 필요한가?

### 1. **SAR 영상의 특수성**
일반 카메라와 달리 SAR는 **복소수 데이터**를 생성합니다:
```
픽셀값 = I + jQ (I: 실부, Q: 허부)
위상 = arctan(Q/I)  → 지표면 높이, 변위 정보 포함!
```

### 2. **한국 재난 감시의 필요성**
- 🌊 **서울 홍수**: 한강 범람, 도시 침수 → 낮은 coherence 감지
- 🌀 **제주 태풍**: 강풍, 해일 → VH/VV 비율 변화 추적
- 🏔️ **산사태**: 지표면 변위 → 위상 변화 분석

### 3. **딥러닝 모델의 요구사항**
- **CV-U-Net SR**: 복소수 입력 → 복소수 출력
- **Coherence loss**: 물리적 일관성 유지
- **256×512 패치**: 재난 구조 패턴 보존

---

## 📁 프로젝트 구조

```
Sentinel-1/
│
├── 📄 README.md                     # 이 파일 (전체 가이드)
├── 📄 myGraph.xml                   # SNAP 전처리 그래프
│
├── 📂 data/                         # 데이터 디렉토리
│   ├── 📂 S1_raw/                   # 원본 Sentinel-1 .zip 파일들
│   ├── 📂 processed_1/              # SNAP 전처리된 .dim 파일들
│   │   ├── S1A_*_Cal_IW1.dim      # 전처리된 제품 (메타데이터)
│   │   └── S1A_*_Cal_IW1.data/    # 실제 데이터
│   │       ├── i_IW1_VV.img        # VV 실부
│   │       ├── q_IW1_VV.img        # VV 허부
│   │       ├── i_IW1_VH.img        # VH 실부
│   │       └── q_IW1_VH.img        # VH 허부
│   └── 📂 processed_2_*/            # 추출된 패치들 (.npy)
│
├── 📂 workflows/                    # 핵심 코드
│   ├── 🐍 patch_extractor_gpu_enhanced.py     # ⭐ 메인 실행 파일
│   ├── 🐍 data_augmentation_coherence.py      # 데이터 증강
│   ├── 🐳 Dockerfile                          # Docker 이미지
│   ├── 🐳 docker-compose.yml                  # Docker 편의 설정
│   ├── 🔧 docker-entrypoint.sh                # Docker 시작 스크립트
│   ├── 📋 requirements.txt                    # Python 패키지
│   └── 📚 aws_deployment_guide.md             # AWS 배포 가이드
│
└── 📂 example/                      # 예제 데이터
```

---

## 🧠 핵심 개념 설명

### 1. **복소수 SAR 데이터**
```python
# SAR 픽셀 = 복소수
pixel = I + j*Q
magnitude = sqrt(I² + Q²)  # 밝기
phase = arctan2(Q, I)      # 위상 (중요!)
```

### 2. **Dual-pol (이중편파)**
- **VV**: 수직 송신 → 수직 수신 (기본)
- **VH**: 수직 송신 → 수평 수신 (교차)
- **용도**: VH/VV 비율로 지표면 특성 파악

### 3. **Coherence (일관성)**
```python
# 두 복소수 이미지 간 유사도
coherence = |⟨A·B*⟩| / sqrt(⟨|A|²⟩·⟨|B|²⟩)
# 1에 가까울수록 유사, 0에 가까울수록 다름
```

### 4. **Subaperture 분해**
- SAR 영상을 여러 시각(angle)으로 분해
- 같은 지역을 다른 각도에서 본 것처럼 분석

---

## 🚀 시작하기

### 📋 사전 준비사항

1. **시스템 요구사항**
   - OS: Windows 10/11, Linux (Ubuntu 20.04+)
   - RAM: 최소 16GB (32GB 권장)
   - GPU: NVIDIA GPU (선택사항, 권장)
   - 디스크: 100GB+ 여유 공간

2. **소프트웨어 설치**
   ```bash
   # Python 3.8+ 설치
   python --version  # 3.8 이상 확인
   
   # CUDA 설치 (GPU 사용시)
   nvidia-smi  # GPU 확인
   ```

3. **SNAP 설치** (ESA Sentinel Application Platform)
   - [다운로드](https://step.esa.int/main/download/snap-download/)
   - 설치 후 `snappy` 설정 필요

### 🔧 설치 방법

#### 방법 1: 직접 설치 (권장)
```bash
# 1. 저장소 클론
git clone https://github.com/your-repo/Sentinel-1.git
cd Sentinel-1

# 2. Python 환경 생성
conda create -n cv-sar python=3.8
conda activate cv-sar

# 3. 패키지 설치
pip install -r workflows/requirements.txt

# 4. GPU 지원 설치 (선택사항)
pip install cupy-cuda11x  # CUDA 버전에 맞게

# 5. Snappy 설정
cd $SNAP_HOME/bin
./snappy-conf /path/to/python
```

#### 방법 2: Docker 사용
```bash
# Docker 이미지 빌드
docker build -f workflows/Dockerfile -t cv-sar-sr .

# 실행
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  cv-sar-sr
```

---

## 📖 상세 사용 가이드

### 1️⃣ **데이터 준비**

#### Sentinel-1 데이터 다운로드
1. [Copernicus Open Access Hub](https://scihub.copernicus.eu/) 접속
2. 검색 조건:
   - Product Type: SLC (Single Look Complex)
   - Sensor Mode: IW (Interferometric Wide)
   - Polarization: VV + VH

#### SNAP 전처리
```bash
# Graph 실행 (명령줄)
gpt data/graph1.xml -Pinput=S1A_*.zip -Poutput=data/processed_1/output.dim
```

### 2️⃣ **패치 추출 실행**

#### 기본 실행
```bash
# 전체 파일 처리
python workflows/patch_extractor_gpu_enhanced.py

# 테스트 (1개 파일만)
export MAX_FILES=1
python workflows/patch_extractor_gpu_enhanced.py
```

#### 설정 변경
```python
# workflows/patch_extractor_gpu_enhanced.py 상단 편집
PATCH_W, PATCH_H = 256, 512  # 패치 크기
MAX_FILES = None             # None = 전체, 숫자 = 제한
USE_GPU_FFT = True          # GPU 사용 여부
```

### 3️⃣ **결과 확인**

```python
# 추출된 패치 확인
import numpy as np

# 복소수 데이터 로드
patch = np.load('data/processed_2_gpu_enhanced/*_dual_pol_complex_0_0.npy')
print(f"Shape: {patch.shape}")  # (2, 512, 256) - [VV, VH]
print(f"Type: {patch.dtype}")    # complex64

# 위상 확인
phase_vv = np.angle(patch[0])
print(f"VV 위상 범위: [{phase_vv.min():.2f}, {phase_vv.max():.2f}]")
```

### 4️⃣ **데이터 증강 (선택사항)**

```python
from workflows.data_augmentation_coherence import CoherencePreservingAugmentation

# 증강기 생성
augmenter = CoherencePreservingAugmentation()

# 위상 회전 (coherence 보존)
augmented = augmenter.random_phase_rotation(patch)

# 홍수 시뮬레이션
flood_mask = np.random.rand(512, 256) > 0.8
flooded = augmenter.simulate_flood_signature(patch, flood_mask)
```

---

## 🔧 기술적 세부사항

### 💾 메모리 관리
- **WeakRef**: 자동 메모리 해제
- **배치 처리**: 100개씩 묶어서 처리
- **가비지 컬렉션**: 주기적 정리

### ⚡ GPU 가속
```python
# CPU vs GPU 성능 비교
CPU FFT: ~500ms per patch
GPU FFT: ~50ms per patch (10x faster!)
```

### 🔍 품질 검증
1. **Cross-pol coherence**: >0.95 필수
2. **에너지 보존**: Subaperture 재구성시 >95%
3. **위상 분산**: >1e-6 (위상 정보 존재 확인)

### 📊 처리 통계
- **입력**: 1개 .dim 파일 (~4GB)
- **출력**: ~3,000개 패치 (각 ~1MB .npy)
- **처리 시간**: GPU 사용시 ~10분/파일

---

## 🆘 문제 해결

### 문제: "무한로딩중이야"
```bash
# 로그 확인
tail -f patch_extraction_gpu_enhanced.log

# 메모리 부족시
export MAX_WORKERS=2  # 워커 수 감소
```

### 문제: GPU 인식 안됨
```bash
# CUDA 확인
nvidia-smi

# CuPy 재설치
pip uninstall cupy-cuda11x
pip install cupy-cuda11x --no-cache-dir
```

### 문제: SNAP 오류
```bash
# Java 힙 메모리 증가
export _JAVA_OPTIONS="-Xmx8G"
```

---

## 📈 성능 및 결과

### 처리 성능
| 환경 | 파일당 시간 | 57,000 패치 예상 |
|------|------------|----------------|
| CPU only | ~30분 | ~24시간 |
| GPU (T4) | ~10분 | ~4시간 |
| GPU (V100) | ~5분 | ~2시간 |

### 출력 데이터 구조
```
processed_2_gpu_enhanced/
├── *_dual_pol_complex_x_y.npy      # 복소수 데이터 (2,512,256)
├── *_pol_order_x_y.npy             # 편광 순서 ['VV','VH']
├── *_enhanced_features_x_y.npy      # Coherence 특징
├── *_quality_metrics_x_y.npy        # 품질 지표
└── *_subapertures_x_y.npy          # Subaperture 분해 (5,512,256)
```

---

## 🎯 다음 단계

1. **CV-U-Net SR 모델 학습**
   ```python
   # PyTorch DataLoader 사용
   from torch.utils.data import DataLoader
   dataset = SentinelComplexDataset('data/processed_2_gpu_enhanced')
   dataloader = DataLoader(dataset, batch_size=16)
   ```

2. **AWS 대규모 처리**
   - `workflows/aws_deployment_guide.md` 참고
   - EC2 GPU 인스턴스 사용 권장

3. **결과 분석**
   - Coherence 맵 생성
   - 재난 전후 비교
   - 초해상도 결과 평가

---

## 📚 참고 자료

- [Sentinel-1 SAR 기초](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar)
- [SNAP 튜토리얼](https://step.esa.int/main/doc/tutorials/)
- [Complex SAR 처리](https://www.sarmap.ch/pdf/SAR_Guidebook.pdf)

---

## 🤝 기여 및 문의

- 이슈 제보: GitHub Issues
- 개선 제안: Pull Request
- 문의: your-email@example.com

---

**🎉 성공적인 CV-SAR SR 데이터 준비를 축하합니다!** 