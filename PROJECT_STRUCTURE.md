# Sentinel-1 SAR Super-Resolution 프로젝트 구조

이 문서는 본 프로젝트의 전체 디렉토리 및 파일 구조와 각 구성요소의 역할을 한눈에 파악할 수 있도록 정리한 문서입니다.

---

## 최상위 디렉토리 구조

```
Sentinel-1/
├── amp_phase_hist.py
├── analysis_results/
├── coherence_histogram.png
├── data/
├── improved_loss_functions.py
├── model/
├── model_weights/
├── plots/
├── results/
├── runs/
├── sar_comparative_analysis.py
├── visualize_subswath.py
├── workflows/
├── 데이터자동다운로드스크립트.py
├── 모델제안.md
```

---

## 주요 디렉토리 및 파일 설명

### 1. **data/**
- **설명:**
  - 원본 및 전처리된 SAR 데이터, 실험용 데이터셋, 데이터 프로파일, XML 그래프 등 데이터 관련 파일을 저장합니다.
  - `processed_1/`, `processed_2/` 등은 전처리 단계별 결과를 구분합니다.
  - `validate.ipynb`: 데이터 검증/분석용 노트북.

### 2. **model/**
- **설명:**
  - 모델 정의, 학습, 유틸리티, 시각화 등 딥러닝 관련 핵심 코드가 위치합니다.
  - **cv_unet.py**: 복소 U-Net 네트워크 구조 정의
  - **train.py**: 전체 학습 파이프라인 및 CLI
  - **utils.py**: 손실함수, 평가지표, 보조 함수 등
  - **data_cache.py**: 데이터 캐싱 및 로딩 최적화
  - **speed_utils.py**: 학습 속도 최적화 유틸리티
  - **visualize_tensorboard.py**: TensorBoard 로그 시각화
  - **plots/**: 학습 곡선 등 시각화 결과 저장

### 3. **workflows/**
- **설명:**
  - 데이터 증강, 패치 추출, 배포, 도커 관련 스크립트 및 문서
  - **data_augmentation_coherence.py**: SAR 데이터 증강
  - **patch_extractor_gpu_enhanced.py**: GPU 기반 패치 추출
  - **docker-compose.yml, Dockerfile**: 컨테이너 환경 설정
  - **aws_deployment_guide.md**: AWS 배포 가이드

### 4. **analysis_results/**
- **설명:**
  - 실험 결과, 분석 리포트, 시각화 이미지 등 결과물 저장
  - **sar_analysis_report.md**: 분석 리포트
  - **statistical_analysis.png**: 통계 분석 시각화

### 5. **model_weights/**
- **설명:**
  - 학습된 모델 가중치 파일 저장 (버전별 디렉토리 구분)

### 6. **runs/**
- **설명:**
  - TensorBoard 로그, 실험별 결과 파일 저장
  - 각 실험별로 타임스탬프 기반 디렉토리 생성

### 7. **plots/**
- **설명:**
  - 학습 곡선, PSNR 등 주요 시각화 결과 저장

### 8. **results/**
- **설명:**
  - 최종 실험 결과, 예측 결과 등 저장 (비어있을 수 있음)

### 9. **기타 주요 파일**
- **amp_phase_hist.py**: 진폭/위상 히스토그램 분석
- **improved_loss_functions.py**: 개선된 손실 함수 구현
- **sar_comparative_analysis.py**: SAR 비교 분석 스크립트
- **visualize_subswath.py**: Subswath 시각화
- **데이터자동다운로드스크립트.py**: 데이터 자동 다운로드
- **모델제안.md**: 모델 구조 제안 및 논의 문서

---

## 코드 품질 및 구조적 특징
- **모듈성/재사용성:**
  - 모델, 데이터, 유틸리티, 시각화, 실험 스크립트가 명확히 분리되어 있어 유지보수 및 확장 용이
- **관심사 분리:**
  - 데이터 처리, 모델 학습, 실험 관리, 배포 등 각 관심사가 별도 디렉토리/파일로 관리됨
- **문서화:**
  - 주요 파이프라인, 배포, 데이터 프로파일, 분석 결과 등 문서화 파일 다수 포함
- **실험 재현성:**
  - 실험별 로그/결과가 `runs/`, `analysis_results/` 등에 체계적으로 저장됨

---

## 참고
- 각 디렉토리/파일의 세부 사용법 및 파이프라인은 해당 Python 파일의 docstring 또는 README, md 파일을 참고하세요.