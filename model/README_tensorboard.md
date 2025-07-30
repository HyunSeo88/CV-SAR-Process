# TensorBoard 시각화 가이드

## 개요

Complex U-Net SAR Super-Resolution 모델의 훈련 과정을 TensorBoard로 시각화하는 완전한 가이드입니다.

## 주요 기능

### 1. 실시간 훈련 모니터링
- **손실 곡선**: 훈련/검증 손실 실시간 추적
- **메트릭 추적**: PSNR, SSIM, CPIF 등 SAR 전용 메트릭
- **Cross-pol Coherence**: 이중편광 coherence 품질 모니터링

### 2. SAR 영상 시각화
- **입력/출력 비교**: LR 입력 vs HR 목표 vs SR 예측 결과
- **로그 스케일 표시**: SAR 데이터 특성에 맞는 시각화
- **다중 채널**: VV/VH 편광 채널별 분석

### 3. 복소수 데이터 분석
- **진폭/위상 통계**: 복소수 텐서의 통계적 분포
- **히스토그램**: 진폭과 위상의 분포 분석
- **시계열 추적**: 훈련 과정에서의 변화 추적

## 사용법

### 1. 훈련 시 TensorBoard 활성화

```python
# 기본 사용법 (자동 타임스탬프 디렉터리)
python train.py

# TensorBoard 비활성화
python train.py --disable-tensorboard

# 커스텀 로그 디렉터리
python train.py --tensorboard-log-dir my_experiment
```

### 2. TensorBoard 서버 실행

#### 방법 1: 자동 실행 스크립트
```bash
# 최신 실험 자동 실행
python visualize_tensorboard.py --launch

# 특정 실험 실행
python visualize_tensorboard.py --launch --log-dir runs/sar_sr_20250129_140000

# 포트 지정
python visualize_tensorboard.py --launch --port 6007
```

#### 방법 2: 직접 TensorBoard 실행
```bash
# 특정 디렉터리
tensorboard --logdir=runs/sar_sr_20250129_140000

# 모든 실험 비교
tensorboard --logdir=runs

# 포트 지정
tensorboard --logdir=runs --port=6006
```

### 3. 로그 분석 및 플롯 내보내기

```bash
# 훈련 로그 요약 분석
python visualize_tensorboard.py --analyze

# 훈련 곡선을 PNG로 내보내기
python visualize_tensorboard.py --export-plots

# 최신 실험 찾기
python visualize_tensorboard.py --find-latest

# 종합 분석
python visualize_tensorboard.py --analyze --export-plots
```

## TensorBoard 대시보드 구성

### 1. SCALARS 탭
- **Loss/Train**: 훈련 손실 (Hybrid Loss: 0.7×amplitude + 0.3×phase)
- **Loss/Val**: 검증 손실
- **PSNR/Train**: 훈련 PSNR (dB)
- **PSNR/Val**: 검증 PSNR (dB)
- **HR_Target/**: 고해상도 목표 이미지 통계
  - Amplitude_Mean, Amplitude_Std, Amplitude_Max
  - Phase_Mean, Phase_Std
- **SR_Prediction/**: 초해상도 예측 통계

### 2. IMAGES 탭
- **SAR_Images/LR_Input**: 저해상도 입력 (128×64)
- **SAR_Images/HR_Target**: 고해상도 목표 (512×256)
- **SAR_Images/SR_Prediction**: 초해상도 예측 (512×256)

### 3. HISTOGRAMS 탭
- **HR_Target/Amplitude_Distribution**: 목표 진폭 분포
- **HR_Target/Phase_Distribution**: 목표 위상 분포
- **SR_Prediction/Amplitude_Distribution**: 예측 진폭 분포
- **SR_Prediction/Phase_Distribution**: 예측 위상 분포

### 4. HPARAMS 탭
- 하이퍼파라미터 설정 추적
- 실험 간 성능 비교

## 데이터 해석 가이드

### 1. 손실 곡선 분석
```
정상적인 패턴:
- 훈련/검증 손실 모두 감소
- 검증 손실이 훈련 손실보다 약간 높음
- 진동 없이 부드러운 감소

문제 패턴:
- 검증 손실 증가 (과적합)
- 큰 진동 (학습률 너무 높음)
- 정체 (학습률 너무 낮음)
```

### 2. PSNR 목표 값
```
SAR 초해상도 PSNR 기준:
- 25-30 dB: 기본 품질
- 30-35 dB: 좋은 품질  
- 35+ dB: 우수한 품질
```

### 3. 복소수 데이터 특성
```
진폭 (Amplitude):
- 평균: SAR 후방산란 강도
- 분포: 레일리/감마 분포 예상
- 최대값: 강한 반사체 (도시, 금속)

위상 (Phase):
- 평균: ~0 (중심화된 위상)
- 분포: [-π, π] 균등 분포
- 표준편차: 위상 노이즈 수준
```

## 실험 관리

### 1. 디렉터리 구조
```
model/
├── runs/                    # TensorBoard 로그
│   ├── sar_sr_20250129_140000/
│   ├── sar_sr_20250129_150000/
│   └── ...
├── plots/                   # 내보낸 플롯
│   ├── training_loss.png
│   └── training_psnr.png
└── cv_unet.pth             # 저장된 모델
```

### 2. 실험 비교
```bash
# 여러 실험 동시 시각화
tensorboard --logdir=runs

# 특정 실험들만 비교
tensorboard --logdir_spec=exp1:runs/sar_sr_20250129_140000,exp2:runs/sar_sr_20250129_150000
```

### 3. 로그 관리
```bash
# 오래된 로그 정리 (30일 이상)
find runs -name "sar_sr_*" -type d -mtime +30 -exec rm -rf {} \;

# 로그 크기 확인
du -sh runs/*
```

## 고급 활용

### 1. 커스텀 메트릭 추가

```python
# train.py에서 추가 메트릭 로깅
if writer:
    # Cross-pol ratio
    vv_amp = torch.abs(hr_batch[:, 0])
    vh_amp = torch.abs(hr_batch[:, 1])
    pol_ratio = torch.mean(vh_amp / (vv_amp + 1e-8))
    writer.add_scalar('Metrics/VH_VV_Ratio', pol_ratio, epoch)
    
    # Coherence quality
    coherence = calculate_cross_pol_coherence(pred_hr[:, 0], pred_hr[:, 1])
    writer.add_scalar('Quality/Cross_Pol_Coherence', coherence, epoch)
```

### 2. 모델 아키텍처 시각화

```python
# 모델 그래프 로깅 (이미 구현됨)
dummy_input = torch.randn(1, 2, 128, 64, dtype=torch.complex64).to(device)
writer.add_graph(model, dummy_input)
```

### 3. 학습률 스케줄링 추적

```python
# 학습률 변화 추적
current_lr = optimizer.param_groups[0]['lr']
writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
```

## 문제 해결

### 1. TensorBoard 접속 안 됨
```bash
# 포트 충돌 확인
netstat -tulpn | grep :6006

# 다른 포트 사용
tensorboard --logdir=runs --port=6007
```

### 2. 메모리 부족
```bash
# 이미지 로깅 빈도 줄이기 (train.py)
if epoch % 10 == 0:  # 10 에폭마다 이미지 로깅
    log_images_to_tensorboard(...)
```

### 3. 로그 파일 손상
```bash
# 이벤트 파일 재생성
rm runs/sar_sr_*/events.out.tfevents.*
# 훈련 재시작
```

## 성능 최적화

### 1. 로깅 빈도 조절
```python
# 배치마다 -> 에폭마다
if batch_idx % 100 == 0:  # 100 배치마다
    writer.add_scalar('Loss/Batch', loss.item(), global_step)
```

### 2. 이미지 로깅 최적화
```python
# 이미지 수 줄이기
log_images_to_tensorboard(writer, lr_batch, hr_batch, pred_batch, epoch, num_images=2)
```

### 3. 히스토그램 로깅 선택적 활성화
```python
# 주요 에폭에서만 히스토그램 로깅
if epoch in [0, num_epochs//2, num_epochs-1]:
    writer.add_histogram('Weights/Conv1', model.enc1.conv.conv_real.weight, epoch)
```

## 결과 해석 및 개선

### 1. 수렴 패턴 분석
- **빠른 수렴**: 학습률 적절, 데이터 품질 좋음
- **느린 수렴**: 학습률 증가 또는 배치 크기 조정 고려
- **진동**: 학습률 감소 또는 gradient clipping 적용

### 2. 품질 메트릭 목표
- **PSNR > 30dB**: 한국 재해 모니터링에 적합한 품질
- **Cross-pol coherence > 0.8**: 편광 정보 보존 양호
- **Phase RMSE < 0.5 rad**: 위상 복원 품질 우수

### 3. 모델 개선 방향
- **Loss 정체**: 더 복잡한 모델 아키텍처 고려
- **과적합**: 데이터 증강 또는 정규화 강화
- **메모리 부족**: 그래디언트 체크포인팅 또는 배치 크기 감소

이 가이드를 통해 SAR 초해상도 모델의 훈련 과정을 효과적으로 모니터링하고 최적화할 수 있습니다. 