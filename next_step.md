# Sentinel-1 SAR-SR Refactor – Next Steps

본 문서는 2025-08-02 기준 코드베이스 상태를 분석하여, 추가로 수행해야 할 **우선 작업**(Blocking)과 **개선 작업**(Nice-to-Have)을 일괄 정리한 로드맵입니다. CI/학습 중단을 최소화하기 위해 **Stage** 단위로 나누어 진행합니다.

---
## Stage 0 · 현재 상태 스냅샷
| 영역 | 현 상태 |
|------|---------|
| 데이터셋 | `np.random.shuffle` 사용, 재현성 X |
| 모델 | `ACSwinUNetPP` 4-채널 지원, Swin Shift OK, Polar SE 미구현 |
| 손실·메트릭 | CPIF·RMSE 포함, CPIF weight 고정(0.1) |
| speed_utils | `get_optimal_workers()` 상한 8 core, profiler path 개선 |
| 로깅 | PSNR/RMSE/CPIF 스칼라, 이미지 로그 아직 대용량 |

---
## Stage 1 🔒 Blocking Fixes (머지 전 필수)
| ID | 항목 | 변경 파일 | 작업 요약 |
|----|------|-----------|-----------|
| A1 | **데이터 분할 재현성** | `train.py` (Dataset class) | ① `seed: int` 인자 추가<br>② `rng = np.random.default_rng(seed)` → `rng.shuffle(...)`<br>③ 선택적 manifest 저장 `splits/train.txt` 등 |
| A2 | **CPIF λ 스케줄** | `model/utils.py`, `train.py` | `sr_loss(..., epoch, tau)` 파라미터 추가 & λ 지수감쇠 적용<br>TensorBoard에 `lambda_cpif` 로깅 |
| A3 | **Shape-guard SyntaxError** | (이미 수정됨) | 유지보수 확인 |

### Acceptance
```bash
pytest tests/test_deterministic_split.py  # split hash check
python train.py --model-type swin --dry-run  # λ 표시
```

---
## Stage 2 🔒 Model 안정화
| ID | 항목 | 변경 파일 | 작업 요약 |
|----|------|-----------|-----------|
| B1 | **PolarSE 잔차 융합** | `model/ac_swin_unet_pp.py` | ① `class PolarSE` 구현(4-채널 SE)<br>② Forward 마지막 `out = PolarSE()(out)` 후 residual 합산 |
| B2 | **PerceptualLoss 옵션화** | `train.py` | CLI `--no-perceptual` 처리 완료(코드 일부 반영, 추가 테스트 필요) |

### Acceptance
* dummy forward ↔ loss 계산 OK (Perceptual On/Off)

---
## Stage 3 ⚙️ 성능·로깅 최적화
| ID | 항목 | 변경 파일 | 작업 요약 |
|----|------|-----------|-----------|
| C1 | **TensorBoard 이미지 경량화** | `train.py` | ① `torchvision.utils.make_grid` + `add_images` 1 그리드<br>② Epoch 0 1회만 로깅 |
| C2 | **auto_adjust_batch_size 개선** | `model/speed_utils.py` | 점진적 factor [2,1.5] 탐색 루프 |
| C3 | **lr/hr 크기 파라미터 명시화** | `train.py` | `lr_size_hw` 명명 &全파일 H,W 순서 통일 |

---
## Stage 4 🌱 연구용 Experiments (선택)
1. 복소 Softmax 대안 검토(phase 보존) – `ComplexWindowAttn`
2. SE vs. Scalar residual ablation → config flag `--residual-mode {scalar,se}`

---
## 일정 & 브랜치 전략
1. `feature/reproducible_split` → Stage1 A1 머지
2. `feature/cpif_schedule`   → Stage1 A2 머지 후 Tag `v0.9.0`
3. `feature/polarSE`          → Stage2 병합, `v1.0.0-rc`

각 단계마다 **pytest + dry-run + static-package** 를 통과하고, 모델 학습 smoke test(5 epoch) 수행 후 메인에 병합합니다.

---
## 참고 커맨드
```bash
# Stage 1 패치 후 테스트
pytest -q
python train.py --model-type swin --dry-run

# Stage 2 SE-block 온/오프 학습 예시
python train.py --model-type swin --residual-mode se --no-perceptual \
               --auto-workers --batch-size-auto
```

---
> 작성: 2025-08-02  (RTX 4070 Ti, CUDA 12.9, PyTorch 2.1)
