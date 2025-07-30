# 📡 Sentinel-1 SAR Dual-Pol 데이터 프로파일

## 📊 데이터 처리 개요

### 처리 통계 (2025-07-29 완료)
- **총 처리 패치**: 152,176개
- **총 처리 시간**: 3,829.4분 (약 63.8시간)
- **평균 처리 속도**: 0.66 패치/초
- **처리 성공률**: 거의 100% (중복 파일 스킵 포함)

### 데이터 소스
- **위성**: Sentinel-1A
- **센서 모드**: Interferometric Wide Swath (IW)
- **제품 타입**: Single Look Complex (SLC)
- **편광**: VV + VH (Dual-polarization)
- **취득 기간**: 2020년 7월 (2020-07-02 ~ 2020-08-02)

## 🔄 데이터 처리 파이프라인

### 1단계: SNAP 전처리 (graph1.xml)

#### 처리 체인
```
Raw SLC Data → Apply-Orbit-File → Calibration → TOPSAR-Split → Processed .dim
```

#### 세부 처리 단계
1. **Read**: 원본 .SAFE.zip 파일 로드
   - 밴드: `i_IW1_VH, q_IW1_VH, i_IW1_VV, q_IW1_VV, i_IW2_VH, q_IW2_VH, i_IW2_VV, q_IW2_VV, i_IW3_VH, q_IW3_VH, i_IW3_VV, q_IW3_VV`
   - 픽셀 영역: `0,0,68956,15110`

2. **Apply-Orbit-File**: 정밀 궤도 파일 적용
   - 궤도 타입: Sentinel Precise (Auto Download)
   - 다항식 차수: 3

3. **Calibration**: 복소수 보정
   - 출력 형태: Complex
   - 선택된 편광: VH, VV
   - Sigma 밴드 출력: true

4. **TOPSAR-Split**: 서브스와스 분할
   - 서브스와스: IW1, IW2, IW3 (개별 처리)
   - 편광: VH, VV
   - Burst 범위: 1-10

5. **Write**: BEAM-DIMAP 형식으로 저장
   - 출력 경로: `D:\Sentinel-1\data\processed_1\*.dim`

### 2단계: Dual-Pol 패치 추출 (patch_extractor_gpu_enhanced.py)

#### 패치 설정
- **패치 크기**: 256×512 픽셀 (Width×Height)
- **스트라이드**: 256×512 (겹침 없음)
- **데이터 타입**: numpy.complex64
- **편광 조합**: VV + VH 스택

#### 처리 변환
1. **복소수 밴드 감지**: 정규식 기반 I/Q 밴드 매칭
2. **직접 픽셀 읽기**: SNAP Subset 대신 직접 readPixels() 사용
3. **복소수 결합**: `I + 1j * Q → complex64`
4. **Dual-pol 스택**: `np.stack([VV, VH], axis=0)` → (2, H, W)
5. **품질 검증**: Cross-pol coherence 계산 (임계값: 0.01)

## 📁 데이터 저장 구조

### 디렉터리 계층
```
D:\Sentinel-1\data\processed_2\
├── S1A_IW_SLC__1SDV_20200702T093116_20200702T093146_033274_03DAEF_84AF_Orb_Cal_IW1/
│   └── IW1/
│       ├── S1A_IW_SLC__1SDV_...._dual_pol_complex_{x}_{y}.npy    # 복소수 데이터
│       ├── S1A_IW_SLC__1SDV_...._pol_order_{x}_{y}.npy          # 편광 순서
│       └── S1A_IW_SLC__1SDV_...._qc_{x}_{y}.json               # 품질 메트릭
├── S1A_IW_SLC__1SDV_20200702T093116_20200702T093146_033274_03DAEF_84AF_Orb_Cal_IW2/
│   └── IW2/
├── S1A_IW_SLC__1SDV_20200702T093116_20200702T093146_033274_03DAEF_84AF_Orb_Cal_IW3/
│   └── IW3/
└── [총 22개 Scene, 각각 1-3개 Subswath]
```

### 파일 타입별 상세

#### 1. Dual-pol 복소수 데이터 파일
- **파일명**: `*_dual_pol_complex_{x}_{y}.npy`
- **형태**: `(2, 512, 256)` numpy.complex64
- **크기**: ~2.0MB per patch
- **채널**: `[0] = VV, [1] = VH`
- **좌표**: `{x}, {y}` = 패치 시작 위치 (픽셀 단위)

#### 2. 편광 순서 메타데이터
- **파일명**: `*_pol_order_{x}_{y}.npy`
- **내용**: `['VV', 'VH']` string array
- **크기**: 144B
- **용도**: 채널 순서 보장

#### 3. 품질 관리 메트릭
- **파일명**: `*_qc_{x}_{y}.json`
- **크기**: ~200B
- **내용**:
```json
{
  "cross_pol_coherence": 1.0,
  "polarizations": ["VV", "VH"],
  "shape": [2, 512, 256],
  "quality_metrics": {},
  "processing_time": 0.021565675735473633
}
```

## 🔢 데이터 규모 및 통계

### Scene별 분포
- **총 Scene**: 22개
- **Subswath별**: IW1, IW2, IW3 각각 처리
- **일부 Scene**: IW3 누락 (취득 범위에 따라)

### 파일 크기 분석
- **평균 패치당 크기**: 2.0MB (dual_pol_complex) + 144B (pol_order) + 200B (qc)
- **총 데이터 볼륨**: ~304GB (152,176 × 2.0MB)
- **메타데이터**: ~52MB (pol_order + qc files)

### 공간 해상도
- **픽셀 간격**: ~2.3m × 14.1m (Range × Azimuth, 대략)
- **패치 실제 크기**: ~590m × 7.2km
- **Scene 크기**: ~160km × 250km (전형적인 IW 모드)

## 📋 데이터 특성

### 복소수 SAR 데이터
- **실수부**: In-phase (I) 성분
- **허수부**: Quadrature (Q) 성분
- **물리적 의미**: 전자기파 후방산란 진폭 및 위상
- **활용**: 간섭계, 편광분석, 변화탐지

### 편광 특성
- **VV (Vertical-Vertical)**: 수직 송신, 수직 수신
  - 용도: 지형, 도시지역, 산림 구조 분석
- **VH (Vertical-Horizontal)**: 수직 송신, 수평 수신
  - 용도: 식생, 해상 거칠기, 홍수 탐지



## 🎯 활용 목적 및 응용

### CV-SAR Super Resolution
- **입력 형태**: (2, 512, 256) dual-pol complex
- **목표**: 고해상도 SAR 영상 복원
- **장점**: Cross-pol 정보 활용으로 세밀한 지상 특성 보존



## 📈 데이터 품질 평가

### 처리 성공률
- **전체 성공**: ~100% (로그 기준)
- **재시도 발생**: 최소한 (안정적인 처리)
- **Low coherence skip**: 매우 적음 (0.01 임계값)

### 데이터 완전성
- **Scene 완료도**: 80% 규칙 적용
- **누락 패치**: 경계 지역의 일부 패치만
- **데이터 일관성**: 모든 패치 동일한 형태/크기

### 메모리 효율성
- **Complex64 사용**: 메모리 최적화
- **스택 구조**: PyTorch 호환성
- **압축되지 않은 저장**: 빠른 I/O 성능

