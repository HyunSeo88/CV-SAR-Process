#!/usr/bin/env python3
# data_augmentation_coherence.py - Dual-pol coherence 보존 데이터 증강
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class CoherencePreservingAugmentation:
    """
    Dual-pol coherence를 보존하는 데이터 증강 기법
    한국 재난 (홍수/태풍) 특화
    """
    
    def __init__(self, preserve_cross_pol: bool = True):
        self.preserve_cross_pol = preserve_cross_pol
    
    def random_phase_rotation(self, dual_pol_complex: np.ndarray, max_rotation: float = np.pi/4) -> np.ndarray:
        """
        위상 회전 증강 - Cross-pol coherence 보존
        
        Args:
            dual_pol_complex: (2, H, W) complex array [VV, VH]
            max_rotation: 최대 회전 각도 (라디안)
        """
        # 동일한 랜덤 위상 회전을 VV와 VH에 적용
        rotation_angle = np.random.uniform(-max_rotation, max_rotation)
        phase_factor = np.exp(1j * rotation_angle)
        
        # 두 편광에 동일한 위상 변화 적용 (coherence 보존)
        augmented = dual_pol_complex * phase_factor
        
        # Cross-pol coherence 검증
        if self.preserve_cross_pol:
            original_coherence = self._calculate_cross_pol_coherence(dual_pol_complex)
            augmented_coherence = self._calculate_cross_pol_coherence(augmented)
            
            if abs(original_coherence - augmented_coherence) > 0.01:
                logger.warning(f"Cross-pol coherence 변화: {original_coherence:.3f} → {augmented_coherence:.3f}")
        
        return augmented
    
    def coherent_speckle_filter(self, dual_pol_complex: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Coherent speckle 필터링 - 물리적 특성 보존
        
        Args:
            dual_pol_complex: (2, H, W) complex array
            strength: 필터 강도 (0-1)
        """
        # Lee filter 기반 coherent speckle 감소
        vv, vh = dual_pol_complex[0], dual_pol_complex[1]
        
        # 3x3 윈도우 평균
        kernel = torch.ones(1, 1, 3, 3) / 9.0
        
        # Complex를 실부/허부로 분리
        vv_real = torch.from_numpy(vv.real).unsqueeze(0).unsqueeze(0).float()
        vv_imag = torch.from_numpy(vv.imag).unsqueeze(0).unsqueeze(0).float()
        vh_real = torch.from_numpy(vh.real).unsqueeze(0).unsqueeze(0).float()
        vh_imag = torch.from_numpy(vh.imag).unsqueeze(0).unsqueeze(0).float()
        
        # Coherent 필터링 (실부/허부 각각)
        vv_real_filtered = F.conv2d(vv_real, kernel, padding=1)
        vv_imag_filtered = F.conv2d(vv_imag, kernel, padding=1)
        vh_real_filtered = F.conv2d(vh_real, kernel, padding=1)
        vh_imag_filtered = F.conv2d(vh_imag, kernel, padding=1)
        
        # 원본과 혼합 (coherence 보존)
        vv_filtered = (vv_real_filtered.squeeze().numpy() + 1j * vv_imag_filtered.squeeze().numpy())
        vh_filtered = (vh_real_filtered.squeeze().numpy() + 1j * vh_imag_filtered.squeeze().numpy())
        
        vv_aug = (1 - strength) * vv + strength * vv_filtered
        vh_aug = (1 - strength) * vh + strength * vh_filtered
        
        return np.stack([vv_aug, vh_aug], axis=0)
    
    def simulate_flood_signature(self, dual_pol_complex: np.ndarray, flood_mask: np.ndarray) -> np.ndarray:
        """
        홍수 시그니처 시뮬레이션 - 서울 홍수 패턴
        
        Args:
            dual_pol_complex: (2, H, W) complex array
            flood_mask: (H, W) binary mask (1=flood, 0=land)
        """
        augmented = dual_pol_complex.copy()
        
        # 홍수 지역: Cross-pol coherence 감소
        flood_indices = flood_mask > 0.5
        
        if np.any(flood_indices):
            # VV 반사 감소 (물 표면)
            augmented[0][flood_indices] *= 0.3 + 0.2 * np.random.rand()
            
            # VH 더 큰 감소 (매끄러운 표면)
            augmented[1][flood_indices] *= 0.1 + 0.1 * np.random.rand()
            
            # 위상 무작위화 (낮은 coherence)
            random_phase = np.random.uniform(-np.pi, np.pi, size=np.sum(flood_indices))
            augmented[0][flood_indices] *= np.exp(1j * random_phase * 0.5)
            augmented[1][flood_indices] *= np.exp(1j * random_phase * 0.8)
        
        return augmented
    
    def simulate_typhoon_pattern(self, dual_pol_complex: np.ndarray, wind_direction: float, wind_speed: float) -> np.ndarray:
        """
        태풍 패턴 시뮬레이션 - 제주 태풍
        
        Args:
            dual_pol_complex: (2, H, W) complex array
            wind_direction: 바람 방향 (라디안)
            wind_speed: 바람 속도 (m/s)
        """
        h, w = dual_pol_complex.shape[1:]
        augmented = dual_pol_complex.copy()
        
        # 바람 방향에 따른 그래디언트 생성
        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # 바람 방향 투영
        wind_projection = np.cos(wind_direction) * x_grid + np.sin(wind_direction) * y_grid
        wind_projection = (wind_projection - wind_projection.min()) / (wind_projection.max() - wind_projection.min())
        
        # 풍속에 따른 변조
        modulation = 1 + 0.3 * wind_speed / 50.0 * np.sin(2 * np.pi * wind_projection * 10)
        
        # VH가 바람에 더 민감 (해상 거칠기)
        augmented[0] *= modulation
        augmented[1] *= modulation * 1.5  # VH 더 강한 변조
        
        # Cross-pol ratio 증가 (거친 해면)
        augmented[1] *= (1 + 0.2 * wind_speed / 50.0)
        
        return augmented
    
    def temporal_decorrelation(self, dual_pol_complex: np.ndarray, time_gap_days: int) -> np.ndarray:
        """
        시간적 decorrelation 시뮬레이션
        
        Args:
            dual_pol_complex: (2, H, W) complex array
            time_gap_days: 시간 간격 (일)
        """
        # 시간에 따른 coherence 감소 모델
        temporal_coherence = np.exp(-0.02 * time_gap_days)
        
        # 랜덤 위상 변화
        phase_std = np.sqrt(1 - temporal_coherence**2) * np.pi
        
        vv_phase_change = np.random.normal(0, phase_std, dual_pol_complex.shape[1:])
        vh_phase_change = np.random.normal(0, phase_std * 1.2, dual_pol_complex.shape[1:])  # VH 더 빠른 decorrelation
        
        augmented = dual_pol_complex.copy()
        augmented[0] *= np.exp(1j * vv_phase_change)
        augmented[1] *= np.exp(1j * vh_phase_change)
        
        return augmented
    
    def geometric_transformation_coherent(self, dual_pol_complex: np.ndarray, max_shift: int = 5) -> np.ndarray:
        """
        기하학적 변환 - Coherence 보존
        
        Args:
            dual_pol_complex: (2, H, W) complex array
            max_shift: 최대 이동 픽셀
        """
        # 동일한 변환을 두 편광에 적용
        shift_y = np.random.randint(-max_shift, max_shift + 1)
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        
        augmented = np.roll(dual_pol_complex, shift=(0, shift_y, shift_x), axis=(0, 1, 2))
        
        # 경계 처리
        if shift_y > 0:
            augmented[:, :shift_y, :] = 0
        elif shift_y < 0:
            augmented[:, shift_y:, :] = 0
        
        if shift_x > 0:
            augmented[:, :, :shift_x] = 0
        elif shift_x < 0:
            augmented[:, :, shift_x:] = 0
        
        return augmented
    
    def _calculate_cross_pol_coherence(self, dual_pol_complex: np.ndarray) -> float:
        """Cross-pol coherence 계산"""
        vv, vh = dual_pol_complex[0], dual_pol_complex[1]
        
        numerator = np.abs(np.mean(vv * np.conj(vh)))
        denominator = np.sqrt(np.mean(np.abs(vv)**2) * np.mean(np.abs(vh)**2))
        
        return numerator / (denominator + 1e-10)
    
    def augment_batch(self, batch: Dict[str, torch.Tensor], p: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        배치 증강 - PyTorch 통합
        
        Args:
            batch: {'dual_pol_complex': (B, 2, H, W), ...}
            p: 각 증강 적용 확률
        """
        B = batch['dual_pol_complex'].shape[0]
        augmented_batch = batch.copy()
        
        for i in range(B):
            dual_pol = batch['dual_pol_complex'][i].numpy()
            
            # 랜덤 증강 적용
            if np.random.rand() < p:
                # 위상 회전
                dual_pol = self.random_phase_rotation(dual_pol)
            
            if np.random.rand() < p * 0.5:
                # Speckle 필터
                dual_pol = self.coherent_speckle_filter(dual_pol, strength=0.2)
            
            if np.random.rand() < p * 0.3:
                # 기하학적 변환
                dual_pol = self.geometric_transformation_coherent(dual_pol, max_shift=3)
            
            if np.random.rand() < p * 0.2:
                # 시간적 decorrelation
                days = np.random.randint(1, 12)
                dual_pol = self.temporal_decorrelation(dual_pol, days)
            
            augmented_batch['dual_pol_complex'][i] = torch.from_numpy(dual_pol)
        
        return augmented_batch


class DisasterSpecificAugmentation:
    """
    한국 재난 시나리오별 특화 증강
    """
    
    @staticmethod
    def generate_flood_training_data(base_patch: np.ndarray, num_variations: int = 10) -> List[np.ndarray]:
        """
        홍수 학습 데이터 생성 - 서울/수도권 홍수 패턴
        """
        augmenter = CoherencePreservingAugmentation()
        variations = []
        
        for i in range(num_variations):
            # 랜덤 홍수 마스크 생성
            h, w = base_patch.shape[1:]
            flood_mask = np.zeros((h, w))
            
            # 하천/저지대 시뮬레이션
            flood_center_y = np.random.randint(h//4, 3*h//4)
            flood_width = np.random.randint(20, 50)
            
            for y in range(h):
                if abs(y - flood_center_y) < flood_width:
                    flood_intensity = 1 - abs(y - flood_center_y) / flood_width
                    flood_mask[y, :] = flood_intensity * (0.5 + 0.5 * np.random.rand(w))
            
            # 홍수 시그니처 적용
            flooded = augmenter.simulate_flood_signature(base_patch, flood_mask)
            
            # 추가 변형
            if i % 2 == 0:
                flooded = augmenter.temporal_decorrelation(flooded, np.random.randint(1, 6))
            
            variations.append(flooded)
        
        return variations
    
    @staticmethod
    def generate_typhoon_training_data(base_patch: np.ndarray, num_variations: int = 10) -> List[np.ndarray]:
        """
        태풍 학습 데이터 생성 - 제주/남해안 태풍 패턴
        """
        augmenter = CoherencePreservingAugmentation()
        variations = []
        
        for i in range(num_variations):
            # 태풍 파라미터
            wind_direction = np.random.uniform(0, 2*np.pi)  # 랜덤 방향
            wind_speed = np.random.uniform(15, 50)  # 15-50 m/s
            
            # 태풍 패턴 적용
            typhoon = augmenter.simulate_typhoon_pattern(base_patch, wind_direction, wind_speed)
            
            # Speckle 증가 (거친 해면)
            typhoon = augmenter.coherent_speckle_filter(typhoon, strength=-0.2)  # 음수 = speckle 증가
            
            # 시간적 변화
            if i % 3 == 0:
                typhoon = augmenter.temporal_decorrelation(typhoon, 1)  # 빠른 변화
            
            variations.append(typhoon)
        
        return variations


# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_patch = np.random.randn(2, 512, 256) + 1j * np.random.randn(2, 512, 256)
    test_patch = test_patch.astype(np.complex64)
    
    # Coherence 보존 증강
    augmenter = CoherencePreservingAugmentation(preserve_cross_pol=True)
    
    # 원본 coherence
    orig_coherence = augmenter._calculate_cross_pol_coherence(test_patch)
    print(f"원본 Cross-pol coherence: {orig_coherence:.3f}")
    
    # 위상 회전
    rotated = augmenter.random_phase_rotation(test_patch)
    rot_coherence = augmenter._calculate_cross_pol_coherence(rotated)
    print(f"회전 후 coherence: {rot_coherence:.3f} (변화: {abs(orig_coherence - rot_coherence):.3f})")
    
    # 홍수 데이터 생성
    flood_data = DisasterSpecificAugmentation.generate_flood_training_data(test_patch, 5)
    print(f"\n생성된 홍수 데이터: {len(flood_data)}개")
    
    # 태풍 데이터 생성
    typhoon_data = DisasterSpecificAugmentation.generate_typhoon_training_data(test_patch, 5)
    print(f"생성된 태풍 데이터: {len(typhoon_data)}개") 