#!/usr/bin/env python3
# patch_extractor_gpu_enhanced.py - GPU 가속 및 재시도 로직 포함
import os
import sys
import traceback
import time
import gc
import numpy as np
import torch
import concurrent.futures
from threading import Lock
from esa_snappy import ProductIO, GPF, HashMap
import logging
from typing import Dict, List, Optional, Tuple, Any
import weakref
import psutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('patch_extraction_gpu_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU 가속 옵션
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy 사용 가능 - GPU FFT 가속 활성화")
except ImportError:
    HAS_CUPY = False
    cp = None
    logger.info("CuPy 없음 - CPU FFT 사용")



# CV-SAR SR 생산 최적화 설정
PATCH_W, PATCH_H = 256, 512
STRIDE_X, STRIDE_Y = 256, 512
MAX_FILES = None  # ✅ 전체 파일 처리로 수정
ENABLE_SUBAPERTURE = True
SUBAPERTURE_VIEWS = 5
ENABLE_DUAL_POL = True
LAZY_SUBAPERTURE = True
MAX_RETRIES = 3  # ✅ 실패 패치 재시도 횟수
USE_GPU_FFT = HAS_CUPY and torch.cuda.is_available()  # ✅ GPU FFT 사용 여부

# 품질 검증 임계값
MIN_CROSS_POL_COHERENCE = 0.95
MIN_ENERGY_RATIO = 0.95
MIN_PHASE_VARIANCE = 1e-6

# 경로 설정
IN_DIR = r'D:\Sentinel-1\data\processed_1'
OUT_DIR = r'D:\Sentinel-1\data\processed_2_gpu_enhanced'
os.makedirs(OUT_DIR, exist_ok=True)

# 전역 메모리 관리
_memory_lock = Lock()
_active_products = weakref.WeakSet()

class GPUAccelerator:
    """GPU 가속 헬퍼 클래스"""
    
    @staticmethod
    def to_gpu(array: np.ndarray):
        """NumPy 배열을 GPU로 전송"""
        if USE_GPU_FFT and HAS_CUPY:
            return cp.asarray(array)
        elif torch.cuda.is_available():
            return torch.from_numpy(array).cuda()
        return array
    
    @staticmethod
    def to_cpu(array):
        """GPU 배열을 CPU로 전송"""
        if hasattr(array, 'get'):  # CuPy array
            return array.get()
        elif hasattr(array, 'cpu'):  # PyTorch tensor
            return array.cpu().numpy()
        return array
    
    @staticmethod
    def fft2_gpu(array: np.ndarray):
        """GPU 가속 2D FFT"""
        if USE_GPU_FFT and HAS_CUPY:
            gpu_array = cp.asarray(array)
            fft_result = cp.fft.fft2(gpu_array)
            return fft_result
        elif torch.cuda.is_available():
            gpu_tensor = torch.from_numpy(array).cuda()
            fft_result = torch.fft.fft2(gpu_tensor)
            return fft_result
        return np.fft.fft2(array)
    
    @staticmethod
    def ifft2_gpu(array):
        """GPU 가속 2D IFFT"""
        if USE_GPU_FFT and HAS_CUPY:
            if not isinstance(array, cp.ndarray):
                array = cp.asarray(array)
            ifft_result = cp.fft.ifft2(array)
            return ifft_result
        elif torch.cuda.is_available():
            if not isinstance(array, torch.Tensor):
                array = torch.from_numpy(array).cuda()
            ifft_result = torch.fft.ifft2(array)
            return ifft_result
        return np.fft.ifft2(array)

def enhanced_subaperture_decomposition_gpu(complex_data: np.ndarray, num_views: int = 5) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """
    GPU 가속 Subaperture 분해
    """
    try:
        h, w = complex_data.shape
        logger.debug(f"GPU Subaperture 분해: {h}x{w}, {num_views}개 시각")
        
        # 적응적 분해 계획
        aspect_ratio = h / w
        if aspect_ratio > 1.5:
            azimuth_views = max(3, int(num_views * 0.6))
            range_views = num_views - azimuth_views
        else:
            range_views = max(3, int(num_views * 0.6))
            azimuth_views = num_views - range_views
        
        # ✅ GPU 가속 FFT
        start_fft = time.time()
        fft_data = GPUAccelerator.fft2_gpu(complex_data)
        fft_time = time.time() - start_fft
        logger.debug(f"{'GPU' if USE_GPU_FFT else 'CPU'} FFT 시간: {fft_time:.3f}초")
        
        subapertures = []
        
        # 방위각 방향 분해
        azimuth_step = h // azimuth_views
        for i in range(azimuth_views):
            start_h = i * azimuth_step
            end_h = min((i + 1) * azimuth_step, h)
            
            window_length = end_h - start_h
            if window_length > 0:
                # Hamming window (GPU에서 계산)
                if USE_GPU_FFT and HAS_CUPY:
                    window = cp.hamming(window_length)[:, cp.newaxis]
                    sub_fft = cp.zeros_like(fft_data)
                else:
                    window = np.hamming(window_length)[:, np.newaxis]
                    sub_fft = np.zeros_like(GPUAccelerator.to_cpu(fft_data))
                
                if USE_GPU_FFT:
                    sub_fft[start_h:end_h, :] = fft_data[start_h:end_h, :] * window
                    subaperture = GPUAccelerator.ifft2_gpu(sub_fft)
                    subaperture_cpu = GPUAccelerator.to_cpu(subaperture).astype(np.complex64)
                else:
                    fft_cpu = GPUAccelerator.to_cpu(fft_data)
                    sub_fft[start_h:end_h, :] = fft_cpu[start_h:end_h, :] * window
                    subaperture_cpu = np.fft.ifft2(sub_fft).astype(np.complex64)
                
                subapertures.append(subaperture_cpu)
        
        # 거리 방향 분해
        range_step = w // range_views
        for i in range(range_views):
            start_w = i * range_step
            end_w = min((i + 1) * range_step, w)
            
            window_length = end_w - start_w
            if window_length > 0:
                if USE_GPU_FFT and HAS_CUPY:
                    window = cp.hamming(window_length)[cp.newaxis, :]
                    sub_fft = cp.zeros_like(fft_data)
                else:
                    window = np.hamming(window_length)[np.newaxis, :]
                    sub_fft = np.zeros_like(GPUAccelerator.to_cpu(fft_data))
                
                if USE_GPU_FFT:
                    sub_fft[:, start_w:end_w] = fft_data[:, start_w:end_w] * window
                    subaperture = GPUAccelerator.ifft2_gpu(sub_fft)
                    subaperture_cpu = GPUAccelerator.to_cpu(subaperture).astype(np.complex64)
                else:
                    fft_cpu = GPUAccelerator.to_cpu(fft_data)
                    sub_fft[:, start_w:end_w] = fft_cpu[:, start_w:end_w] * window
                    subaperture_cpu = np.fft.ifft2(sub_fft).astype(np.complex64)
                
                subapertures.append(subaperture_cpu)
        
        if not subapertures:
            return None, {}
        
        result = np.stack(subapertures, axis=0)
        
        # 정확도 검증
        from workflows.patch_extractor_production_final import validate_subaperture_accuracy
        is_valid, metrics = validate_subaperture_accuracy(complex_data, result)
        
        total_time = time.time() - start_fft
        metrics['fft_time'] = float(fft_time)
        metrics['total_time'] = float(total_time)
        metrics['gpu_used'] = USE_GPU_FFT
        
        logger.debug(f"Subaperture 검증: {metrics}")
        
        if not is_valid:
            logger.warning(f"Subaperture 품질 미달: energy_ratio={metrics.get('energy_ratio', 0):.3f}")
            return None, metrics
        
        return result, metrics
        
    except Exception as e:
        logger.warning(f"GPU Subaperture 분해 실패: {str(e)}")
        return None, {}

def process_single_patch_with_retry(args, max_retries: int = 3) -> Tuple[bool, Dict]:
    """
    ✅ 재시도 로직이 포함된 단일 패치 처리
    """
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # 기존 process_single_patch 로직 import
            from workflows.patch_extractor_production_final import (
                extract_dual_pol_complex_patch,
                calculate_enhanced_coherence_features,
                save_enhanced_patch_data,
                LazySubaperture
            )
            
            subset, complex_pairs, x, y, out_path_base = args
            
            # 1. Dual-pol 복소수 추출
            complex_data_dict, extract_time, extract_msg = extract_dual_pol_complex_patch(
                subset, complex_pairs, x, y
            )
            if complex_data_dict is None:
                raise Exception(f"복소수 추출 실패: {extract_msg}")
            
            # 2. Enhanced coherence 특징 계산
            coherence_start = time.time()
            features, quality_metrics = calculate_enhanced_coherence_features(complex_data_dict)
            if features is None:
                logger.warning(f"Patch ({x}, {y}) 품질 미달: {quality_metrics}")
                if retry_count < max_retries - 1:
                    logger.info(f"재시도 {retry_count + 1}/{max_retries}...")
                    retry_count += 1
                    time.sleep(0.5)  # 짧은 대기
                    continue
                else:
                    return False, {
                        'error': f'품질 미달 (재시도 {max_retries}회 실패): {quality_metrics}',
                        'position': (x, y),
                        'quality_metrics': quality_metrics
                    }
            
            # 3. GPU 가속 Subaperture 분해
            lazy_subaperture = None
            if ENABLE_SUBAPERTURE and complex_data_dict:
                first_pol_data = next(iter(complex_data_dict.values()))
                
                # GPU 가속 버전 사용
                class LazySubapertureGPU(LazySubaperture):
                    def compute(self):
                        if not self._is_computed:
                            logger.debug("GPU LazySubaperture 계산 실행")
                            self._computed, self._metrics = enhanced_subaperture_decomposition_gpu(
                                self.complex_data, self.num_views
                            )
                            self._is_computed = True
                        return self._computed, self._metrics
                
                lazy_subaperture = LazySubapertureGPU(first_pol_data, SUBAPERTURE_VIEWS)
            
            coherence_time = time.time() - coherence_start
            
            # 4. 데이터 패키징
            patch_data = {
                'complex_data': complex_data_dict,
                'features': features,
                'quality_metrics': quality_metrics,
                'subapertures': lazy_subaperture
            }
            
            # 5. 저장
            save_success, save_time, save_msg, save_metrics = save_enhanced_patch_data(
                patch_data, out_path_base, x, y
            )
            
            if not save_success:
                raise Exception(f"저장 실패: {save_msg}")
            
            # 성공
            all_metrics = {}
            all_metrics.update(quality_metrics)
            all_metrics.update(save_metrics)
            
            result = {
                'position': (x, y),
                'extract_time': extract_time,
                'coherence_time': coherence_time,
                'save_time': save_time,
                'total_time': extract_time + coherence_time + save_time,
                'polarizations': list(complex_data_dict.keys()),
                'save_msg': save_msg,
                'quality_metrics': all_metrics,
                'retry_count': retry_count,
                'gpu_used': USE_GPU_FFT
            }
            
            return True, result
            
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"패치 ({x}, {y}) 처리 실패 (재시도 {retry_count}/{max_retries}): {last_error}")
                time.sleep(1.0 * retry_count)  # 점진적 대기
            else:
                logger.error(f"패치 ({x}, {y}) 최종 실패: {last_error}")
                return False, {
                    'error': f'최대 재시도 횟수 초과: {last_error}',
                    'position': (x, y),
                    'retry_count': retry_count
                }
    
    return False, {'error': '예상치 못한 오류', 'position': args[2:4]}

def extract_gpu_enhanced_patches(dim_path: str) -> int:
    """
    GPU 가속 및 재시도 로직이 포함된 패치 추출
    """
    logger.info(f"GPU Enhanced 패치 추출: {os.path.basename(dim_path)}")
    logger.info(f"GPU FFT 사용: {'✅ 활성화' if USE_GPU_FFT else '❌ CPU 모드'}")
    
    # 나머지 로직은 production_final과 동일하지만 process_single_patch_with_retry 사용
    from workflows.patch_extractor_production_final import (
        MemoryMonitor, MemoryManager, get_all_complex_band_pairs,
        get_optimal_workers
    )
    
    MemoryMonitor.log_memory_usage("시작")
    
    prod = None
    patch_count = 0
    start_time = time.time()
    
    try:
        # 제품 로드
        prod = ProductIO.readProduct(dim_path)
        if prod is None:
            logger.error("제품 로드 실패")
            return 0
        
        MemoryManager.register_product(prod)
        
        width = prod.getSceneRasterWidth()
        height = prod.getSceneRasterHeight()
        logger.info(f"이미지 크기: {width} x {height}")
        
        # 복소수 밴드 쌍 찾기
        complex_pairs = get_all_complex_band_pairs(prod)
        if complex_pairs is None:
            return 0
        
        base = os.path.basename(dim_path).replace('.dim', '')
        
        # 예상 패치 수 계산
        expected_patches_x = (width - PATCH_W) // STRIDE_X + 1
        expected_patches_y = (height - PATCH_H) // STRIDE_Y + 1
        total_expected = expected_patches_x * expected_patches_y
        logger.info(f"예상 패치 수: {total_expected}")
        
        # GPU 메모리 확인
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU 메모리: {gpu_memory:.2f} GB")
        
        # 동적 워커 수
        optimal_workers = get_optimal_workers(total_expected)
        logger.info(f"최적화된 워커 수: {optimal_workers}")
        
        # 서브셋 파라미터
        subset_params_base = HashMap()
        subset_params_base.put('copyMetadata', 'false')
        
        # 통계 변수
        successful_patches = 0
        failed_patches = 0
        retried_patches = 0
        gpu_speedup_total = 0
        
        # 배치 처리
        batch_size = min(100, total_expected)
        
        for batch_start in range(0, total_expected, batch_size):
            batch_end = min(batch_start + batch_size, total_expected)
            batch_args = []
            
            # 배치 준비
            for patch_idx in range(batch_start, batch_end):
                y_idx = patch_idx // expected_patches_x
                x_idx = patch_idx % expected_patches_x
                
                x = x_idx * STRIDE_X
                y = y_idx * STRIDE_Y
                
                if x + PATCH_W > width or y + PATCH_H > height:
                    continue
                
                # 출력 경로
                out_path_base = os.path.join(OUT_DIR, f'{base}_{complex_pairs["subswath"]}')
                complex_path = f"{out_path_base}_dual_pol_complex_{x}_{y}.npy"
                
                # 이미 존재하는 파일 건너뛰기
                if os.path.exists(complex_path):
                    patch_count += 1
                    continue
                
                # 서브셋 생성
                subset_params = HashMap(subset_params_base)
                subset_params.put('pixelRegion', f'{x},{y},{PATCH_W},{PATCH_H}')
                
                subset = GPF.createProduct('Subset', subset_params, prod)
                if subset is None:
                    logger.warning(f"서브셋 생성 실패: ({x}, {y})")
                    continue
                
                MemoryManager.register_product(subset)
                batch_args.append((subset, complex_pairs, x, y, out_path_base))
            
            # 병렬 처리 (재시도 로직 포함)
            if batch_args:
                current_workers = get_optimal_workers(len(batch_args), optimal_workers)
                logger.info(f"배치 처리: {len(batch_args)}개 패치, {current_workers}개 워커")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=current_workers) as executor:
                    batch_results = list(executor.map(
                        lambda args: process_single_patch_with_retry(args, MAX_RETRIES),
                        batch_args
                    ))
                
                # 결과 처리
                for success, result in batch_results:
                    if success:
                        successful_patches += 1
                        if result.get('retry_count', 0) > 0:
                            retried_patches += 1
                        if result.get('gpu_used', False):
                            gpu_speedup_total += 1
                    else:
                        failed_patches += 1
                
                # 메모리 정리
                for subset, _, _, _, _ in batch_args:
                    MemoryManager.safe_dispose(subset)
                
                MemoryManager.cleanup_products()
            
            patch_count = successful_patches
        
        # 최종 통계
        elapsed = time.time() - start_time
        rate = successful_patches / elapsed if elapsed > 0 else 0
        
        logger.info(f"파일 처리 완료: {os.path.basename(dim_path)}")
        logger.info(f"성공: {successful_patches}개, 실패: {failed_patches}개, 재시도: {retried_patches}개")
        logger.info(f"GPU 가속 사용: {gpu_speedup_total}개 패치")
        logger.info(f"소요시간: {elapsed/60:.1f}분, 속도: {rate:.2f}패치/초")
        
        MemoryMonitor.log_memory_usage("완료")
        
        return successful_patches
        
    except Exception as e:
        logger.error(f"파일 처리 중 오류: {str(e)}")
        return 0
        
    finally:
        if prod is not None:
            MemoryManager.safe_dispose(prod)
        MemoryManager.cleanup_products()

def main():
    """GPU Enhanced 메인"""
    logger.info("=" * 90)
    logger.info("🚀 CV-SAR SR GPU Enhanced Pipeline")
    logger.info("개선사항:")
    logger.info(f"  ✅ GPU FFT 가속: {'활성화' if USE_GPU_FFT else '비활성화'}")
    logger.info(f"  ✅ 실패 패치 재시도: 최대 {MAX_RETRIES}회")
    logger.info(f"  ✅ 전체 파일 처리: MAX_FILES={MAX_FILES}")
    logger.info("=" * 90)
    
    # GPU 정보
    if torch.cuda.is_available():
        logger.info(f"GPU 장치: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.info("GPU 사용 불가 - CPU 모드")
    
    # 입력 확인
    if not os.path.exists(IN_DIR):
        logger.error(f"입력 디렉토리가 존재하지 않습니다: {IN_DIR}")
        return
    
    # .dim 파일 찾기
    dim_files = [f for f in sorted(os.listdir(IN_DIR)) if f.endswith('.dim')]
    if MAX_FILES is not None:
        dim_files = dim_files[:MAX_FILES]
    
    logger.info(f"처리할 .dim 파일 수: {len(dim_files)}")
    
    if not dim_files:
        logger.warning("처리할 .dim 파일이 없습니다.")
        return
    
    total_patches = 0
    total_start_time = time.time()
    
    # 파일 처리
    for file_idx, file in enumerate(dim_files, 1):
        logger.info(f"\n[{file_idx}/{len(dim_files)}] GPU Enhanced 처리: {file}")
        
        try:
            dim_path = os.path.join(IN_DIR, file)
            patches_created = extract_gpu_enhanced_patches(dim_path)
            total_patches += patches_created
            
        except Exception as e:
            logger.error(f"파일 처리 실패 {file}: {str(e)}")
            continue
    
    # 최종 결과
    total_elapsed = time.time() - total_start_time
    avg_rate = total_patches / total_elapsed if total_elapsed > 0 else 0
    
    logger.info("\n" + "=" * 90)
    logger.info("🎉 GPU Enhanced 처리 완료!")
    logger.info(f"총 패치: {total_patches}개")
    logger.info(f"총 시간: {total_elapsed/60:.1f}분")
    logger.info(f"평균 속도: {avg_rate:.2f}패치/초")
    
    # GPU 가속 효과
    if USE_GPU_FFT:
        logger.info("✅ GPU FFT 가속으로 성능 향상")
    
    logger.info("=" * 90)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {str(e)}")
        logger.debug(f"상세 오류: {traceback.format_exc()}") 