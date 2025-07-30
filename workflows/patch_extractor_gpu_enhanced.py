#!/usr/bin/env python3
# patch_extractor_gpu_enhanced.py - Professional Dual-Pol SAR Patch Extractor
"""
Dual-Pol SAR Patch Extractor for CV-SAR Super Resolution
========================================================

Professional-grade Sentinel-1 SLC dual-pol (VV+VH) patch extraction pipeline.
Optimized for Korean disaster monitoring with cross-pol coherence validation.

Features:
- Dual-pol stacking: (2, H, W) complex64 arrays
- Scene/Subswath directory hierarchy
- Cross-pol coherence quality control
- Resume capability with 80% completion rule
- GPU-accelerated FFT processing
- Memory-efficient batch processing
"""

import os
import sys
import re
import json
import traceback
import time
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import weakref
import logging
from dataclasses import dataclass, asdict

import numpy as np
import torch
import concurrent.futures
from threading import Lock

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    from esa_snappy import ProductIO, GPF, HashMap 
    HAS_SNAP = True
except ImportError:
    HAS_SNAP = False
    print("Warning: SNAP not available, running in test mode")

import psutil

# ===== CONFIGURATION =====
@dataclass
class Config:
    """Configuration parameters for dual-pol patch extraction"""
    # Patch dimensions
    patch_width: int = 256
    patch_height: int = 512
    stride_x: int = 256
    stride_y: int = 512
    
    # Quality thresholds
    min_cross_pol_coherence: float = 0.01
    min_phase_variance: float = 1e-6
    scene_completion_threshold: float = 0.8  # 80% completion
    
    # Processing options
    enable_coherence_check: bool = True
    max_retries: int = 5
    batch_size: int = 5  # Conservative for dual-pol
    
    # Paths
    input_dir: Path = Path(r'D:\Sentinel-1\data\processed_1')
    output_dir: Path = Path(r'D:\Sentinel-1\data\processed_2')
    
    # Performance
    use_gpu_fft: bool = HAS_CUPY and torch.cuda.is_available()
    num_workers: Optional[int] = None  # Auto-detect
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "patch_extraction_dual_pol.log"

# Global config instance
config = Config()

# ===== LOGGING SETUP =====
def setup_logging(level: str = "INFO", log_file: str = "patch_extraction.log"):
    """Setup structured logging"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging(config.log_level, config.log_file)

# ===== PATH BUILDER =====
class PathBuilder:
    """Centralized path building for consistent naming"""
    
    def __init__(self, base_dir: Path, scene_name: str, subswath: str):
        self.base_dir = Path(base_dir)
        self.scene_name = scene_name
        self.subswath = subswath
        self.scene_dir = self.base_dir / scene_name
        self.subswath_dir = self.scene_dir / subswath
        
        # Ensure directories exist
        self.subswath_dir.mkdir(parents=True, exist_ok=True)
    
    def dual_pol_path(self, x: int, y: int) -> Path:
        """Dual-pol complex data path"""
        return self.subswath_dir / f"{self.scene_name}_dual_pol_complex_{x}_{y}.npy"
    
    def pol_order_path(self, x: int, y: int) -> Path:
        """Polarization order metadata path"""
        return self.subswath_dir / f"{self.scene_name}_pol_order_{x}_{y}.npy"
    
    def qc_path(self, x: int, y: int) -> Path:
        """Quality control metrics path"""
        return self.subswath_dir / f"{self.scene_name}_qc_{x}_{y}.json"
    
    def single_pol_path(self, x: int, y: int) -> Path:
        """Single-pol complex data path (fallback)"""
        return self.subswath_dir / f"{self.scene_name}_single_pol_complex_{x}_{y}.npy"

# ===== UTILITIES =====
def get_optimal_workers() -> int:
    """Calculate optimal worker count for I/O bound tasks"""
    cpu_count = os.cpu_count() or 1
    # I/O bound: CPU count * 2, but cap at 8 for stability
    optimal = min(cpu_count * 2, 8)
    
    # Environment variable override
    env_workers = os.environ.get('SAR_WORKERS')
    if env_workers:
        try:
            optimal = int(env_workers)
            logger.debug(f"Worker count overridden by env: {optimal}")
        except ValueError:
            logger.warning(f"Invalid SAR_WORKERS value: {env_workers}")
    
    return optimal

def parse_band_name(band_name: str) -> Optional[Tuple[str, str]]:
    """Parse band name to extract subswath and polarization"""
    # Pattern: i_IW1_VV or q_IW2_VH
    pattern = r'[iq]_(IW\d)_(V[HV])'
    match = re.match(pattern, band_name)
    if match:
        return match.group(1), match.group(2)  # subswath, polarization
    return None

# ===== SCENE COMPLETION CHECKER =====
def check_scene_completion(scene_dir: Path, expected_patches: int) -> bool:
    """
    Check if scene is ≥80% complete based on existing dual-pol files
    
    Args:
        scene_dir: Scene directory path
        expected_patches: Expected number of patches
        
    Returns:
        True if scene completion ≥ threshold
    """
    if not scene_dir.exists():
        return False
    
    # Find all dual-pol complex files using glob pattern
    dual_pol_pattern = "*_dual_pol_complex_*.npy"
    existing_files = list(scene_dir.rglob(dual_pol_pattern))
    
    completion_rate = len(existing_files) / expected_patches if expected_patches > 0 else 0
    
    if completion_rate >= config.scene_completion_threshold:
        logger.info(f"Scene {scene_dir.name}: {completion_rate:.1%} complete "
                   f"({len(existing_files)}/{expected_patches}) - SKIPPING")
        return True
    else:
        logger.debug(f"Scene {scene_dir.name}: {completion_rate:.1%} complete "
                    f"({len(existing_files)}/{expected_patches}) - PROCESSING")
        return False

# ===== DUAL-POL BAND DETECTION =====
def get_all_complex_band_pairs(product) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Find dual-pol complex band pairs using regex matching
    
    Returns:
        Dict with polarization info, e.g.:
        {'VV': {'i_band': 'i_IW1_VV', 'q_band': 'q_IW1_VV', 'subswath': 'IW1'},
         'VH': {'i_band': 'i_IW1_VH', 'q_band': 'q_IW1_VH', 'subswath': 'IW1'}}
    """
    if not HAS_SNAP:
        # Mock for testing
        return {
            'VV': {'i_band': 'i_IW1_VV', 'q_band': 'q_IW1_VV', 'subswath': 'IW1'},
            'VH': {'i_band': 'i_IW1_VH', 'q_band': 'q_IW1_VH', 'subswath': 'IW1'}
        }
    
    band_names = [band.getName() for band in product.getBands()]
    logger.debug(f"Available bands: {[b for b in band_names if 'i_' in b or 'q_' in b]}")
    
    # Group by subswath and polarization
    i_bands = {}  # (subswath, pol) -> band_name
    q_bands = {}
    
    for band_name in band_names:
        parsed = parse_band_name(band_name)
        if parsed:
            subswath, pol = parsed
            if band_name.startswith('i_'):
                i_bands[(subswath, pol)] = band_name
            elif band_name.startswith('q_'):
                q_bands[(subswath, pol)] = band_name
    
    # Find matching I/Q pairs, prioritize IW1 → IW2 → IW3
    for subswath in ['IW1', 'IW2', 'IW3']:
        pairs = {}
        
        for pol in ['VV', 'VH']:
            key = (subswath, pol)
            if key in i_bands and key in q_bands:
                pairs[pol] = {
                    'i_band': i_bands[key],
                    'q_band': q_bands[key],
                    'subswath': subswath
                }
        
        if pairs:
            if len(pairs) == 2:
                logger.info(f" Dual-pol found: {subswath} (VV+VH)")
            else:
                pol_name = list(pairs.keys())[0]
                logger.warning(f" Single-pol only: {subswath} ({pol_name})")
            return pairs
    
    logger.error("❌ No complex band pairs found")
    return None

# ===== QUALITY METRICS =====
def calculate_cross_pol_coherence(vv_data: np.ndarray, vh_data: np.ndarray) -> float:
    """Calculate cross-pol coherence: |⟨VV * conj(VH)⟩| / sqrt(⟨|VV|²⟩ * ⟨|VH|²⟩)"""
    try:
        cross_corr = np.abs(np.mean(vv_data * np.conj(vh_data)))
        vv_power = np.mean(np.abs(vv_data)**2)
        vh_power = np.mean(np.abs(vh_data)**2)
        
        if vv_power > 1e-10 and vh_power > 1e-10:
            coherence = cross_corr / np.sqrt(vv_power * vh_power)
            return float(coherence)
        return 0.0
    except Exception as e:
        logger.debug(f"Cross-pol coherence calculation failed: {e}")
        return 0.0

def check_phase_variance(complex_data: np.ndarray) -> float:
    """Check phase variance for quality assessment"""
    try:
        phase = np.angle(complex_data)
        return float(np.var(phase))
    except Exception:
        return 0.0

# ===== GPU ACCELERATION =====
class GPUAccelerator:
    """GPU acceleration utilities"""
    
    @staticmethod
    def fft2_gpu(array: np.ndarray):
        """GPU-accelerated 2D FFT"""
        if config.use_gpu_fft and HAS_CUPY:
            gpu_array = cp.asarray(array)
            return cp.fft.fft2(gpu_array)
        elif torch.cuda.is_available():
            gpu_tensor = torch.from_numpy(array).cuda()
            return torch.fft.fft2(gpu_tensor)
        return np.fft.fft2(array)
    
    @staticmethod
    def to_cpu(array):
        """Convert GPU array to CPU"""
        if hasattr(array, 'get'):  # CuPy array
            return array.get()
        elif hasattr(array, 'cpu'):  # PyTorch tensor
            return array.cpu().numpy()
        return array

# ===== MEMORY MANAGEMENT =====
class MemoryManager:
    """Memory management utilities"""
    _products = []
    
    @staticmethod
    def register_product(prod):
        MemoryManager._products.append(prod)
    
    @staticmethod
    def safe_dispose(prod):
        try:
            if prod and hasattr(prod, 'dispose'):
                prod.dispose()
        except Exception as e:
            logger.debug(f"Product disposal warning: {e}")
    
    @staticmethod
    def cleanup_products():
        for prod in MemoryManager._products:
            MemoryManager.safe_dispose(prod)
        MemoryManager._products.clear()
        gc.collect()
        
        # JVM GC
        try:
            import java.lang.System as System
            System.gc()
        except Exception:
            pass
    
    @staticmethod
    def log_memory_usage(stage: str):
        memory = psutil.virtual_memory()
        logger.debug(f"{stage} - Memory: {memory.percent:.1f}%")

# ===== JVM HEAP MONITORING =====
def log_jvm_heap():
    """Log JVM heap memory usage"""
    try:
        import jpy
        Runtime = jpy.get_type('java.lang.Runtime')
        rt = Runtime.getRuntime()
        used_mb = (rt.totalMemory() - rt.freeMemory()) / 1e6
        total_mb = rt.totalMemory() / 1e6
        logger.debug(f"JVM heap: {used_mb:.1f}/{total_mb:.1f} MB ({used_mb/total_mb*100:.1f}%)")
    except Exception as e:
        logger.debug(f"JVM heap monitoring failed: {e}")

# ===== PATCH PROCESSING =====
@dataclass
class ProcessResult:
    """Standardized processing result"""
    success: bool
    position: Tuple[int, int]
    polarizations: List[str]
    shape: Optional[Tuple[int, ...]] = None
    cross_pol_coherence: float = 0.0
    extract_time: float = 0.0
    coherence_time: float = 0.0
    save_time: float = 0.0
    total_time: float = 0.0
    quality_metrics: Dict[str, float] = None
    retry_count: int = 0
    error_message: str = ""
    
    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = {}

def process_single_patch_with_retry(args, max_retries: int = None) -> ProcessResult:
    """
    Process single dual-pol patch with retry logic
    
    Args:
        args: (x, y, complex_pairs, product, path_builder)
        max_retries: Maximum retry attempts
        
    Returns:
        ProcessResult with success status and metrics
    """
    if max_retries is None:
        max_retries = config.max_retries
        
    x, y, complex_pairs, product, path_builder = args
    
    for retry_count in range(max_retries):
        try:
            start_time = time.time()
            
            # Pre-allocate buffers for memory efficiency
            patch_size = config.patch_width * config.patch_height
            buf_i = np.empty(patch_size, dtype=np.float32)
            buf_q = np.empty(patch_size, dtype=np.float32)
            
            # Collect dual-pol data
            complex_stack = []
            pol_list = []
            quality_metrics = {}
            
            extract_start = time.time()
            
            for pol in ['VV', 'VH']:
                if pol not in complex_pairs:
                    continue
                    
                pair_info = complex_pairs[pol]
                
                if HAS_SNAP:
                    i_band = product.getBand(pair_info['i_band'])
                    q_band = product.getBand(pair_info['q_band'])
                    
                    if i_band is None or q_band is None:
                        logger.debug(f"Band missing: {pol} at ({x}, {y})")
                        continue
                    
                    # Read pixels with pre-allocated buffers
                    i_band.readPixels(x, y, config.patch_width, config.patch_height, buf_i)
                    q_band.readPixels(x, y, config.patch_width, config.patch_height, buf_q)
                else:
                    # Mock data for testing
                    buf_i[:] = np.random.randn(patch_size).astype(np.float32)
                    buf_q[:] = np.random.randn(patch_size).astype(np.float32)
                
                # Reshape and create complex data
                i_data = buf_i.reshape(config.patch_height, config.patch_width)
                q_data = buf_q.reshape(config.patch_height, config.patch_width)
                complex_pol = (i_data + 1j * q_data).astype(np.complex64)
                
                # Optional phase variance check
                if config.enable_coherence_check:
                    phase_var = check_phase_variance(complex_pol)
                    quality_metrics[f'{pol}_phase_variance'] = phase_var
                    
                    if phase_var < config.min_phase_variance:
                        logger.debug(f"{pol} low phase variance: {phase_var:.2e}")
                
                complex_stack.append(complex_pol)
                pol_list.append(pol)
            
            extract_time = time.time() - extract_start
            
            if not complex_stack:
                raise ValueError("All polarization data extraction failed")
            
            # Stack creation
            stacked_data = np.stack(complex_stack, axis=0)
            
            # Cross-pol coherence check
            coherence_start = time.time()
            coherence_score = 1.0
            
            if len(complex_stack) == 2 and config.enable_coherence_check:
                vv_data, vh_data = complex_stack[0], complex_stack[1]
                coherence_score = calculate_cross_pol_coherence(vv_data, vh_data)
                quality_metrics['cross_pol_coherence'] = coherence_score
                
                if coherence_score < config.min_cross_pol_coherence:
                    if retry_count == 0:  # Only retry on first attempt
                        logger.debug(f"Low cross-pol coherence: {coherence_score:.3f}, retrying...")
                        raise ValueError(f"Cross-pol coherence below threshold: {coherence_score:.3f}")
                    else:
                        logger.warning(f"Persistent low coherence, skipping: {coherence_score:.3f}")
                        return ProcessResult(
                            success=False,
                            position=(x, y),
                            polarizations=pol_list,
                            cross_pol_coherence=coherence_score,
                            quality_metrics=quality_metrics,
                            retry_count=retry_count,
                            error_message=f"Cross-pol coherence below threshold: {coherence_score:.3f}"
                        )
            
            coherence_time = time.time() - coherence_start
            
            # File saving
            save_start = time.time()
            
            # Choose appropriate file paths
            if len(pol_list) == 2:
                data_path = path_builder.dual_pol_path(x, y)
            else:
                data_path = path_builder.single_pol_path(x, y)
            
            pol_order_path = path_builder.pol_order_path(x, y)
            qc_path = path_builder.qc_path(x, y)
            
            # Save data files
            np.save(data_path, stacked_data)
            np.save(pol_order_path, np.array(pol_list))
            
            # Save quality metrics
            qc_data = {
                'cross_pol_coherence': coherence_score,
                'polarizations': pol_list,
                'shape': list(stacked_data.shape),
                'quality_metrics': quality_metrics,
                'processing_time': time.time() - start_time
            }
            
            with open(qc_path, 'w') as f:
                json.dump(qc_data, f, indent=2)
            
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            
            logger.debug(f"Patch ({x},{y}) saved: {data_path.name}")
            
            return ProcessResult(
                success=True,
                position=(x, y),
                polarizations=pol_list,
                shape=stacked_data.shape,
                cross_pol_coherence=coherence_score,
                extract_time=extract_time,
                coherence_time=coherence_time,
                save_time=save_time,
                total_time=total_time,
                quality_metrics=quality_metrics,
                retry_count=retry_count
            )
            
        except Exception as e:
            if retry_count < max_retries - 1:
                wait_time = 2 ** retry_count
                logger.debug(f"Patch ({x},{y}) failed (retry {retry_count+1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Patch ({x},{y}) final failure: {e}")
                return ProcessResult(
                    success=False,
                    position=(x, y),
                    polarizations=[],
                    retry_count=retry_count,
                    error_message=str(e)
                )
    
    return ProcessResult(
        success=False,
        position=(x, y),
        polarizations=[],
        error_message="Unexpected error in retry loop"
    )

# ===== MAIN EXTRACTION PIPELINE =====
def extract_dual_pol_patches(dim_path: Path) -> int:
    """
    Main dual-pol patch extraction pipeline
    
    Args:
        dim_path: Path to .dim file
        
    Returns:
        Number of successfully processed patches
    """
    logger.info(f"Processing: {dim_path.name}")
    
    MemoryManager.log_memory_usage("Start")
    
    product = None
    successful_patches = 0
    
    try:
        # Load product
        if HAS_SNAP:
            product = ProductIO.readProduct(str(dim_path))
            if product is None:
                logger.error("Failed to load product")
                return 0
            MemoryManager.register_product(product)
            
            width = product.getSceneRasterWidth()
            height = product.getSceneRasterHeight()
        else:
            # Mock dimensions for testing
            width, height = 25000, 16000
        
        logger.info(f"Image dimensions: {width} x {height}")
        
        # Find complex band pairs
        complex_pairs = get_all_complex_band_pairs(product)
        if complex_pairs is None:
            return 0
        
        # Setup paths
        scene_name = dim_path.stem
        subswath = list(complex_pairs.values())[0]['subswath']
        path_builder = PathBuilder(config.output_dir, scene_name, subswath)
        
        # Calculate expected patches
        expected_patches_x = (width - config.patch_width) // config.stride_x + 1
        expected_patches_y = (height - config.patch_height) // config.stride_y + 1
        total_expected = expected_patches_x * expected_patches_y
        
        logger.info(f"Expected patches: {total_expected}")
        
        # Scene completion check
        if check_scene_completion(path_builder.scene_dir, total_expected):
            return total_expected  # Assume all completed
        
        # Processing statistics
        failed_patches = 0
        retried_patches = 0
        low_coherence_skips = 0
        
        # Batch processing
        batch_size = config.batch_size
        num_workers = config.num_workers or get_optimal_workers()
        
        logger.info(f"Processing with {num_workers} workers, batch size {batch_size}")
        
        for batch_start in range(0, total_expected, batch_size):
            batch_end = min(batch_start + batch_size, total_expected)
            batch_args = []
            
            log_jvm_heap()
            
            # Prepare batch
            for patch_idx in range(batch_start, batch_end):
                y_idx = patch_idx // expected_patches_x
                x_idx = patch_idx % expected_patches_x
                
                x = x_idx * config.stride_x
                y = y_idx * config.stride_y
                
                if x + config.patch_width > width or y + config.patch_height > height:
                    continue
                
                # Check if already exists
                data_path = path_builder.dual_pol_path(x, y)
                if not data_path.exists():
                    data_path = path_builder.single_pol_path(x, y)
                
                if data_path.exists():
                    logger.debug(f"Skipping existing: {data_path.name}")
                    successful_patches += 1
                    continue
                
                batch_args.append((x, y, complex_pairs, product, path_builder))
            
            # Process batch
            if batch_args:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    batch_results = list(executor.map(process_single_patch_with_retry, batch_args))
                
                # Collect results
                for result in batch_results:
                    if result.success:
                        successful_patches += 1
                        if result.retry_count > 0:
                            retried_patches += 1
                    else:
                        failed_patches += 1
                        if "coherence" in result.error_message.lower():
                            low_coherence_skips += 1
            
            log_jvm_heap()
        
        # Final statistics
        elapsed = time.time()
        rate = successful_patches / elapsed if elapsed > 0 else 0
        
        logger.info(f"Scene {scene_name} completed:")
        logger.info(f"   Success: {successful_patches}")
        logger.info(f"   Failed: {failed_patches}")
        logger.info(f"   Retried: {retried_patches}")
        logger.info(f"   Low coherence skips: {low_coherence_skips}")
        logger.info(f"   Output: {path_builder.scene_dir}")
        
        return successful_patches
        
    except Exception as e:
        logger.error(f"Scene processing error: {e}")
        logger.debug(traceback.format_exc())
        return 0
        
    finally:
        MemoryManager.cleanup_products()
        MemoryManager.log_memory_usage("End")

# ===== CLI AND MAIN =====
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Dual-Pol SAR Patch Extractor for CV-SAR Super Resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python patch_extractor_gpu_enhanced.py
  
  # Custom parameters
  python patch_extractor_gpu_enhanced.py --batch-size 3 --workers 4 --log-level DEBUG
  
  # Quick processing (skip coherence checks)
  python patch_extractor_gpu_enhanced.py --no-coherence-check
        """
    )
    
    parser.add_argument('--input-dir', type=Path, default=config.input_dir,
                        help='Input directory with .dim files')
    parser.add_argument('--output-dir', type=Path, default=config.output_dir,
                        help='Output directory for patches')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=config.num_workers,
                        help='Number of worker threads')
    parser.add_argument('--coherence-threshold', type=float, 
                        default=config.min_cross_pol_coherence,
                        help='Cross-pol coherence threshold')
    parser.add_argument('--no-coherence-check', action='store_true',
                        help='Disable coherence checking for faster processing')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default=config.log_level, help='Logging level')
    parser.add_argument('--max-files', type=int, help='Limit number of files to process')
    
    return parser

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Update config from args
    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.num_workers = args.workers
    config.min_cross_pol_coherence = args.coherence_threshold
    config.enable_coherence_check = not args.no_coherence_check
    config.log_level = args.log_level
    
    # Setup logging with new level
    global logger
    logger = setup_logging(config.log_level, config.log_file)
    
    # Banner
    logger.info("=" * 80)
    logger.info(" Dual-Pol SAR Patch Extractor v2.0 ")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"   Input: {config.input_dir}")
    logger.info(f"   Output: {config.output_dir}")
    logger.info(f"   Patch size: {config.patch_width}x{config.patch_height}")
    logger.info(f"   Stride: {config.stride_x}x{config.stride_y}")
    logger.info(f"   Coherence check: {'ON' if config.enable_coherence_check else 'OFF'}")
    logger.info(f"   Coherence threshold: {config.min_cross_pol_coherence}")
    logger.info(f"   Workers: {config.num_workers or get_optimal_workers()}")
    logger.info(f"   Batch size: {config.batch_size}")
    logger.info(f"   GPU FFT: {'ON' if config.use_gpu_fft else 'OFF'}")
    logger.info("=" * 80)
    
    # Check directories
    if not config.input_dir.exists():
        logger.error(f"Input directory not found: {config.input_dir}")
        return 1
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find .dim files
    dim_files = list(config.input_dir.glob("*.dim"))
    if args.max_files:
        dim_files = dim_files[:args.max_files]
    
    logger.info(f"Found {len(dim_files)} .dim files")
    
    if not dim_files:
        logger.warning("No .dim files found")
        return 0
    
    # Process files
    total_patches = 0
    total_start = time.time()
    
    for i, dim_file in enumerate(dim_files, 1):
        logger.info(f"\n[{i}/{len(dim_files)}] Processing: {dim_file.name}")
        
        try:
            patches = extract_dual_pol_patches(dim_file)
            total_patches += patches
        except Exception as e:
            logger.error(f"Failed to process {dim_file.name}: {e}")
            continue
    
    # Final summary
    total_elapsed = time.time() - total_start
    avg_rate = total_patches / total_elapsed if total_elapsed > 0 else 0
    
    logger.info("\n" + "=" * 80)
    logger.info(" Processing Complete! ")
    logger.info(f" Total patches: {total_patches}")
    logger.info(f" Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f" Average rate: {avg_rate:.2f} patches/second")
    logger.info(f" Output directory: {config.output_dir}")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1) 