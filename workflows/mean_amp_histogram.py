#!/usr/bin/env python3
"""
Patch-wise mean-power histogram creator
--------------------------------------

* 입력 : Sentinel-1 dual-pol complex 패치(.npy) 디렉터리
* 출력 : mean_db_histogram.png (히스토그램)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def db_histogram(data_dir: str,
                 pattern: str = '*_dual_pol_complex_*.npy',
                 max_files: int = None,
                 num_bins: int = 80,
                 save_path: str = 'mean_db_histogram.png'):
    """
    VV 평균 전력(dB)을 패치별로 계산해 히스토그램을 저장한다.
    lr_cache 하위 파일과 *_lr.npy 캐시는 자동 제외.
    """
    # ① 재귀 검색 + ② lr_cache / *_lr.npy 필터
    files = [
        f for f in Path(data_dir).rglob(pattern)
        if ('lr_cache' not in f.parts) and (not f.name.endswith('_lr.npy'))
    ]
    if max_files:
        files = files[:max_files]

    mean_db_list = []
    for f in tqdm(files, desc='scanning', unit='patch'):
        z = np.load(f, mmap_mode='r')          # (2,H,W) complex64
        vv = z[0]                              # VV 편광
        power = np.mean(np.abs(vv)**2)         # 평균 전력
        mean_db = 10 * np.log10(power + 1e-12) # dB 변환
        mean_db_list.append(mean_db)

    # 히스토그램
    plt.figure(figsize=(7, 4))
    plt.hist(mean_db_list, bins=num_bins, edgecolor='k', alpha=0.85)
    plt.xlabel('Mean VV power (dB)')
    plt.ylabel('Patch count')
    plt.title(f'Patch-wise mean power distribution  (N={len(mean_db_list)})')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Histogram saved to  {save_path}')


if __name__ == '__main__':
    db_histogram(
        data_dir=r'D:\Sentinel-1\data\processed_2\S1A_IW_SLC__1SDV_20200702T093143_20200702T093211_033274_03DAEF_148B_Orb_Cal_IW2\IW2',
        pattern='*_dual_pol_complex_*.npy',
        num_bins=80,
        max_files=None,            # 전체
        save_path='148B_IW2.png',

    )
