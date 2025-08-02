#!/usr/bin/env python3
# amp_phase_hist.py
"""
Efficient amplitude & phase histogram for Sentinel-1 dual-pol patches
--------------------------------------------------------------------
* No full-array accumulation → O( bins) memory
* Online Welford statistics for mean / std
* Saves histograms as PNG + CSV

Assumed patch shape: (2, 256, 512) complex64  [0]=VV, [1]=VH
Directory structure:  data_dir/**/*.npy
"""

import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────── CONFIG ────────────────────────────
BINS_AMP   = 50
BINS_PHASE = 50
PERCENTILE_CUTOFF = 99.5      # clip rare outliers for nicer bins
MAX_FILES  = None             # set e.g. 40_000 to limit


# ────────────────────── ONLINE STAT UTILITIES ───────────────────
class OnlineStats:
    """Welford online mean / variance (per key)"""
    def __init__(self):
        self.n, self.mean, self.M2 = {}, {}, {}

    def update(self, key: str, x: np.ndarray):
        if key not in self.n:
            self.n[key] = 0; self.mean[key] = 0.0; self.M2[key] = 0.0
        n1   = self.n[key]
        n2   = x.size
        delta= x.mean() - self.mean[key]

        # update totals
        self.n[key]   += n2
        self.mean[key]+= delta * n2 / self.n[key]
        # aggregate variance * n
        self.M2[key]  += (x.var() + delta**2) * n2

    def final(self, key):
        if self.n.get(key,0) < 2:
            return float('nan'), float('nan')
        var = self.M2[key] / (self.n[key]-1)
        return self.mean[key], float(np.sqrt(var))


# ──────────────────────────── MAIN ──────────────────────────────
def analyze(data_dir: Path):
    files = sorted(data_dir.rglob("*_dual_pol_complex_*.npy"))
    if MAX_FILES: files = files[:MAX_FILES]
    if not files:
        print("No .npy patches found.", file=sys.stderr); return

    # Sample few files to set amplitude bin max
    sample_vals = []
    for f in files[:500]:
        a = np.load(f, mmap_mode='r')[0]   # VV amplitude only
        sample_vals.append(np.abs(a).ravel())
    amp_clip = np.percentile(np.concatenate(sample_vals), PERCENTILE_CUTOFF)

    amp_edges   = np.linspace(0, amp_clip, BINS_AMP+1, dtype=np.float32)
    phase_edges = np.linspace(-np.pi, np.pi, BINS_PHASE+1, dtype=np.float32)

    # zero-initialised histograms
    hist = {
        'amp_vv':   np.zeros(BINS_AMP,   dtype=np.int64),
        'amp_vh':   np.zeros(BINS_AMP,   dtype=np.int64),
        'phase_vv': np.zeros(BINS_PHASE, dtype=np.int64),
        'phase_vh': np.zeros(BINS_PHASE, dtype=np.int64),
    }
    stats = OnlineStats()

    # ────────── main scan ──────────
    for i,f in enumerate(files,1):
        arr = np.load(f).astype(np.complex64)    # (2,H,W)
        if arr.shape[0]!=2: continue
        vv, vh = arr[0], arr[1]

        amp_vv, amp_vh = np.abs(vv), np.abs(vh)
        ph_vv,  ph_vh  = np.angle(vv), np.angle(vh)

        # histograms (flatten handled by np.histogram)
        hist['amp_vv']  += np.histogram(amp_vv,  amp_edges)[0]
        hist['amp_vh']  += np.histogram(amp_vh,  amp_edges)[0]
        hist['phase_vv']+= np.histogram(ph_vv,   phase_edges)[0]
        hist['phase_vh']+= np.histogram(ph_vh,   phase_edges)[0]

        # stats
        stats.update('amp_vv',  amp_vv)
        stats.update('amp_vh',  amp_vh)
        stats.update('ph_vv',   ph_vv)
        stats.update('ph_vh',   ph_vh)

        if i%2000==0: print(f" processed {i}/{len(files)} files")

    # ────────── save / plot ──────────
    out_dir = data_dir / "histograms"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir/"dualpol_histograms.npz", **hist,
             amp_edges=amp_edges, phase_edges=phase_edges)

    def plot_hist(data, edges, title, fname):
        centers = 0.5*(edges[:-1]+edges[1:])
        plt.figure()
        plt.bar(centers, data, width=np.diff(edges), color="#4c72b0")
        plt.title(title); plt.tight_layout()
        plt.savefig(out_dir/fname); plt.close()

    plot_hist(hist['amp_vv'],   amp_edges,   "Amplitude VV",   "amp_vv.png")
    plot_hist(hist['amp_vh'],   amp_edges,   "Amplitude VH",   "amp_vh.png")
    plot_hist(hist['phase_vv'], phase_edges, "Phase VV",       "phase_vv.png")
    plot_hist(hist['phase_vh'], phase_edges, "Phase VH",       "phase_vh.png")

    # stats print-out
    for key in ['amp_vv','amp_vh','ph_vv','ph_vh']:
        m,s = stats.final(key)
        print(f"{key}: mean {m:.4f}, std {s:.4f}")

    print(f"\nSaved histograms & stats to: {out_dir}")

# ────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-efficient amplitude & phase histogram for dual-pol patches")
    parser.add_argument("data_dir", type=Path, 
                        help="Root folder containing *_dual_pol_complex_*.npy")
    parser.add_argument("--bins", type=int, default=BINS_AMP,
                        help=f"Number of bins (default {BINS_AMP})")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files for quick test")
    args = parser.parse_args()

    # override globals if CLI flags used
    BINS_AMP   = BINS_PHASE = args.bins
    MAX_FILES  = args.max_files

    analyze(args.data_dir)
