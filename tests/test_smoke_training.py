#!/usr/bin/env python3
import os
import torch

from model.train import train_model


def test_smoke_training_tiny(tmp_path):
    """Run a tiny end-to-end training for <=50 steps to ensure the loop works.
    Uses synthetic data via empty directory (dataset will synthesize).
    """
    save_path = tmp_path / 'tiny_model.pth'
    model, history = train_model(
        data_dir=str(tmp_path),
        model_save_path=str(save_path),
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        early_stop_patience=2,
        early_stop_threshold=1e-4,
        enable_tensorboard=False,
        enable_perceptual=False,
        use_cache=False,
        gpu_degrade=False,
        num_workers=0,
        max_samples=None,
        resume_checkpoint_path=None,
        rebuild_lr_cache=False,
        psnr_ref='fixed',
        amp='off',
        lr_mode='complex_lp',
        lp_kind='gaussian',
        lp_sigma=1.0,
        lp_size=7,
        lp_beta=12.0,
        sr_scale=4,
        enl=None,
        noise_std=None,
        # Physical loss weights (small non-zero to exercise code paths)
        w_mag=1.0,
        w_phase=1.0,
        w_coh=0.1,
        w_spec=0.0,
        w_dc=0.0,
        spec_cutoff=0.45,
        phase_window=7,
        norm='gn',
        norm_groups=8,
        attn_mag_renorm='on',
        attn_temp=1.0,
        eval_window=7,
        profile_steps=0,
    )
    assert save_path.exists()
    # Ensure history has entries
    assert len(history['train_loss']) >= 1


