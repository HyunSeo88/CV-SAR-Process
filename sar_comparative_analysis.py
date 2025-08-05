#!/usr/bin/env python3
"""
SAR Image Visual Comparison
===========================

A streamlined script to visually compare super-resolved SAR images
with their ground truth counterparts.

This script loads a trained model, processes SAR patches from a directory,
generates super-resolved images, and saves side-by-side comparisons
of the amplitude images, which is standard for visual SAR analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import warnings
import pickle
import argparse
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add model directory to path to import model components
sys.path.append('./model')
try:
    from cv_unet import ComplexUNet
    from train import SARSuperResDataset
except ImportError as e:
    print(f"Error: Could not import model components. Make sure 'cv_unet.py' and 'train.py' are in the './model' directory.")
    print(f"Details: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')
plt.style.use('default')

class SARImageAnalyzer:
    """
    A tool to visually compare super-resolved vs. ground truth SAR images.
    """
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._load_model()

    def _load_model(self) -> ComplexUNet:
        """Load a trained Complex U-Net model."""
        model = ComplexUNet()
        print(f"ComplexUNet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

        if not Path(self.model_path).exists():
            print(f"ERROR: Model checkpoint not found at {self.model_path}")
            sys.exit(1)
        
        # Attempt to load the checkpoint. In PyTorch >= 2.6 the default is `weights_only=True`,
        # which can break loading of older checkpoints created with `torch.save(model_state_dict)`.
        # We first try the safe default and fall back to `weights_only=False` (unsafe) if necessary.
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        except (RuntimeError, pickle.UnpicklingError) as e:
            # Detect the specific PyTorch 2.6 "weights_only" failure mode
            err_msg = str(e)
            if "weights_only" in err_msg or "WeightsUnpickler" in err_msg or "weights only load failed" in err_msg.lower():
                print("[WARN] Retrying model load with weights_only=False (unsafe, but required for legacy checkpoints).")
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            else:
                print(f"ERROR: Failed to load model weights from {self.model_path}. Error: {e}")
                sys.exit(1)

        # Populate the network with the loaded weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict directly.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def generate_super_resolved_images(self, lr_patches: torch.Tensor) -> torch.Tensor:
        """Generate super-resolved images from low-resolution patches.

        The trained `ComplexUNet` expects 3 input channels:
            0) VV real
            1) VV imag
            2) |VH| magnitude
        However, the dataset currently returns 4 channels (VV-Re, VV-Im, VH-Re, VH-Im).
        This helper converts the 4-channel representation to the expected 3-channel
        format on-the-fly so we don't have to retrain or rewrite the dataset class.
        """
        with torch.no_grad():
            lr_patches = lr_patches.to(self.device)

            # If we get 4 channels, convert to 3-channel layout expected by the model
            if lr_patches.shape[1] == 4:
                vv_real = lr_patches[:, 0:1, :, :]
                vv_imag = lr_patches[:, 1:2, :, :]
                # Add small epsilon inside sqrt for numerical stability
                vh_mag = torch.sqrt(lr_patches[:, 2:3, :, :].pow(2) + lr_patches[:, 3:4, :, :].pow(2) + 1e-8)
                model_input = torch.cat([vv_real, vv_imag, vh_mag], dim=1)
            else:
                model_input = lr_patches  # Already in correct shape

            sr_patches = self.model(model_input)
            return sr_patches.cpu()

    def create_visual_comparison(self, sr_image: torch.Tensor, gt_image: torch.Tensor, sample_idx: int):
        """
        Creates and saves a side-by-side plot of GT and SR VV-polarized amplitude images.
        """
        # The model output and dataset provide 2-channel real/imag tensors.
        # Combine them into complex tensors to calculate amplitude.
        sr_complex_vv = torch.complex(sr_image[0], sr_image[1])
        gt_complex_vv = torch.complex(gt_image[0], gt_image[1])

        # Apply log scaling to amplitude for better visual contrast, which is standard for SAR.
        sr_amp_log = torch.log10(torch.abs(sr_complex_vv) + 1e-8).numpy()
        gt_amp_log = torch.log10(torch.abs(gt_complex_vv) + 1e-8).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Visual Comparison - Sample {sample_idx}', fontsize=16)

        # Plot Ground Truth
        im1 = axes[0].imshow(gt_amp_log, cmap='gray')
        axes[0].set_title('Ground Truth (VV Amplitude - Log Scale)')
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0])

        # Plot Super-Resolved
        im2 = axes[1].imshow(sr_amp_log, cmap='gray')
        axes[1].set_title('Super-Resolved (VV Amplitude - Log Scale)')
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = self.output_dir / f'comparison_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
    def run_analysis(self, data_dir: str, num_samples: int, batch_size: int):
        """
        Run the visual analysis by processing samples from the dataset.
        """
        print("\n[SAR ANALYSIS] Starting SAR Image Visual Comparison")
        print("=" * 60)
        
        # --- Find all files and create the dataset ---
        all_files = sorted([str(p) for p in Path(data_dir).rglob("*.npy")])
        if not all_files:
            print(f"[ERROR] No SAR data (.npy files) found in {data_dir}. Check the path.")
            return

        # Create dataset. Caching is disabled to prevent potential errors during analysis.
        dataset = SARSuperResDataset(file_list=all_files, data_dir=data_dir, use_cache=False)
        
        if len(dataset) == 0:
            print(f"[ERROR] No SAR data found in {data_dir}. Check the path.")
            return

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        num_to_process = min(num_samples, len(dataset))
        print(f"[DATA] Analyzing {num_to_process} sample(s) from '{data_dir}'...")
        
        processed_count = 0
        with tqdm(total=num_to_process, desc="Processing Batches") as pbar:
            for lr_batch, gt_batch in dataloader:
                if processed_count >= num_to_process:
                    break
                
                sr_batch = self.generate_super_resolved_images(lr_batch)
                
                for i in range(lr_batch.shape[0]):
                    if processed_count >= num_to_process:
                        break
                    
                    self.create_visual_comparison(sr_batch[i], gt_batch[i], processed_count)
                    processed_count += 1
                    pbar.update(1)

        print("\n[SUCCESS] Analysis completed!")
        print(f"All visualizations saved to: {self.output_dir}")

def main():
    """Main execution function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visual comparison of SAR SR images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ckpt", default="acswin_unet_pp.pth", help="Path to trained model checkpoint.")
    parser.add_argument("--patch-dir", required=True, help="Directory with SAR test patches.")
    parser.add_argument("--out-dir", default="./sar_comparison_results", help="Directory for output images.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to compare.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model inference.")
    
    args = parser.parse_args()
    
    analyzer = SARImageAnalyzer(model_path=args.ckpt, output_dir=args.out_dir)
    analyzer.run_analysis(data_dir=args.patch_dir, num_samples=args.num_samples, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
