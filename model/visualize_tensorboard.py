#!/usr/bin/env python3
"""
TensorBoard Visualization Script for SAR Super-Resolution
=========================================================

Utilities for analyzing and visualizing TensorBoard logs from SAR training.
Provides additional custom visualizations specific to SAR data analysis.
"""

import os
import subprocess
import webbrowser
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch


def launch_tensorboard(log_dir: str, port: int = 6006, auto_open: bool = True):
    """
    Launch TensorBoard server
    
    Args:
        log_dir: Directory containing TensorBoard logs
        port: Port for TensorBoard server
        auto_open: Automatically open browser
    """
    if not Path(log_dir).exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    print(f"Launching TensorBoard on port {port}...")
    print(f"Log directory: {log_dir}")
    
    # Launch TensorBoard
    cmd = f"tensorboard --logdir={log_dir} --port={port}"
    
    try:
        # Start TensorBoard process
        process = subprocess.Popen(cmd, shell=True)
        
        # Open browser if requested
        if auto_open:
            url = f"http://localhost:{port}"
            print(f"Opening browser: {url}")
            webbrowser.open(url)
        
        print(f"TensorBoard server started (PID: {process.pid})")
        print(f"Access at: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        # Wait for user interrupt
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping TensorBoard server...")
            process.terminate()
            process.wait()
            print("TensorBoard server stopped")
            
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")


def analyze_training_logs(log_dir: str):
    """
    Analyze training logs and generate summary statistics
    
    Args:
        log_dir: Directory containing TensorBoard logs
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    # Find event files
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    print(f"Analyzing logs from: {log_dir}")
    print(f"Found {len(event_files)} event files")
    
    # Load events
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    # Get available tags
    scalar_tags = ea.Tags()['scalars']
    image_tags = ea.Tags()['images']
    histogram_tags = ea.Tags()['histograms']
    
    print(f"\nAvailable scalar metrics: {len(scalar_tags)}")
    for tag in sorted(scalar_tags):
        print(f"  - {tag}")
    
    print(f"\nAvailable images: {len(image_tags)}")
    for tag in sorted(image_tags):
        print(f"  - {tag}")
    
    print(f"\nAvailable histograms: {len(histogram_tags)}")
    for tag in sorted(histogram_tags):
        print(f"  - {tag}")
    
    # Generate summary statistics
    if 'Loss/Train' in scalar_tags and 'Loss/Val' in scalar_tags:
        train_loss = ea.Scalars('Loss/Train')
        val_loss = ea.Scalars('Loss/Val')
        
        print(f"\nTraining Summary:")
        print(f"  Total epochs: {len(train_loss)}")
        print(f"  Final train loss: {train_loss[-1].value:.6f}")
        print(f"  Final val loss: {val_loss[-1].value:.6f}")
        print(f"  Best val loss: {min(s.value for s in val_loss):.6f}")
    
    if 'PSNR/Train' in scalar_tags and 'PSNR/Val' in scalar_tags:
        train_psnr = ea.Scalars('PSNR/Train')
        val_psnr = ea.Scalars('PSNR/Val')
        
        print(f"\nPSNR Summary:")
        print(f"  Final train PSNR: {train_psnr[-1].value:.2f} dB")
        print(f"  Final val PSNR: {val_psnr[-1].value:.2f} dB")
        print(f"  Best val PSNR: {max(s.value for s in val_psnr):.2f} dB")


def export_training_curves(log_dir: str, output_dir: str = "plots"):
    """
    Export training curves as matplotlib plots
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_dir: Directory to save plots
    """
    log_path = Path(log_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not log_path.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    # Load events
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    
    # Plot loss curves
    if 'Loss/Train' in scalar_tags and 'Loss/Val' in scalar_tags:
        train_loss = ea.Scalars('Loss/Train')
        val_loss = ea.Scalars('Loss/Val')
        
        train_steps = [s.step for s in train_loss]
        train_values = [s.value for s in train_loss]
        val_steps = [s.step for s in val_loss]
        val_values = [s.value for s in val_loss]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_values, label='Train Loss', color='blue')
        plt.plot(val_steps, val_values, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VV SAR Super-Resolution Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        loss_plot_path = output_path / "training_loss.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {loss_plot_path}")
    
    # Plot PSNR curves
    if 'PSNR/Train' in scalar_tags and 'PSNR/Val' in scalar_tags:
        train_psnr = ea.Scalars('PSNR/Train')
        val_psnr = ea.Scalars('PSNR/Val')
        
        train_steps = [s.step for s in train_psnr]
        train_values = [s.value for s in train_psnr]
        val_steps = [s.step for s in val_psnr]
        val_values = [s.value for s in val_psnr]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, train_values, label='Train PSNR', color='blue')
        plt.plot(val_steps, val_values, label='Validation PSNR', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('VV SAR Super-Resolution PSNR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        psnr_plot_path = output_path / "training_psnr.png"
        plt.savefig(psnr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PSNR plot saved to: {psnr_plot_path}")


def find_latest_run(runs_dir: str = "runs") -> str:
    """
    Find the latest TensorBoard run directory
    
    Args:
        runs_dir: Directory containing run subdirectories
        
    Returns:
        Path to latest run directory
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"Runs directory {runs_dir} does not exist")
        return None
    
    # Find all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
    
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return None
    
    # Sort by modification time to get latest
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest_run)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="TensorBoard Visualization for SAR Super-Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch TensorBoard for latest run
  python visualize_tensorboard.py --launch
  
  # Launch TensorBoard for specific run
  python visualize_tensorboard.py --launch --log-dir runs/sar_sr_20250129_140000
  
  # Analyze logs and export plots
  python visualize_tensorboard.py --analyze --export-plots
  
  # Find latest run directory
  python visualize_tensorboard.py --find-latest
        """
    )
    
    parser.add_argument('--launch', action='store_true',
                        help='Launch TensorBoard server')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze training logs')
    parser.add_argument('--export-plots', action='store_true',
                        help='Export training curves as plots')
    parser.add_argument('--find-latest', action='store_true',
                        help='Find latest run directory')
    parser.add_argument('--log-dir', type=str,
                        help='TensorBoard log directory')
    parser.add_argument('--port', type=int, default=6006,
                        help='TensorBoard server port')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Output directory for exported plots')
    
    args = parser.parse_args()
    
    # Find log directory
    log_dir = args.log_dir
    if not log_dir:
        log_dir = find_latest_run()
        if not log_dir:
            print("No log directory specified and no runs found")
            return
        print(f"Using latest run: {log_dir}")
    
    # Execute requested actions
    if args.find_latest:
        latest = find_latest_run()
        print(f"Latest run: {latest}")
    
    if args.analyze:
        analyze_training_logs(log_dir)
    
    if args.export_plots:
        export_training_curves(log_dir, args.output_dir)
    
    if args.launch:
        launch_tensorboard(log_dir, args.port, not args.no_browser)


if __name__ == "__main__":
    main() 