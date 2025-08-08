#!/usr/bin/env python3
"""NPY to PNG Visualization Script
==============================
Converts .npy files containing SAR complex data to PNG images for visualization.

Usage:
# Basic visualization of a single file (VV polarization)
python npy2png.py --input path/to/file.npy --output path/to/output.png
# Visualize VH polarization
python npy2png.py --input path/to/file.npy --output path/to/output.png --polarization VH
# Visualize amplitude and phase separately
python npy2png.py --input path/to/file.npy --output path/to/output --separate --polarization VV
# Batch process multiple files with VH polarization
python npy2png.py --input-dir path/to/npy/files --output-dir path/to/png/files --polarization VH
# Custom visualization options
python npy2png.py --input file.npy --output output.png --mode amplitude --colormap gray --polarization VV --dpi 300
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def load_npy_file(file_path: Path) -> np.ndarray:
    """Load .npy file and validate its structure."""
    try:
        data = np.load(file_path)
        if data.shape[0] not in [2, 4]:
            raise ValueError(f"Invalid shape {data.shape}. Expected (2, H, W) or (4, H, W).")
        
        print(f"Loaded data: shape={data.shape}, dtype={data.dtype}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def visualize_complex_data(data: np.ndarray, mode: str = 'amplitude', colormap: str = 'gray', 
                          dpi: int = 150, polarization: str = 'VV'):
    """Visualize complex SAR data."""
    # Handle different data formats
    if data.shape[0] == 4:
        # 4-channel data (VV-Re, VV-Im, VH-Re, VH-Im)
        vv_complex = data[0] + 1j * data[1]
        vh_complex = data[2] + 1j * data[3]
        
        # Select polarization
        if polarization.upper() == 'VV':
            complex_data = vv_complex
        elif polarization.upper() == 'VH':
            complex_data = vh_complex
        else:
            raise ValueError(f"Invalid polarization: {polarization}. Use 'VV' or 'VH'")
            
    elif data.shape[0] == 2:
        # 2-channel data - assume it's dual-pol complex: VV and VH
        # Each channel should already be complex (complex64)
        if data.dtype == np.complex64 or data.dtype == np.complex128:
            # Data is already complex
            if polarization.upper() == 'VV':
                complex_data = data[0]
            elif polarization.upper() == 'VH':
                complex_data = data[1]
            else:
                raise ValueError(f"Invalid polarization: {polarization}. Use 'VV' or 'VH'")
        else:
            # Data is real - treat as separate real channels (legacy support)
            print(f"Warning: 2-channel real data detected. Expected complex data.")
            if polarization.upper() == 'VV':
                complex_data = data[0]  # Use first channel as real data
            elif polarization.upper() == 'VH':
                complex_data = data[1]  # Use second channel as real data
            else:
                raise ValueError(f"Invalid polarization: {polarization}. Use 'VV' or 'VH'")
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    # Create title with polarization info
    pol_suffix = f"({polarization.upper()})"
    
    if mode == 'amplitude':
        # Calculate amplitude (sqrt(real^2 + imag^2))
        amp = np.abs(complex_data)
        title = f"Amplitude {pol_suffix}"
        vis_data = amp
    elif mode == 'phase':
        # Calculate phase (angle)
        phase = np.angle(complex_data)
        title = f"Phase {pol_suffix}"
        vis_data = phase
    elif mode == 'real':
        real = np.real(complex_data)
        title = f"Real Part {pol_suffix}"
        vis_data = real
    elif mode == 'imag':
        imag = np.imag(complex_data)
        title = f"Imaginary Part {pol_suffix}"
        vis_data = imag
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Get dimensions - data.shape is (channels, height, width)
    height_pixels, width_pixels = data.shape[1], data.shape[2]
    
    # Calculate aspect ratio based on pixel dimensions
    pixel_aspect_ratio = width_pixels / height_pixels  # 256/512 = 0.5
    
    # Create figure with proper pixel aspect ratio
    fig_height = 8  # inches
    fig_width = fig_height * pixel_aspect_ratio  # 8 * 0.5 = 4 inches
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Display image without physical scaling
    im = ax.imshow(vis_data, cmap=colormap, aspect='equal', origin='lower')
    
    # Add title with dimensions info
    title_with_info = f"{title} ({height_pixels}×{width_pixels} pixels)"
    ax.set_title(title_with_info, fontsize=12)
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig

def save_visualization(fig, output_path: Path, dpi: int = 150):
    """Save the visualization to a PNG file."""
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def process_single_file(input_path: Path, output_path: Path, mode: str = 'amplitude',
                       colormap: str = 'gray', dpi: int = 150, separate: bool = False,
                       polarization: str = 'VV'):
    """Process a single .npy file and create visualization(s)."""
    print(f"Processing: {input_path}")
    
    # Check if input is a directory
    if input_path.is_dir():
        print(f"Input is a directory. Processing all .npy files in {input_path}")
        process_directory(input_path, output_path.parent, mode, colormap, dpi, separate, polarization)
        return
    
    data = load_npy_file(input_path)
    if data is None:
        return

    if separate:
        # Create separate files for amplitude and phase
        fig_amp = visualize_complex_data(data, 'amplitude', colormap, dpi, polarization)
        amp_output = output_path.parent / f"{output_path.stem}_amplitude_{polarization.lower()}{output_path.suffix}"
        save_visualization(fig_amp, amp_output, dpi)

        fig_phase = visualize_complex_data(data, 'phase', 'twilight', dpi, polarization)
        phase_output = output_path.parent / f"{output_path.stem}_phase_{polarization.lower()}{output_path.suffix}"
        save_visualization(fig_phase, phase_output, dpi)
    else:
        # Create single visualization
        fig = visualize_complex_data(data, mode, colormap, dpi, polarization)
        save_visualization(fig, output_path, dpi)

def process_directory(input_dir: Path, output_dir: Path, mode: str = 'amplitude',
                     colormap: str = 'gray', dpi: int = 150, separate: bool = False,
                     polarization: str = 'VV'):
    """Process all .npy files in a directory."""
    # Find all .npy files
    npy_files = list(input_dir.rglob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for npy_file in tqdm(npy_files, desc="Processing files"):
        try:
            # Determine output path
            rel_path = npy_file.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.png')
            
            # Create subdirectories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process file
            process_single_file(npy_file, output_path, mode, colormap, dpi, separate, polarization)
            
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert .npy files to PNG visualizations")
    
    # Input/Output arguments
    parser.add_argument('--input', type=Path, help='Input .npy file path or directory')
    parser.add_argument('--output', type=Path, help='Output PNG file path or directory')
    parser.add_argument('--input-dir', type=Path, help='Input directory containing .npy files')
    parser.add_argument('--output-dir', type=Path, help='Output directory for PNG files')
    
    # Visualization options
    parser.add_argument('--mode', type=str, default='amplitude',
                       choices=['amplitude', 'phase', 'both', 'real', 'imag'],
                       help='Visualization mode')
    parser.add_argument('--colormap', type=str, default='gray',
                       help='Matplotlib colormap (default: gray)')
    parser.add_argument('--polarization', type=str, default='VV',
                       choices=['VV', 'VH', 'vv', 'vh'],
                       help='Polarization to visualize (default: VV)')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    parser.add_argument('--separate', action='store_true',
                       help='If True, create separate files for amplitude and phase')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir")
    
    if args.output and args.output_dir:
        parser.error("Cannot specify both --output and --output-dir")
    
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")
    
    # Process files
    try:
        if args.input:
            # Check if input is a directory
            if args.input.is_dir():
                # Directory processing
                if not args.output.is_dir():
                    # Create output directory if it doesn't exist
                    args.output.mkdir(parents=True, exist_ok=True)
                process_directory(args.input, args.output, args.mode,
                               args.colormap, args.dpi, args.separate, args.polarization)
            else:
                # Single file processing
                process_single_file(args.input, args.output, args.mode, 
                                  args.colormap, args.dpi, args.separate, args.polarization)
        elif args.input_dir:
            # Directory processing
            process_directory(args.input_dir, args.output_dir, args.mode,
                           args.colormap, args.dpi, args.separate, args.polarization)
        else:
            parser.error("Must specify either --input or --input-dir")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()