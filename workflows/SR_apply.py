#!/usr/bin/env python3
"""
Super Resolution Model Application Script
=======================================
Applies a trained super resolution model to .npy files and saves the results.

Usage:
  # Apply to a single file
  python SR_apply.py --input path/to/file.npy --output path/to/output.npy --model path/to/model.pth

  # Apply to all files in a directory
  python SR_apply.py --input-dir path/to/input/dir --output-dir path/to/output/dir --model path/to/model.pth

  # Apply with specific model configuration
  python SR_apply.py --input file.npy --output output.npy --model model.pth --model-type swin_unet

  # Apply with GPU acceleration
  python SR_apply.py --input file.npy --output output.npy --model model.pth --use-gpu
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / 'model'))

try:
    from ac_swin_unet_pp import create_model as create_swin_model
    from cv_unet import ComplexUNet
except ImportError as e:
    print(f"Warning: Could not import model modules: {e}")
    print("Make sure the model files are in the model/ directory")

def load_model(model_path: Path, model_type: str = 'swin_unet', device: str = 'cpu'):
    """
    Load the trained super resolution model.
    
    Args:
        model_path: Path to the trained model weights
        model_type: Type of model ('swin_unet', 'cv_unet')
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model
    """
    device = torch.device(device)
    
    if model_type == 'swin_unet':
        model = create_swin_model()
    elif model_type == 'cv_unet':
        model = ComplexUNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights with weights_only=False for compatibility
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=False, trying with weights_only=True...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e2:
            print(f"Error loading model: {e2}")
            raise
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from {model_path}")
    return model

def preprocess_data(data: np.ndarray, device: str = 'cpu', model_type: str = 'swin_unet') -> torch.Tensor:
    """
    Preprocess input data for model inference.
    
    Args:
        data: Input data with shape (2, H, W) for complex64 SAR data
        device: Device to put tensor on
        model_type: Type of model to determine input format
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Validate input data
    if data.shape[0] != 2:
        raise ValueError(f"Expected data with shape (2, H, W), got {data.shape}")
    
    if data.dtype != np.complex64:
        raise ValueError(f"Expected complex64 data, got {data.dtype}")
    
    # Model-specific preprocessing
    if model_type == 'swin_unet':
        # ACSwinUNetPP expects 4-channel real input (VV-Re, VV-Im, VH-Re, VH-Im)
        # Convert from complex SAR data to 4-channel real
        vv_complex = data[0]  # VV polarization
        vh_complex = data[1]  # VH polarization
        
        # Create 4-channel real data
        real_data = np.stack([
            vv_complex.real,  # VV-Re
            vv_complex.imag,  # VV-Im
            vh_complex.real,  # VH-Re
            vh_complex.imag   # VH-Im
        ], axis=0)  # (4, H, W)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(real_data).float().unsqueeze(0).to(device)  # (1, 4, H, W)
        return tensor
        
    elif model_type == 'cv_unet':
        # ComplexUNet expects different format - keeping original logic for compatibility
        # Convert complex to real representation
        vv_complex = data[0]
        vh_complex = data[1]
        
        # Use magnitude and components
        magnitude = np.abs(vv_complex)
        real_part = vv_complex.real
        imag_part = vv_complex.imag
        
        real_data = np.stack([magnitude, real_part, imag_part], axis=0)  # (3, H, W)
        tensor = torch.from_numpy(real_data).float().unsqueeze(0).to(device)  # (1, 3, H, W)
        return tensor
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def postprocess_data(output: torch.Tensor, model_type: str = 'swin_unet') -> np.ndarray:
    """
    Postprocess model output to get final result.
    
    Args:
        output: Model output tensor
        model_type: Type of model to determine output format
        
    Returns:
        Processed numpy array with shape (2, H, W) complex64 for SAR data
    """
    # Remove batch dimension if present
    if output.dim() == 4:
        output = output.squeeze(0)
    
    # Convert to numpy
    output = output.detach().cpu().numpy()
    
    if model_type == 'swin_unet':
        # Model outputs 4-channel real data (VV-Re, VV-Im, VH-Re, VH-Im)
        # Convert back to complex SAR format (2, H, W) complex64
        if output.shape[0] == 4:
            vv_complex = output[0] + 1j * output[1]  # VV-Re + j*VV-Im
            vh_complex = output[2] + 1j * output[3]  # VH-Re + j*VH-Im
            
            complex_output = np.stack([vv_complex, vh_complex], axis=0).astype(np.complex64)
            return complex_output
        else:
            print(f"Warning: Expected 4-channel output, got {output.shape[0]} channels")
            return output.astype(np.complex64)
    
    elif model_type == 'cv_unet':
        # ComplexUNet output processing (adjust based on actual output format)
        return output.astype(np.complex64)
    
    else:
        return output.astype(np.complex64)

def apply_sr_model(model: torch.nn.Module, input_data: np.ndarray, 
                   device: str = 'cpu', model_type: str = 'swin_unet') -> np.ndarray:
    """
    Apply super resolution model to input data.
    
    Args:
        model: Loaded super resolution model
        input_data: Input data with shape (2, H, W)
        device: Device to run inference on
        model_type: Type of model for preprocessing
        
    Returns:
        Super resolved output with shape (2, H*scale, W*scale)
    """
    # Preprocess
    input_tensor = preprocess_data(input_data, device, model_type)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    output_data = postprocess_data(output, model_type)
    
    return output_data

def process_single_file(input_path: Path, output_path: Path, model: torch.nn.Module,
                       device: str = 'cpu', model_type: str = 'swin_unet'):
    """
    Process a single .npy file with the SR model.
    
    Args:
        input_path: Input .npy file path
        output_path: Output .npy file path
        model: Loaded super resolution model
        device: Device to run inference on
        model_type: Type of model for preprocessing
    """
    print(f"Processing: {input_path}")
    
    try:
        # Load input data
        input_data = np.load(input_path)
        
        # Validate input data
        if input_data.ndim != 3 or input_data.shape[0] != 2:
            raise ValueError(f"Expected 3D array with shape (2, H, W), got {input_data.shape}")
        
        if input_data.dtype != np.complex64:
            print(f"Warning: Expected complex64 data, got {input_data.dtype}. Converting...")
            input_data = input_data.astype(np.complex64)
        
        print(f"Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        # Apply SR model
        output_data = apply_sr_model(model, input_data, device, model_type)
        
        print(f"Output shape: {output_data.shape}")
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, output_data)
        
        print(f"Saved: {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_input(input_path: Path, output_path: Path, model: torch.nn.Module,
                 device: str = 'cpu', model_type: str = 'swin_unet'):
    """
    Process input path - can be either a file or directory.
    
    Args:
        input_path: Input path (file or directory)
        output_path: Output path (file or directory)
        model: Loaded super resolution model
        device: Device to run inference on
        model_type: Type of model for preprocessing
    """
    if input_path.is_file():
        # Single file processing
        process_single_file(input_path, output_path, model, device, model_type)
    elif input_path.is_dir():
        # Directory processing
        print(f"Input is a directory. Processing all .npy files in {input_path}")
        
        # Create output directory if it doesn't exist
        if not output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
        
        process_directory(input_path, output_path, model, device, model_type)
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")

def process_directory(input_dir: Path, output_dir: Path, model: torch.nn.Module,
                     device: str = 'cpu', model_type: str = 'swin_unet'):
    """
    Process all .npy files in a directory.
    
    Args:
        input_dir: Input directory containing .npy files
        output_dir: Output directory for processed files
        model: Loaded super resolution model
        device: Device to run inference on
        model_type: Type of model for preprocessing
    """
    # Find all .npy files
    npy_files = list(input_dir.rglob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    # Process each file
    for npy_file in tqdm(npy_files, desc="Processing files"):
        try:
            # Determine output path
            rel_path = npy_file.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Process file
            process_single_file(npy_file, output_path, model, device, model_type)
            
        except Exception as e:
            print(f"Error processing {npy_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply super resolution model to .npy files")
    
    # Input/Output arguments
    parser.add_argument('--input', type=Path, help='Input .npy file path or directory')
    parser.add_argument('--output', type=Path, help='Output .npy file path or directory')
    parser.add_argument('--input-dir', type=Path, help='Input directory containing .npy files')
    parser.add_argument('--output-dir', type=Path, help='Output directory for processed files')
    
    # Model arguments
    parser.add_argument('--model', type=Path, required=True, help='Path to trained model weights')
    parser.add_argument('--model-type', type=str, default='swin_unet',
                       choices=['swin_unet', 'cv_unet'],
                       help='Type of model to use (default: swin_unet)')
    
    # Processing options
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--device', type=str, default=None,
                       help='Specific device to use (e.g., "cuda:0", "cpu")')
    
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
    
    # Determine device
    if args.device:
        device = args.device
    elif args.use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not args.model.exists():
        print(f"Error: Model file {args.model} does not exist")
        sys.exit(1)
    
    try:
        # Load model
        print(f"Loading {args.model_type} model...")
        model = load_model(args.model, args.model_type, device)
        
        # Process files
        if args.input:
            # Check if input exists
            if not args.input.exists():
                print(f"Error: Input path {args.input} does not exist")
                sys.exit(1)
            
            # Process input (file or directory)
            process_input(args.input, args.output, model, device, args.model_type)
        elif args.input_dir:
            # Directory processing
            process_directory(args.input_dir, args.output_dir, model, device, args.model_type)
        else:
            parser.error("Must specify either --input or --input-dir")
            
        print("Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
