#!/usr/bin/env python3
"""
Script to download and extract a specific tensor from a safetensors file
Usage: python extract_weight.py
"""

from safetensors import safe_open
from huggingface_hub import hf_hub_download
import numpy as np
import os

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
SAFETENSORS_FILE = "model.safetensors"
TENSOR_NAME = "model.layers.0.mlp.up_proj.qweight"
SCALE_TENSOR_NAME = "model.layers.0.mlp.up_proj.scales"
OUTPUT_FILE = "up_proj_qweight.bin"
SCALE_OUTPUT_FILE = "up_proj_scales.bin"

def extract_tensor(f, tensor_name, output_file):
    """Helper function to extract and save a tensor"""
    if tensor_name not in f.keys():
        print(f"Warning: Tensor '{tensor_name}' not found!")
        return False

    tensor = f.get_tensor(tensor_name)
    print(f"Tensor '{tensor_name}':")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Size: {tensor.nbytes} bytes")

    # Save to binary file
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    tensor.tofile(output_path)
    print(f"  Saved to: {output_path}")

    # Also save metadata
    meta_file = output_path.replace('.bin', '_meta.txt')
    with open(meta_file, 'w') as mf:
        mf.write(f"Tensor name: {tensor_name}\n")
        mf.write(f"Shape: {tensor.shape}\n")
        mf.write(f"Dtype: {tensor.dtype}\n")
        mf.write(f"Size (bytes): {tensor.nbytes}\n")
    print(f"  Metadata: {meta_file}\n")

    return True

def main():
    print(f"Downloading {SAFETENSORS_FILE} from {MODEL_ID}...")

    # Download the safetensors file
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=SAFETENSORS_FILE,
        cache_dir=None  # Uses default HF cache
    )

    print(f"File downloaded to: {model_path}\n")

    # Open safetensors file and extract tensors
    with safe_open(model_path, framework="numpy") as f:
        # Extract qweight
        if not extract_tensor(f, TENSOR_NAME, OUTPUT_FILE):
            print(f"Available tensors: {list(f.keys())[:20]}...")
            return

        # Extract scales
        extract_tensor(f, SCALE_TENSOR_NAME, SCALE_OUTPUT_FILE)

if __name__ == "__main__":
    main()
