#!/usr/bin/env python3
"""
Extract FP8 weights from Llama-3.1-8B-Instruct-FP8 model
"""
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download

MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-FP8"
SHARD_FILE = "model-00002-of-00002.safetensors"
TENSOR_NAME = "model.layers.18.mlp.up_proj.weight"

print(f"Downloading {SHARD_FILE} from {MODEL_ID}...")
model_path = hf_hub_download(repo_id=MODEL_ID, filename=SHARD_FILE)
print(f"Downloaded to: {model_path}")

print(f"\nExtracting tensor: {TENSOR_NAME}")
with safe_open(model_path, framework="pt") as f:
    # List all tensors in this shard
    print("\nTensors in this shard:")
    for key in f.keys():
        if "layer.18" in key:
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    # Extract the specific tensor (PyTorch tensor)
    weight = f.get_tensor(TENSOR_NAME)
    print(f"\n✓ Extracted {TENSOR_NAME}")
    print(f"  Shape: {weight.shape}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  Size: {weight.numel()} elements")

    # Convert FP8 tensor to bytes
    # PyTorch FP8 types: torch.float8_e4m3fn (most common for weights)
    # View as uint8 directly in PyTorch, then convert to numpy
    weight_bytes = weight.cpu().view(dtype=torch.uint8).numpy()
    print(f"  Bytes: {len(weight_bytes):,}")

    # Save to binary file
    output_file = "up_proj_fp8_weight.bin"
    weight_bytes.tofile(output_file)
    print(f"\n✓ Saved to: {output_file}")

    # Print first few raw byte values for verification
    print(f"\nFirst 10 raw bytes (as uint8): {weight_bytes[:10]}")
    print(f"Shape in file: {weight.shape} -> {weight.shape[0]} x {weight.shape[1]}")

    # Save metadata
    with open("up_proj_fp8_weight_meta.txt", "w") as meta:
        meta.write(f"tensor_name: {TENSOR_NAME}\n")
        meta.write(f"shape: {weight.shape}\n")
        meta.write(f"dtype: {weight.dtype}\n")
        meta.write(f"size_bytes: {weight.nbytes}\n")

    print("\n✓ Done!")
