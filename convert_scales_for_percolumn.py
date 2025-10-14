#!/usr/bin/env python3
"""
Convert grouped scales (16, 11008) to per-column scales (1, 11008)
by averaging across groups for each column
"""

import numpy as np

# Load grouped scales
scales_grouped = np.fromfile('up_proj_scales.bin', dtype=np.float16).reshape(16, 11008)

print(f"Grouped scales shape: {scales_grouped.shape}")
print(f"First 5 values of group 0: {scales_grouped[0, :5]}")

# For per-column, we need one scale per column
# Take the mean of all groups for each column
scales_percolumn = scales_grouped.mean(axis=0, keepdims=True)  # (1, 11008)

print(f"\nPer-column scales shape: {scales_percolumn.shape}")
print(f"First 5 per-column scales: {scales_percolumn[0, :5]}")

# Save
scales_percolumn.astype(np.float16).tofile('up_proj_scales_percolumn.bin')

print(f"\n✓ Saved up_proj_scales_percolumn.bin")
print(f"Size: {scales_percolumn.nbytes} bytes")
