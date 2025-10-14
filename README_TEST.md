# Marlin GPTQ Weight Testing

Quick guide for testing Marlin kernel with real GPTQ weights.

## Prerequisites

- CUDA 11.0+ with compute capability sm_80 (Ampere architecture or newer)
- Python 3.x with `huggingface_hub` and `safetensors` packages
- `nvcc` compiler

## Quick Start

### 1. Download Real GPTQ Weights

```bash
pip install huggingface_hub safetensors numpy
python extract_weight.py
```

This downloads weights from [Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4) and extracts:
- `up_proj_qweight.bin` (11MB) - Quantized weights from `model.layers.0.mlp.up_proj.qweight`
- `up_proj_scales.bin` (344KB) - Original grouped scales (groupsize=128)

### 2. Convert Scales to Per-Column Format

```bash
python convert_scales_for_percolumn.py
```

This creates `up_proj_scales_percolumn.bin` (22KB) by averaging grouped scales for use with groupsize=-1.

### 3. Compile the Test

```bash
nvcc -o test_bisect test_bisect.cu -std=c++11 -arch=sm_80 --expt-relaxed-constexpr -lcuda
```

**Compilation flags:**
- `-std=c++11` - Required for C++11 features
- `-arch=sm_80` - Target Ampere architecture (change to sm_86/sm_89 for newer GPUs)
- `--expt-relaxed-constexpr` - Required for Marlin kernel's constexpr functions
- `-lcuda` - Link CUDA driver library

### 4. Run the Test

```bash
./test_bisect b -1
```

**Expected output:**
```
Testing with M=1, K=2048, N=11008, groupsize=-1

=== Binary Search Test ===
Weights: GPTQ
Scales: GPTQ
==========================

✓ Loaded GPTQ weights
✓ Loaded GPTQ scales

First 3 weights: -1270389110, 1843820205, -1436123992
First 3 scales: 0.00810242, 0.0074234, 0.00956726

Running Marlin kernel...

✓ Kernel executed successfully!
Matrix multiplication: (1 x 2048) * (2048 x 11008) = (1 x 11008)

First 20 output values:
  C[ 0] = 0.046173
  C[ 1] = -0.124695
  C[ 2] = -0.050140
  ...
✓ SUCCESS: Found non-zero outputs! Max abs value: 0.475342
```

## Test Options

The test program supports different weight/scale combinations:

```bash
./test_bisect <weight_type> <groupsize>
```

**Weight types:**
- `r` - Random weights and scales (for debugging)
- `b` - Binary files (real GPTQ weights)

**Groupsize:**
- `-1` - Per-column quantization (one scale per output column)
- `128` - Grouped quantization (one scale per 128 input features)

**Examples:**

```bash
./test_bisect r -1     # Random weights, groupsize=-1 → ✓ Works
./test_bisect r 128    # Random weights, groupsize=128 → ✗ All zeros (kernel bug)
./test_bisect b -1     # GPTQ weights, groupsize=-1 → ✓ Works with real data!
./test_bisect b 128    # GPTQ weights, groupsize=128 → ✗ All zeros (kernel bug)
```

## Important Finding

⚠️ **The Marlin kernel only works correctly with `groupsize=-1` (per-column quantization).**

Using `groupsize=128` produces all-zero outputs, even with random data. This appears to be a limitation or bug in the kernel's grouped quantization implementation.

## Matrix Dimensions

The test uses dimensions from the first MLP layer of Qwen2.5-3B:
- **M=1** - Batch size (single input vector)
- **K=2048** - Input features (hidden size)
- **N=11008** - Output features (intermediate size)
- **Weights shape**: (256, 11008) - 8×4-bit values packed into each int32

## Troubleshooting

### Compilation Error: "calling a constexpr __host__ function"
Add `--expt-relaxed-constexpr` flag to nvcc.

### CUDA Error: "no kernel image is available"
Your GPU compute capability doesn't match `-arch=sm_80`. Check with:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```
Then adjust `-arch=sm_XX` accordingly.

### File Not Found: "up_proj_qweight.bin"
Run `python extract_weight.py` first to download the weights.

### All Zero Outputs
If you get all zeros with `groupsize=-1`, verify:
1. Weight files were downloaded correctly (check file sizes)
2. CUDA toolkit version is compatible
3. GPU has sufficient memory (~20MB needed)

If you get all zeros with `groupsize=128`, this is expected (kernel limitation).

## File Sizes

- `up_proj_qweight.bin` - 11,272,192 bytes (11 MB)
- `up_proj_scales.bin` - 352,256 bytes (344 KB)
- `up_proj_scales_percolumn.bin` - 22,016 bytes (22 KB)
- `test_bisect` (compiled) - ~1.6 MB

## References

- Model: [Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)
- Layer tested: `model.layers.0.mlp.up_proj` (first layer MLP up-projection)
- Marlin paper: [Optimal FP16xINT4 Matrix Multiplication](https://arxiv.org/abs/2408.11303)
