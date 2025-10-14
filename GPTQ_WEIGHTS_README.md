# Real GPTQ Weights with Marlin Kernel

This directory contains the working configuration for running Marlin kernel with real GPTQ weights from Qwen2.5-3B-Instruct-GPTQ-Int4.

## Files

### Weight and Scale Files
- `up_proj_qweight.bin` - Real GPTQ quantized weights (int32 packed) from `model.layers.0.mlp.up_proj.qweight`
  - Shape: (256, 11008) - 8 packed 4-bit values per int32
  - Original dimensions: K=2048, N=11008

- `up_proj_scales.bin` - Original grouped quantization scales (float16)
  - Shape: (16, 11008) - for groupsize=128

- `up_proj_scales_percolumn.bin` - Per-column scales (float16) for groupsize=-1
  - Shape: (1, 11008) - averaged from grouped scales

### Scripts
- `extract_weight.py` - Downloads and extracts weights from HuggingFace model
- `convert_scales_for_percolumn.py` - Converts grouped scales to per-column scales

### Test Program
- `test_bisect.cu` - Diagnostic tool for testing different weight/scale combinations
- `test_bisect` - Compiled binary

## Working Configuration

**Kernel Parameters:**
- M=1 (batch size)
- K=2048 (input features)
- N=11008 (output features)
- **groupsize=-1** (per-column quantization)

**Important Finding:** The Marlin kernel only works correctly with `groupsize=-1` (per-column quantization). Using `groupsize=128` produces all-zero outputs, even with random data.

## Usage

### Download Weights
```bash
python extract_weight.py
```

### Convert Scales to Per-Column Format
```bash
python convert_scales_for_percolumn.py
```

### Run Test
```bash
# Compile
nvcc -o test_bisect test_bisect.cu -std=c++11 -arch=sm_80 --expt-relaxed-constexpr -lcuda

# Test with real GPTQ weights + groupsize=-1
./test_bisect b -1
```

Expected output:
```
Loading GPTQ weights from binary files...
Successfully loaded GPTQ qweight: shape=(256, 11008)
Successfully loaded per-column scales: shape=(1, 11008)

Configuration: M=1, K=2048, N=11008, groupsize=-1
Testing with: GPTQ weights + per-column scales

First 20 output values:
  C[ 0] = 0.046173
  C[ 1] = -0.124695
  C[ 2] = -0.050140
  ...
✓ SUCCESS: Found non-zero outputs! Max abs value: 0.475342
```

## Test Options

```bash
./test_bisect r -1     # Random weights + random scales, groupsize=-1 (works)
./test_bisect r 128    # Random weights + random scales, groupsize=128 (fails - all zeros)
./test_bisect b -1     # GPTQ weights + per-column scales, groupsize=-1 (works!)
./test_bisect b 128    # GPTQ weights + grouped scales, groupsize=128 (fails - all zeros)
```

## Source

Model: [Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4)

Layer: `model.layers.0.mlp.up_proj` (first layer MLP up-projection)
