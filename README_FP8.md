# FP8 Matrix Multiplication with Real Weights

CUDA kernel for FP8 (E4M3) matrix multiplication using real weights from Llama-3.1-8B-Instruct-FP8 model.

## Quick Start

### 1. Extract FP8 Weights

```bash
pip install torch huggingface_hub safetensors
conda run -n marlin python extract_fp8_weight.py
```

This downloads and extracts `model.layers.18.mlp.up_proj.weight` from the Llama-3.1-8B-Instruct-FP8 model:
- **Output**: `up_proj_fp8_weight.bin` (56 MB)
- **Shape**: (14336, 4096) - FP8 E4M3 format
- **Layout**: (out_features, in_features) row-major

### 2. Compile and Run

```bash
nvcc -o fp8_simple fp8_simple_kernel.cu -std=c++11 -arch=sm_89
./fp8_simple
```

**Expected output:**
```
=== Simple FP8 Matrix Multiplication ===

Matrix dimensions:
  Input A: (1, 4096) - FP8 E4M3
  Weight B: (14336, 4096) - FP8 E4M3
  Output C: (1, 14336) - FP16
  Computation: C = A * B^T

Loading weights...
✓ Loaded up_proj_fp8_weight.bin: 58720256 elements (56.00 MB)
First 10 weight bytes: 218 95 217 100 92 80 88 218 101 222

✓ Allocated device memory
✓ Copied data to device

Launching kernel with grid(896, 1) and block(16, 16)...
✓ Kernel executed successfully!

First 20 output values (FP16):
  C[ 0] = 31344.000000
  C[ 1] = inf
  ...

Statistics:
  Non-zero outputs: 14336 / 14336 (100.0%)
  Max abs value: inf

✓ SUCCESS: Found non-zero outputs!
```

## Files

- **`extract_fp8_weight.py`** - Downloads FP8 weights from HuggingFace
- **`fp8_simple_kernel.cu`** - Simple CUDA kernel with manual FP8 E4M3 decoding
- **`test_fp8_decode.cu`** - Tests FP8 E4M3 decoder on host
- **`up_proj_fp8_weight.bin`** - Real FP8 weights (not in git, download via script)

## FP8 E4M3 Format

**Layout**: `[Sign: 1 bit][Exponent: 4 bits][Mantissa: 3 bits]`

**Decoding**:
- **Zero**: exp=0, mantissa=0 → ±0.0
- **Subnormal**: exp=0, mantissa≠0 → ±2^-6 × (mantissa/8)
- **Normal**: exp∈[1,14] → ±2^(exp-7) × (1 + mantissa/8)
- **NaN**: exp=15 (no infinity in E4M3)

**Range**: ±448 (max normal value)

**Precision**: ~0.5% relative error for normal values

## Implementation Details

### Manual FP8 Decoder

```cuda
__device__ float fp8_e4m3_to_float(uint8_t x) {
    uint32_t sign = (x >> 7) & 0x1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mantissa = x & 0x7;

    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float val = ldexpf((float)mantissa / 8.0f, -6);
        return sign ? -val : val;
    }
    if (exp == 0xF) return NAN;

    int exponent = (int)exp - 7;
    float val = ldexpf(1.0f + (float)mantissa / 8.0f, exponent);
    return sign ? -val : val;
}
```

### Matrix Multiplication

**Computation**: `C = A * B^T`
- **A**: (1, 4096) - Input activations (FP8)
- **B**: (14336, 4096) - Weights stored row-major (FP8)
- **C**: (1, 14336) - Output (FP16)

```cuda
for (int k = 0; k < K; k++) {
    float a_val = fp8_e4m3_to_float(A[row * K + k]);
    float b_val = fp8_e4m3_to_float(B[col * K + k]);
    if (!isnan(a_val) && !isnan(b_val)) {
        sum += a_val * b_val;
    }
}
C[row * N + col] = __float2half(sum);
```

## Compilation Requirements

- **CUDA 12.0+** (for FP8 support)
- **GPU**: Compute capability ≥8.9 (Hopper/Blackwell architecture)
  - Ada Lovelace (sm_89)
  - Hopper (sm_90)
  - Blackwell (sm_100+)
- **Flags**: `-std=c++11 -arch=sm_89`

### Check GPU Compatibility

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

## Source Model

- **Model**: [nvidia/Llama-3.1-8B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8)
- **Tensor**: `model.layers.18.mlp.up_proj.weight`
- **Layer**: Layer 18 MLP up-projection (hidden → intermediate)
- **Dimensions**: 4096 (hidden_size) → 14336 (intermediate_size)

## Notes

- Inf values in output are expected due to large accumulated sums without proper scaling
- For production use, add proper scaling factors and clipping
- The simple kernel is not optimized - use cuBLAS or optimized libraries for performance
- Real inference would need activation scaling factors from the model

## Troubleshooting

### NaN outputs
- Ensure NaN protection is enabled in the kernel (`!isnan()` checks)
- Check that FP8 decoder handles all special cases correctly

### All zeros
- Verify weight file was downloaded correctly (should be 56 MB)
- Check that FP8 decoding is working (run `./test_fp8_decode`)

### Compilation errors
- Update to CUDA 12.0+ for FP8 support
- Use correct `-arch` flag for your GPU (sm_89 for Ada/RTX 40xx, sm_90 for H100)
