/*
 * Simple FP8 Matrix Multiplication Kernel
 *
 * Uses real FP8 weights from Llama-3.1-8B-Instruct-FP8
 * Tensor: model.layers.18.mlp.up_proj.weight
 * Shape: (14336, 4096) - stored as (out_features, in_features) row-major
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Decode FP8 E4M3 manually
// E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t x) {
    uint32_t sign = (x >> 7) & 0x1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mantissa = x & 0x7;

    // Handle special cases
    if (exp == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;  // Zero
        // Subnormal: 2^-6 * (mantissa / 8)
        float val = ldexpf((float)mantissa / 8.0f, -6);
        return sign ? -val : val;
    }
    if (exp == 0xF) {
        // NaN (E4M3 doesn't have infinity)
        return __int_as_float(0x7FC00000);
    }

    // Normal number: (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    int exponent = (int)exp - 7;
    float val = ldexpf(1.0f + (float)mantissa / 8.0f, exponent);
    return sign ? -val : val;
}

// Simple FP8 matmul kernel: C = A * B^T
// A: (M, K) FP8, B: (N, K) FP8, C: (M, N) FP16
__global__ void fp8_matmul_kernel(
    const uint8_t* A,  // (M, K)
    const uint8_t* B,  // (N, K)
    __half* C,         // (M, N)
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Compute dot product of A[row, :] and B[col, :]
        for (int k = 0; k < K; k++) {
            float a_val = fp8_e4m3_to_float(A[row * K + k]);
            float b_val = fp8_e4m3_to_float(B[col * K + k]);

            // Skip NaN values
            if (!isnan(a_val) && !isnan(b_val)) {
                sum += a_val * b_val;
            }
        }

        // Convert result to FP16
        C[row * N + col] = __float2half(sum);
    }
}

// Helper function to load binary file
template<typename T>
std::vector<T> loadBinary(const char* filename, size_t expected_size = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = size / sizeof(T);
    if (expected_size > 0 && num_elements != expected_size) {
        fprintf(stderr, "Size mismatch in %s: expected %zu elements, got %zu\n",
                filename, expected_size, num_elements);
        exit(EXIT_FAILURE);
    }

    std::vector<T> buffer(num_elements);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        fprintf(stderr, "Failed to read file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    printf("✓ Loaded %s: %zu elements (%.2f MB)\n",
           filename, num_elements, size / (1024.0 * 1024.0));
    return buffer;
}

int main() {
    printf("=== Simple FP8 Matrix Multiplication ===\n\n");

    // Matrix dimensions
    // Weight shape in file: (14336, 4096) stored as (out_features, in_features) row-major
    // For computation: C = A * B^T where A is (M, K), B is (N, K), C is (M, N)
    const int M = 1;      // Batch size
    const int K = 4096;   // Input features
    const int N = 14336;  // Output features

    printf("Matrix dimensions:\n");
    printf("  Input A: (%d, %d) - FP8 E4M3\n", M, K);
    printf("  Weight B: (%d, %d) - FP8 E4M3\n", N, K);
    printf("  Output C: (%d, %d) - FP16\n", M, N);
    printf("  Computation: C = A * B^T\n\n");

    // Load FP8 weights
    printf("Loading weights...\n");
    auto weight_bytes = loadBinary<uint8_t>("up_proj_fp8_weight.bin", N * K);

    printf("First 10 weight bytes: ");
    for (int i = 0; i < 10; i++) {
        printf("%u ", weight_bytes[i]);
    }
    printf("\n\n");

    // Allocate device memory
    uint8_t *d_A, *d_B;
    __half *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_B, N * K * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(__half)));

    printf("✓ Allocated device memory\n");

    // Create random input (FP8)
    std::vector<uint8_t> input_bytes(M * K);
    for (int i = 0; i < M * K; i++) {
        // Generate reasonable FP8 values (moderate range)
        input_bytes[i] = (rand() % 64) + 64;
    }

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, input_bytes.data(), M * K * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, weight_bytes.data(), N * K * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    printf("✓ Copied data to device\n\n");

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    printf("Launching kernel with grid(%d, %d) and block(%d, %d)...\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    fp8_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("✓ Kernel executed successfully!\n\n");

    // Copy result back
    std::vector<__half> output(M * N);
    CHECK_CUDA(cudaMemcpy(output.data(), d_C, M * N * sizeof(__half),
                          cudaMemcpyDeviceToHost));

    // Print first 20 output values
    printf("First 20 output values (FP16):\n");
    for (int i = 0; i < 20 && i < M * N; i++) {
        printf("  C[%2d] = %.6f\n", i, __half2float(output[i]));
    }

    // Check for non-zero outputs
    float max_abs = 0.0f;
    int non_zero_count = 0;
    for (int i = 0; i < M * N; i++) {
        float val = __half2float(output[i]);
        float abs_val = fabsf(val);
        if (abs_val > 1e-6f) non_zero_count++;
        if (abs_val > max_abs) max_abs = abs_val;
    }

    printf("\n");
    printf("Statistics:\n");
    printf("  Non-zero outputs: %d / %d (%.1f%%)\n",
           non_zero_count, M * N, 100.0f * non_zero_count / (M * N));
    printf("  Max abs value: %.6f\n", max_abs);

    if (max_abs > 0.0f) {
        printf("\n✓ SUCCESS: Found non-zero outputs!\n");
    } else {
        printf("\n✗ WARNING: All outputs are zero!\n");
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    printf("\n✓ Cleaned up resources\n");

    return 0;
}
