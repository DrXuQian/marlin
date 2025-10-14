#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Decode FP8 E4M3 manually
float fp8_e4m3_to_float(uint8_t x) {
    uint32_t sign = (x >> 7) & 0x1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mantissa = x & 0x7;

    printf("  Byte: %3u (0x%02x) -> sign=%u, exp=%u (0x%x), mantissa=%u\n",
           x, x, sign, exp, exp, mantissa);

    // Handle special cases
    if (exp == 0) {
        if (mantissa == 0) {
            printf("    -> Zero\n");
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal: 2^-6 * (mantissa / 8)
        float val = ldexpf((float)mantissa / 8.0f, -6);
        printf("    -> Subnormal: %f\n", sign ? -val : val);
        return sign ? -val : val;
    }
    if (exp == 0xF) {
        printf("    -> NaN\n");
        return NAN;
    }

    // Normal number: (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    int exponent = (int)exp - 7;
    float val = ldexpf(1.0f + (float)mantissa / 8.0f, exponent);
    printf("    -> Normal: 2^%d * %.3f = %f\n", exponent, 1.0f + (float)mantissa / 8.0f, sign ? -val : val);
    return sign ? -val : val;
}

int main() {
    printf("Testing FP8 E4M3 decoder with real weight bytes:\n\n");

    // First 10 weight bytes from the file
    uint8_t test_bytes[] = {218, 95, 217, 100, 92, 80, 88, 218, 101, 222};

    for (int i = 0; i < 10; i++) {
        printf("Value %d:\n", i);
        float result = fp8_e4m3_to_float(test_bytes[i]);
        printf("  Result: %f\n\n", result);
    }

    return 0;
}
