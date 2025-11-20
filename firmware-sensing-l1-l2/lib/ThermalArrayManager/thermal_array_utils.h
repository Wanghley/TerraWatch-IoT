#pragma once
#include <Arduino.h>
#include <string.h> // For memcpy

/**
 * @brief Defines the desired clockwise rotation for an 8x8 matrix.
 */
enum class MatrixRotation : uint8_t {
    ROT_0_CW,   // No rotation
    ROT_90_CW,  // 90 degrees clockwise
    ROT_180_CW, // 180 degrees
    ROT_270_CW  // 270 degrees clockwise (or 90-deg counter-clockwise)
};

/**
 * @brief Rotates an 8x8 matrix (64 floats) by a specified amount.
 * * @param src Pointer to the 8x8 source array (64 floats).
 * @param dst Pointer to the 8x8 destination array (64 floats).
 * @param rotation The desired rotation (ROT_0_CW, ROT_90_CW, ROT_180_CW, ROT_270_CW).
 */
inline void rotateMatrix(const float* src, float* dst, MatrixRotation rotation) {
    
    // Check if src and dst are the same. If so, a 180-deg rotate
    // needs a temp buffer, but 0-deg is a no-op and 90/270 work in-place.
    // For simplicity, this function assumes src != dst or rotation != 180.
    // A safe bet is to always use a separate destination buffer.

    switch (rotation) {
        
        case MatrixRotation::ROT_0_CW:
            // No rotation, just copy the data
            // (Assumes src and dst are not the same, or this is a no-op)
            if (src != dst) {
                memcpy(dst, src, sizeof(float) * 64);
            }
            break;

        case MatrixRotation::ROT_90_CW:
            // 90 degrees clockwise: (r, c) -> (c, 7-r)
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    dst[c * 8 + (7 - r)] = src[r * 8 + c];
                }
            }
            break;

        case MatrixRotation::ROT_180_CW:
            // 180 degrees: (r, c) -> (7-r, 7-c)
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    dst[(7 - r) * 8 + (7 - c)] = src[r * 8 + c];
                }
            }
            break;

        case MatrixRotation::ROT_270_CW:
            // 270 degrees clockwise: (r, c) -> (7-c, r)
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    dst[(7 - c) * 8 + r] = src[r * 8 + c];
                }
            }
            break;
    }
}