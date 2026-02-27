/*
 * Copyright (c) 2026, Lakeworks. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

// AVX-512 intra prediction for 64-wide blocks.
// One ZMM register = one full row of 64 bytes (no loop needed per row).
// Targets AMD Zen 5 with true 512-bit execution units.

#include <immintrin.h>

#include "config/aom_dsp_rtcd.h"

// ===== DC predictor helpers =====

// Sum 64 bytes using SAD against zero.
static inline unsigned int dc_sum_64_avx512(const uint8_t *ref) {
  const __m512i x = _mm512_loadu_si512((const __m512i *)ref);
  const __m512i zero = _mm512_setzero_si512();
  // SAD computes sum of absolute differences in 8-byte groups -> 8 uint64
  const __m512i sad = _mm512_sad_epu8(x, zero);
  // Reduce 8 uint64 values to scalar
  return (unsigned int)_mm512_reduce_add_epi64(sad);
}

// Sum 128 bytes (for DC_128x* using above reference).
static inline unsigned int dc_sum_128_avx512(const uint8_t *ref) {
  return dc_sum_64_avx512(ref) + dc_sum_64_avx512(ref + 64);
}

// Store one 64-byte row.
static inline void row_store_64_avx512(const __m512i *val, uint8_t *dst) {
  _mm512_storeu_si512((__m512i *)dst, *val);
}

// Fill h rows of 64 bytes with the same value.
static inline void dc_store_64xh(const __m512i *row, int h, uint8_t *dst,
                                 ptrdiff_t stride) {
  for (int i = 0; i < h; i++) {
    row_store_64_avx512(row, dst);
    dst += stride;
  }
}

// ===== DC Predictors =====

// DC predictor: average of above (64) + left (h) pixels.
void aom_dc_predictor_64x64_avx512(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above,
                                    const uint8_t *left) {
  const unsigned int sum_above = dc_sum_64_avx512(above);
  const unsigned int sum_left = dc_sum_64_avx512(left);
  const unsigned int sum = sum_above + sum_left;
  const uint8_t dc = (uint8_t)((sum + 64) >> 7);  // (sum + (64+64)/2) / 128
  const __m512i row = _mm512_set1_epi8((char)dc);
  dc_store_64xh(&row, 64, dst, stride);
}

void aom_dc_predictor_64x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above,
                                    const uint8_t *left) {
  // 64 above + 32 left = 96 pixels. Round: (sum + 48) / 96
  // Use multiplication trick: (sum * 171 + 8192) >> 14 ≈ sum / 96
  // Simpler: (sum + 48) / 96
  const __m256i left_data = _mm256_loadu_si256((const __m256i *)left);
  const __m256i zero256 = _mm256_setzero_si256();
  const __m256i sad_left = _mm256_sad_epu8(left_data, zero256);
  const __m128i lo = _mm256_castsi256_si128(sad_left);
  const __m128i hi = _mm256_extracti128_si256(sad_left, 1);
  const __m128i sum_left128 = _mm_add_epi64(lo, hi);
  const unsigned int sum_left =
      (unsigned int)(_mm_cvtsi128_si64(sum_left128) +
                     _mm_extract_epi64(sum_left128, 1));

  const unsigned int sum_above = dc_sum_64_avx512(above);
  const unsigned int sum = sum_above + sum_left;
  const uint8_t dc = (uint8_t)((sum + 48) / 96);
  const __m512i row = _mm512_set1_epi8((char)dc);
  dc_store_64xh(&row, 32, dst, stride);
}

#if !CONFIG_REALTIME_ONLY
void aom_dc_predictor_64x16_avx512(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above,
                                    const uint8_t *left) {
  // 64 above + 16 left = 80 pixels. (sum + 40) / 80
  const __m128i left_data = _mm_loadu_si128((const __m128i *)left);
  const __m128i zero128 = _mm_setzero_si128();
  const __m128i sad_left = _mm_sad_epu8(left_data, zero128);
  const unsigned int sum_left =
      (unsigned int)(_mm_cvtsi128_si64(sad_left) +
                     _mm_extract_epi64(sad_left, 1));

  const unsigned int sum_above = dc_sum_64_avx512(above);
  const unsigned int sum = sum_above + sum_left;
  const uint8_t dc = (uint8_t)((sum + 40) / 80);
  const __m512i row = _mm512_set1_epi8((char)dc);
  dc_store_64xh(&row, 16, dst, stride);
}
#endif  // !CONFIG_REALTIME_ONLY

// DC top (above only)
void aom_dc_top_predictor_64x64_avx512(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above,
                                        const uint8_t *left) {
  (void)left;
  const unsigned int sum = dc_sum_64_avx512(above);
  const uint8_t dc = (uint8_t)((sum + 32) >> 6);
  const __m512i row = _mm512_set1_epi8((char)dc);
  dc_store_64xh(&row, 64, dst, stride);
}

// DC left (left only)
void aom_dc_left_predictor_64x64_avx512(uint8_t *dst, ptrdiff_t stride,
                                         const uint8_t *above,
                                         const uint8_t *left) {
  (void)above;
  const unsigned int sum = dc_sum_64_avx512(left);
  const uint8_t dc = (uint8_t)((sum + 32) >> 6);
  const __m512i row = _mm512_set1_epi8((char)dc);
  dc_store_64xh(&row, 64, dst, stride);
}

// DC 128 (no reference available)
void aom_dc_128_predictor_64x64_avx512(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above,
                                        const uint8_t *left) {
  (void)above;
  (void)left;
  const __m512i row = _mm512_set1_epi8((char)128);
  dc_store_64xh(&row, 64, dst, stride);
}

void aom_dc_128_predictor_64x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above,
                                        const uint8_t *left) {
  (void)above;
  (void)left;
  const __m512i row = _mm512_set1_epi8((char)128);
  dc_store_64xh(&row, 32, dst, stride);
}

#if !CONFIG_REALTIME_ONLY
void aom_dc_128_predictor_64x16_avx512(uint8_t *dst, ptrdiff_t stride,
                                        const uint8_t *above,
                                        const uint8_t *left) {
  (void)above;
  (void)left;
  const __m512i row = _mm512_set1_epi8((char)128);
  dc_store_64xh(&row, 16, dst, stride);
}
#endif  // !CONFIG_REALTIME_ONLY

// ===== V (Vertical) Predictors =====
// Copy the above row to every row.

void aom_v_predictor_64x64_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above,
                                   const uint8_t *left) {
  (void)left;
  const __m512i row = _mm512_loadu_si512((const __m512i *)above);
  dc_store_64xh(&row, 64, dst, stride);
}

void aom_v_predictor_64x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above,
                                   const uint8_t *left) {
  (void)left;
  const __m512i row = _mm512_loadu_si512((const __m512i *)above);
  dc_store_64xh(&row, 32, dst, stride);
}

#if !CONFIG_REALTIME_ONLY
void aom_v_predictor_64x16_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above,
                                   const uint8_t *left) {
  (void)left;
  const __m512i row = _mm512_loadu_si512((const __m512i *)above);
  dc_store_64xh(&row, 16, dst, stride);
}
#endif  // !CONFIG_REALTIME_ONLY

// ===== H (Horizontal) Predictors =====
// Broadcast each left pixel to fill the corresponding row.

void aom_h_predictor_64x64_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above,
                                   const uint8_t *left) {
  (void)above;
  for (int i = 0; i < 64; i++) {
    const __m512i row = _mm512_set1_epi8((char)left[i]);
    _mm512_storeu_si512((__m512i *)dst, row);
    dst += stride;
  }
}

void aom_h_predictor_64x32_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above,
                                   const uint8_t *left) {
  (void)above;
  for (int i = 0; i < 32; i++) {
    const __m512i row = _mm512_set1_epi8((char)left[i]);
    _mm512_storeu_si512((__m512i *)dst, row);
    dst += stride;
  }
}

#if !CONFIG_REALTIME_ONLY
void aom_h_predictor_64x16_avx512(uint8_t *dst, ptrdiff_t stride,
                                   const uint8_t *above,
                                   const uint8_t *left) {
  (void)above;
  for (int i = 0; i < 16; i++) {
    const __m512i row = _mm512_set1_epi8((char)left[i]);
    _mm512_storeu_si512((__m512i *)dst, row);
    dst += stride;
  }
}
#endif  // !CONFIG_REALTIME_ONLY
