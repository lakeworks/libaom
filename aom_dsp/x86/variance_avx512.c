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

// AVX-512 variance functions for 64+ wide blocks.
// Processes 64 bytes per call (2x wider than AVX2's 32-byte kernel).
// Targets AMD Zen 5 with true 512-bit execution units.

#include <immintrin.h>

#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/x86/synonyms.h"

// Core variance kernel: processes 64 bytes (one ZMM register width).
// Computes per-element difference (src - ref) and accumulates:
//   sum += diff
//   sse += diff * diff
static inline void variance_kernel_avx512(const __m512i src, const __m512i ref,
                                          __m512i *const sse,
                                          __m512i *const sum) {
  const __m512i adj_sub = _mm512_set1_epi16((short)0xff01);  // (1, -1)

  // Interleave src and ref bytes for maddubs
  const __m512i src_ref0 = _mm512_unpacklo_epi8(src, ref);
  const __m512i src_ref1 = _mm512_unpackhi_epi8(src, ref);

  // Compute diff = src * 1 + ref * (-1) = src - ref (as signed int16)
  const __m512i diff0 = _mm512_maddubs_epi16(src_ref0, adj_sub);
  const __m512i diff1 = _mm512_maddubs_epi16(src_ref1, adj_sub);

  // Compute diff^2 as int32
  const __m512i madd0 = _mm512_madd_epi16(diff0, diff0);
  const __m512i madd1 = _mm512_madd_epi16(diff1, diff1);

  // Accumulate (sum is int16, sse is int32)
  *sum = _mm512_add_epi16(*sum, _mm512_add_epi16(diff0, diff1));
  *sse = _mm512_add_epi32(*sse, _mm512_add_epi32(madd0, madd1));
}

// Reduce 512-bit sum (int16) to scalar int32.
// For blocks <= 2048 pixels, int16 accumulators won't overflow.
static inline int sum_reduce_i16_avx512(const __m512i vsum) {
  // Widen int16 -> int32 in two halves, then add
  const __m256i lo = _mm512_castsi512_si256(vsum);
  const __m256i hi = _mm512_extracti64x4_epi64(vsum, 1);
  const __m256i sum256 = _mm256_add_epi16(lo, hi);

  const __m128i sum128 = _mm_add_epi16(_mm256_castsi256_si128(sum256),
                                       _mm256_extracti128_si256(sum256, 1));
  const __m128i sum64 = _mm_add_epi16(sum128, _mm_srli_si128(sum128, 8));
  // Sign-extend int16 elements to int32 for final sum
  const __m128i sum32 = _mm_cvtepi16_epi32(sum64);
  const __m128i sum_hi = _mm_srli_si128(sum32, 8);
  const __m128i sum_all = _mm_add_epi32(sum32, sum_hi);
  const __m128i sum_final =
      _mm_add_epi32(sum_all, _mm_srli_si128(sum_all, 4));
  return _mm_cvtsi128_si32(sum_final);
}

// Reduce 512-bit sse (int32) to scalar uint32.
static inline unsigned int sse_reduce_i32_avx512(const __m512i vsse) {
  const __m256i lo = _mm512_castsi512_si256(vsse);
  const __m256i hi = _mm512_extracti64x4_epi64(vsse, 1);
  const __m256i sum256 = _mm256_add_epi32(lo, hi);

  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256),
                                       _mm256_extracti128_si256(sum256, 1));
  const __m128i sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
  const __m128i sum_final =
      _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
  return (unsigned int)_mm_cvtsi128_si32(sum_final);
}

// For large blocks (>2048 pixels), accumulate sum in int32.
static inline __m512i sum_to_32bit_avx512(const __m512i sum) {
  const __m256i lo = _mm512_castsi512_si256(sum);
  const __m256i hi = _mm512_extracti64x4_epi64(sum, 1);
  const __m256i sum_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo));
  const __m256i sum_lo2 =
      _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo, 1));
  const __m256i sum_hi = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi));
  const __m256i sum_hi2 =
      _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi, 1));
  const __m256i a = _mm256_add_epi32(sum_lo, sum_lo2);
  const __m256i b = _mm256_add_epi32(sum_hi, sum_hi2);
  const __m256i c = _mm256_add_epi32(a, b);
  return _mm512_castsi256_si512(c);
}

// Reduce int32 sum (in low 256 bits of zmm) to scalar
static inline int sum_reduce_i32_to_scalar(const __m512i vsum32) {
  const __m256i v = _mm512_castsi512_si256(vsum32);
  const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v),
                                       _mm256_extracti128_si256(v, 1));
  const __m128i sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
  const __m128i sum_final = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
  return _mm_cvtsi128_si32(sum_final);
}

// Process one row of 64 bytes.
static inline void variance64_kernel_avx512(const uint8_t *const src,
                                            const uint8_t *const ref,
                                            __m512i *const sse,
                                            __m512i *const sum) {
  const __m512i s = _mm512_loadu_si512((const __m512i *)src);
  const __m512i r = _mm512_loadu_si512((const __m512i *)ref);
  variance_kernel_avx512(s, r, sse, sum);
}

// Process one row of 128 bytes (2x ZMM loads).
static inline void variance128_kernel_avx512(const uint8_t *const src,
                                             const uint8_t *const ref,
                                             __m512i *const sse,
                                             __m512i *const sum) {
  variance64_kernel_avx512(src, ref, sse, sum);
  variance64_kernel_avx512(src + 64, ref + 64, sse, sum);
}

// ===== 64-wide variance (no loop needed for small heights) =====
static inline void variance64_avx512(const uint8_t *src, const int src_stride,
                                     const uint8_t *ref, const int ref_stride,
                                     const int h, __m512i *const vsse,
                                     __m512i *const vsum) {
  *vsum = _mm512_setzero_si512();

  for (int i = 0; i < h; i++) {
    variance64_kernel_avx512(src, ref, vsse, vsum);
    src += src_stride;
    ref += ref_stride;
  }
}

// ===== 128-wide variance =====
static inline void variance128_avx512(const uint8_t *src, const int src_stride,
                                      const uint8_t *ref, const int ref_stride,
                                      const int h, __m512i *const vsse,
                                      __m512i *const vsum) {
  *vsum = _mm512_setzero_si512();

  for (int i = 0; i < h; i++) {
    variance128_kernel_avx512(src, ref, vsse, vsum);
    src += src_stride;
    ref += ref_stride;
  }
}

// Variance for blocks where int16 sum accumulators won't overflow
// (pixels <= 2048 = 64 * 32, which covers most cases).
#define AOM_VAR_NO_LOOP_AVX512(bw, bh, bits)                                  \
  unsigned int aom_variance##bw##x##bh##_avx512(                              \
      const uint8_t *src, int src_stride, const uint8_t *ref,                 \
      int ref_stride, unsigned int *sse) {                                    \
    __m512i vsse = _mm512_setzero_si512();                                    \
    __m512i vsum;                                                             \
    variance##bw##_avx512(src, src_stride, ref, ref_stride, bh, &vsse,        \
                          &vsum);                                             \
    *sse = sse_reduce_i32_avx512(vsse);                                       \
    const int sum = sum_reduce_i16_avx512(vsum);                              \
    return *sse - (uint32_t)(((int64_t)sum * sum) >> bits);                   \
  }

// 64-wide: 64xH (H <= 32 for int16 safety)
AOM_VAR_NO_LOOP_AVX512(64, 32, 11)

#if !CONFIG_REALTIME_ONLY
AOM_VAR_NO_LOOP_AVX512(64, 16, 10)
#endif

// Variance for larger blocks that need periodic int32 accumulation.
// Process 'uh' rows at a time in int16, then widen to int32.
#define AOM_VAR_LOOP_AVX512(bw, bh, bits, uh)                                 \
  unsigned int aom_variance##bw##x##bh##_avx512(                              \
      const uint8_t *src, int src_stride, const uint8_t *ref,                 \
      int ref_stride, unsigned int *sse) {                                    \
    __m512i vsse = _mm512_setzero_si512();                                    \
    __m512i vsum32 = _mm512_setzero_si512();                                  \
    for (int i = 0; i < (bh / uh); i++) {                                     \
      __m512i vsum16;                                                         \
      variance##bw##_avx512(src, src_stride, ref, ref_stride, uh, &vsse,      \
                            &vsum16);                                         \
      const __m512i chunk32 = sum_to_32bit_avx512(vsum16);                    \
      const __m256i chunk =                                                   \
          _mm256_add_epi32(_mm512_castsi512_si256(vsum32),                    \
                           _mm512_castsi512_si256(chunk32));                  \
      vsum32 = _mm512_castsi256_si512(chunk);                                 \
      src += uh * src_stride;                                                 \
      ref += uh * ref_stride;                                                 \
    }                                                                         \
    *sse = sse_reduce_i32_avx512(vsse);                                       \
    const int sum = sum_reduce_i32_to_scalar(vsum32);                         \
    return *sse - (unsigned int)(((int64_t)sum * sum) >> bits);               \
  }

// 64x64, 64x128: process 32 rows at a time (64*32 = 2048 max int16)
AOM_VAR_LOOP_AVX512(64, 64, 12, 32)
AOM_VAR_LOOP_AVX512(64, 128, 13, 32)

// 128x64, 128x128: process 16 rows at a time (128*16 = 2048 max int16)
AOM_VAR_LOOP_AVX512(128, 64, 13, 16)
AOM_VAR_LOOP_AVX512(128, 128, 14, 16)
