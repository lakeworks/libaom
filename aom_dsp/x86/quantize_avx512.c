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

// AVX-512 quantization — processes 32 coefficients per iteration (2x AVX2).
// Uses __mmask32 for zero-coefficient detection instead of movemask.
// Targets AMD Zen 5 with true 512-bit execution units.

#include <immintrin.h>
#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"

// NOTE: 512-bit accesses here must stay UNALIGNED. The coefficient buffers
// are allocated with aom_memalign(32, ...) in av1/encoder/context_tree.c,
// which guarantees 32-byte alignment only; _mm512_load_si512 requires 64 and
// faults (#GP) on a 32-mod-64 address. The AVX2 reference is safe only
// because _mm256_load_si256 needs exactly the 32 bytes memalign provides.
// Load 32 tran_low_t (int32) coefficients and pack to int16.
static inline __m512i load_coefficients_avx512(const tran_low_t *coeff_ptr) {
  const __m512i c0 = _mm512_loadu_si512((const __m512i *)coeff_ptr);
  const __m512i c1 = _mm512_loadu_si512((const __m512i *)(coeff_ptr + 16));
  return _mm512_packs_epi32(c0, c1);
}

// Store 32 int16 values as 32 tran_low_t (int32).
static inline void store_coefficients_avx512(__m512i coeff_vals,
                                             tran_low_t *coeff_ptr) {
  const __m512i sign = _mm512_srai_epi16(coeff_vals, 15);
  const __m512i lo = _mm512_unpacklo_epi16(coeff_vals, sign);
  const __m512i hi = _mm512_unpackhi_epi16(coeff_vals, sign);
  _mm512_storeu_si512((__m512i *)coeff_ptr, lo);
  _mm512_storeu_si512((__m512i *)(coeff_ptr + 16), hi);
}

// Build a quantizer-parameter vector matching the packed coefficient layout.
//
// _mm512_packs_epi32 interleaves per 128-bit lane, so the int16 lane -> source
// coefficient map is NOT linear:
//   lanes  0-3 -> coeff[0..3]     lanes  4-7  -> coeff[16..19]
//   lanes  8-11 -> coeff[4..7]    lanes 12-15 -> coeff[20..23]
//   lanes 16-19 -> coeff[8..11]   lanes 20-23 -> coeff[24..27]
//   lanes 24-27 -> coeff[12..15]  lanes 28-31 -> coeff[28..31]
//
// The only DC coefficient is coeff[0], at lane 0. Every other lane is AC.
// param_ptr[0] is the DC value; param_ptr[1..7] are all the same AC value
// (av1_build_quantizer replicates index 1 into 2..7), so a single AC
// broadcast with DC blended into lane 0 is exact.
//
// Do NOT rebuild this by broadcasting a 256-bit pattern: _mm512_broadcast_i64x4
// duplicates param_ptr[0] into lane 16, which is coeff[8] -- an AC coefficient.
static inline __m512i dc_ac_param_vector_avx512(const int16_t *param_ptr) {
  const __m512i ac = _mm512_set1_epi16(param_ptr[1]);
  const __m512i dc = _mm512_set1_epi16(param_ptr[0]);
  return _mm512_mask_blend_epi16((__mmask32)1, ac, dc);
}

// Load quantization parameters into 512-bit registers.
static inline void load_b_values_avx512(
    const int16_t *zbin_ptr, __m512i *zbin, const int16_t *round_ptr,
    __m512i *round, const int16_t *quant_ptr, __m512i *quant,
    const int16_t *dequant_ptr, __m512i *dequant, const int16_t *shift_ptr,
    __m512i *shift, int log_scale) {
  *zbin = dc_ac_param_vector_avx512(zbin_ptr);
  if (log_scale > 0) {
    const __m512i rnd = _mm512_set1_epi16((int16_t)(1 << (log_scale - 1)));
    *zbin = _mm512_add_epi16(*zbin, rnd);
    *zbin = _mm512_srai_epi16(*zbin, log_scale);
  }
  *zbin = _mm512_sub_epi16(*zbin, _mm512_set1_epi16(1));

  *round = dc_ac_param_vector_avx512(round_ptr);
  if (log_scale > 0) {
    const __m512i rnd = _mm512_set1_epi16((int16_t)(1 << (log_scale - 1)));
    *round = _mm512_add_epi16(*round, rnd);
    *round = _mm512_srai_epi16(*round, log_scale);
  }

  *quant = dc_ac_param_vector_avx512(quant_ptr);

  *dequant = dc_ac_param_vector_avx512(dequant_ptr);

  *shift = dc_ac_param_vector_avx512(shift_ptr);
}

// Quantize 32 coefficients at log_scale=0.
static AOM_FORCE_INLINE __m512i quantize_b_logscale0_32(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, __m512i *v_quant, __m512i *v_dequant,
    __m512i *v_round, __m512i *v_zbin, __m512i *v_quant_shift) {
  const __m512i v_coeff = load_coefficients_avx512(coeff_ptr);
  const __m512i v_abs_coeff = _mm512_abs_epi16(v_coeff);
  const __mmask32 v_zbin_mask = _mm512_cmpgt_epi16_mask(v_abs_coeff, *v_zbin);

  if (v_zbin_mask == 0) {
    _mm512_storeu_si512((__m512i *)qcoeff_ptr, _mm512_setzero_si512());
    _mm512_storeu_si512((__m512i *)(qcoeff_ptr + 16), _mm512_setzero_si512());
    _mm512_storeu_si512((__m512i *)dqcoeff_ptr, _mm512_setzero_si512());
    _mm512_storeu_si512((__m512i *)(dqcoeff_ptr + 16), _mm512_setzero_si512());
    return _mm512_setzero_si512();
  }

  // Mask-add round only where abs_coeff > zbin
  const __m512i v_tmp_rnd = _mm512_maskz_adds_epi16(
      v_zbin_mask, v_abs_coeff, *v_round);

  // Quantize: ((tmp * quant >> 16) + tmp) * quant_shift >> 16
  const __m512i v_tmp32_a = _mm512_mulhi_epi16(v_tmp_rnd, *v_quant);
  const __m512i v_tmp32_b = _mm512_add_epi16(v_tmp32_a, v_tmp_rnd);
  const __m512i v_tmp32 = _mm512_mulhi_epi16(v_tmp32_b, *v_quant_shift);

  // Detect non-zero quantized coefficients
  const __mmask32 v_nz_mask =
      _mm512_cmpgt_epi16_mask(v_tmp32, _mm512_setzero_si512());

  // Apply original sign
  const __m512i v_qcoeff =
      _mm512_mask_sub_epi16(v_tmp32, _mm512_cmplt_epi16_mask(v_coeff, _mm512_setzero_si512()),
                            _mm512_setzero_si512(), v_tmp32);

  // Dequantize
  const __m512i v_dqcoeff = _mm512_mullo_epi16(v_qcoeff, *v_dequant);

  store_coefficients_avx512(v_qcoeff, qcoeff_ptr);
  store_coefficients_avx512(v_dqcoeff, dqcoeff_ptr);

  // Return mask as vector for eob tracking
  return _mm512_movm_epi16(v_nz_mask);
}

// Quantize 32 coefficients with log_scale > 0 (32x32, 64x64 blocks).
static AOM_FORCE_INLINE __m512i quantize_b_logscale_32(
    const tran_low_t *coeff_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, __m512i *v_quant, __m512i *v_dequant,
    __m512i *v_round, __m512i *v_zbin, __m512i *v_quant_shift,
    int log_scale) {
  const __m512i v_coeff = load_coefficients_avx512(coeff_ptr);
  const __m512i v_abs_coeff = _mm512_abs_epi16(v_coeff);
  const __mmask32 v_zbin_mask = _mm512_cmpgt_epi16_mask(v_abs_coeff, *v_zbin);

  if (v_zbin_mask == 0) {
    _mm512_storeu_si512((__m512i *)qcoeff_ptr, _mm512_setzero_si512());
    _mm512_storeu_si512((__m512i *)(qcoeff_ptr + 16), _mm512_setzero_si512());
    _mm512_storeu_si512((__m512i *)dqcoeff_ptr, _mm512_setzero_si512());
    _mm512_storeu_si512((__m512i *)(dqcoeff_ptr + 16), _mm512_setzero_si512());
    return _mm512_setzero_si512();
  }

  const __m512i v_tmp_rnd = _mm512_maskz_adds_epi16(
      v_zbin_mask, v_abs_coeff, *v_round);

  const __m512i v_tmp32_a = _mm512_mulhi_epi16(v_tmp_rnd, *v_quant);
  const __m512i v_tmp32_b = _mm512_add_epi16(v_tmp32_a, v_tmp_rnd);

  const __m512i v_tmp32_hi = _mm512_slli_epi16(
      _mm512_mulhi_epi16(v_tmp32_b, *v_quant_shift), log_scale);
  const __m512i v_tmp32_lo = _mm512_srli_epi16(
      _mm512_mullo_epi16(v_tmp32_b, *v_quant_shift), 16 - log_scale);
  const __m512i v_tmp32 = _mm512_or_si512(v_tmp32_hi, v_tmp32_lo);

  const __m512i v_dqcoeff_hi = _mm512_slli_epi16(
      _mm512_mulhi_epi16(v_tmp32, *v_dequant), 16 - log_scale);
  const __m512i v_dqcoeff_lo =
      _mm512_srli_epi16(_mm512_mullo_epi16(v_tmp32, *v_dequant), log_scale);
  const __m512i v_dqcoeff_abs =
      _mm512_or_si512(v_dqcoeff_hi, v_dqcoeff_lo);

  const __mmask32 neg_mask =
      _mm512_cmplt_epi16_mask(v_coeff, _mm512_setzero_si512());
  const __m512i v_dqcoeff = _mm512_mask_sub_epi16(
      v_dqcoeff_abs, neg_mask, _mm512_setzero_si512(), v_dqcoeff_abs);

  const __m512i v_qcoeff = _mm512_mask_sub_epi16(
      v_tmp32, neg_mask, _mm512_setzero_si512(), v_tmp32);

  const __mmask32 v_nz_mask =
      _mm512_cmpgt_epi16_mask(v_tmp32, _mm512_setzero_si512());

  store_coefficients_avx512(v_qcoeff, qcoeff_ptr);
  store_coefficients_avx512(v_dqcoeff, dqcoeff_ptr);

  return _mm512_movm_epi16(v_nz_mask);
}

// Track end-of-block position across 32 coefficients.
// Permute iscan to match _mm512_packs_epi32 lane-crossing order:
// packs interleaves per 128-bit lane: [a0-3,b0-3 | a4-7,b4-7 | ...]
// so register position 4 = coeff[16], position 8 = coeff[4], etc.
// The permutation {0,4,1,5,2,6,3,7} on 64-bit qwords maps linear iscan
// to match the interleaved coefficient positions.
static inline __m512i get_max_lane_eob_avx512(const int16_t *iscan,
                                               __m512i v_eobmax,
                                               __m512i v_mask) {
  const __m512i v_iscan = _mm512_loadu_si512((const __m512i *)iscan);
  const __m512i perm_idx = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  const __m512i v_iscan_perm = _mm512_permutexvar_epi64(perm_idx, v_iscan);
  const __m512i v_iscan_plus1 = _mm512_sub_epi16(v_iscan_perm, v_mask);
  const __m512i v_nz_iscan = _mm512_and_si512(v_iscan_plus1, v_mask);
  return _mm512_max_epi16(v_eobmax, v_nz_iscan);
}

// Reduce eob across all 32 lanes.
static inline int16_t accumulate_eob512(__m512i eob512) {
  const __m256i lo = _mm512_castsi512_si256(eob512);
  const __m256i hi = _mm512_extracti64x4_epi64(eob512, 1);
  __m256i eob256 = _mm256_max_epi16(lo, hi);
  const __m128i eob_lo = _mm256_castsi256_si128(eob256);
  const __m128i eob_hi = _mm256_extracti128_si256(eob256, 1);
  __m128i eob = _mm_max_epi16(eob_lo, eob_hi);
  __m128i eob_shuffled = _mm_shuffle_epi32(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0xe);
  eob = _mm_max_epi16(eob, eob_shuffled);
  eob_shuffled = _mm_shufflelo_epi16(eob, 0x1);
  eob = _mm_max_epi16(eob, eob_shuffled);
  return _mm_extract_epi16(eob, 1);
}

void aom_quantize_b_avx512(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                            const int16_t *zbin_ptr, const int16_t *round_ptr,
                            const int16_t *quant_ptr,
                            const int16_t *quant_shift_ptr,
                            tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,
                            const int16_t *dequant_ptr, uint16_t *eob_ptr,
                            const int16_t *scan, const int16_t *iscan) {
  // The vector prologue below unconditionally processes 32 coefficients.
  // TX_4X4 has n_coeffs == 16 (av1_get_max_eob, blockd.h), which would
  // overrun qcoeff/dqcoeff by 64 bytes. Defer sub-vector block sizes to the
  // 256-bit implementation, whose prologue is 16 wide.
  if (n_coeffs < 32) {
    aom_quantize_b_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr, quant_ptr,
                        quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr, dequant_ptr,
                        eob_ptr, scan, iscan);
    return;
  }
  (void)scan;
  __m512i v_zbin, v_round, v_quant, v_dequant, v_quant_shift;
  __m512i v_eobmax = _mm512_setzero_si512();

  load_b_values_avx512(zbin_ptr, &v_zbin, round_ptr, &v_round, quant_ptr,
                       &v_quant, dequant_ptr, &v_dequant, quant_shift_ptr,
                       &v_quant_shift, 0);

  // Process first 32 coefficients (DC + 31 AC)
  __m512i v_nz_mask =
      quantize_b_logscale0_32(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant,
                              &v_dequant, &v_round, &v_zbin, &v_quant_shift);
  v_eobmax = get_max_lane_eob_avx512(iscan, v_eobmax, v_nz_mask);

  // After first iteration, broadcast AC values (drop DC-specific values)
  v_round = _mm512_unpackhi_epi64(v_round, v_round);
  v_quant = _mm512_unpackhi_epi64(v_quant, v_quant);
  v_dequant = _mm512_unpackhi_epi64(v_dequant, v_dequant);
  v_quant_shift = _mm512_unpackhi_epi64(v_quant_shift, v_quant_shift);
  v_zbin = _mm512_unpackhi_epi64(v_zbin, v_zbin);

  // Process remaining coefficients, 32 at a time
  for (intptr_t count = n_coeffs - 32; count > 0; count -= 32) {
    coeff_ptr += 32;
    qcoeff_ptr += 32;
    dqcoeff_ptr += 32;
    iscan += 32;
    v_nz_mask =
        quantize_b_logscale0_32(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant,
                                &v_dequant, &v_round, &v_zbin, &v_quant_shift);
    v_eobmax = get_max_lane_eob_avx512(iscan, v_eobmax, v_nz_mask);
  }

  *eob_ptr = accumulate_eob512(v_eobmax);
}

static AOM_FORCE_INLINE void quantize_b_no_qmatrix_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *iscan, int log_scale) {
  __m512i v_zbin, v_round, v_quant, v_dequant, v_quant_shift;
  __m512i v_eobmax = _mm512_setzero_si512();

  load_b_values_avx512(zbin_ptr, &v_zbin, round_ptr, &v_round, quant_ptr,
                       &v_quant, dequant_ptr, &v_dequant, quant_shift_ptr,
                       &v_quant_shift, log_scale);

  __m512i v_nz_mask = quantize_b_logscale_32(
      coeff_ptr, qcoeff_ptr, dqcoeff_ptr, &v_quant, &v_dequant, &v_round,
      &v_zbin, &v_quant_shift, log_scale);
  v_eobmax = get_max_lane_eob_avx512(iscan, v_eobmax, v_nz_mask);

  v_round = _mm512_unpackhi_epi64(v_round, v_round);
  v_quant = _mm512_unpackhi_epi64(v_quant, v_quant);
  v_dequant = _mm512_unpackhi_epi64(v_dequant, v_dequant);
  v_quant_shift = _mm512_unpackhi_epi64(v_quant_shift, v_quant_shift);
  v_zbin = _mm512_unpackhi_epi64(v_zbin, v_zbin);

  for (intptr_t count = n_coeffs - 32; count > 0; count -= 32) {
    coeff_ptr += 32;
    qcoeff_ptr += 32;
    dqcoeff_ptr += 32;
    iscan += 32;
    v_nz_mask = quantize_b_logscale_32(coeff_ptr, qcoeff_ptr, dqcoeff_ptr,
                                       &v_quant, &v_dequant, &v_round, &v_zbin,
                                       &v_quant_shift, log_scale);
    v_eobmax = get_max_lane_eob_avx512(iscan, v_eobmax, v_nz_mask);
  }

  *eob_ptr = accumulate_eob512(v_eobmax);
}

void aom_quantize_b_32x32_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan) {
  // See aom_quantize_b_avx512: the shared helper prologue is 32 wide.
  if (n_coeffs < 32) {
    aom_quantize_b_32x32_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                               quant_ptr, quant_shift_ptr, qcoeff_ptr,
                               dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);
    return;
  }
  (void)scan;
  quantize_b_no_qmatrix_avx512(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                                quant_ptr, quant_shift_ptr, qcoeff_ptr,
                                dqcoeff_ptr, dequant_ptr, eob_ptr, iscan, 1);
}

void aom_quantize_b_64x64_avx512(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,
    const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan) {
  // See aom_quantize_b_avx512: the shared helper prologue is 32 wide.
  if (n_coeffs < 32) {
    aom_quantize_b_64x64_avx2(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                               quant_ptr, quant_shift_ptr, qcoeff_ptr,
                               dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan);
    return;
  }
  (void)scan;
  quantize_b_no_qmatrix_avx512(coeff_ptr, n_coeffs, zbin_ptr, round_ptr,
                                quant_ptr, quant_shift_ptr, qcoeff_ptr,
                                dqcoeff_ptr, dequant_ptr, eob_ptr, iscan, 2);
}
