// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
// Integer-only log-mel front end matching the course-project training code
// (embedded-systme-project/training/features.py, log_mel):
//   frame 400 / hop 160 at 16 kHz, periodic Hann, 512-point real FFT power
//   spectrum, 64 Slaney mel bins, ln(max(mel, 1e-6)).
//
// Integer pipeline per frame (no FPU needed; Cortex-M0+ friendly):
//   1. y = pcm * hann_q16                (int32; x*w = y / 2^31)
//   2. block scaling: shift y so its peak uses the FFT input range (per frame,
//      so quiet frames keep their precision)
//   3. kissfft real FFT, 16-bit (q15) or 32-bit (q31) fixed point
//      (CREST_LOGMEL_FFT_BITS), output = FFT / 2^CREST_LOGMEL_FFT_GAIN_LOG2
//   4. power = re^2 + im^2, normalized to < 2^40
//   5. mel = sum(power * weight_u16) over each bin's sparse support (uint64)
//   6. ln via a fixed-point log2 (CLZ + 257-entry LUT, Q16), plus the known
//      exponents; floor at ln(1e-6)
// Outputs are natural-log values in Q16 (int32). crest_logmel_quantize() maps
// them to the model's int8 input with round-half-to-even, like
// np.clip(np.rint(x / scale) + zero_point, -128, 127) in the training repo.
//
// PCM convention: int16 samples, float value = pcm / 32768 (soundfile float32
// in [-1, 1)), 16 kHz mono.
//
// Header-only: include it in exactly one translation unit (the sketch). The
// 16-bit FFT comes from TFLM's kiss_fft_int16 (compiled by the library); the
// 32-bit FFT is compiled here from the same kissfft sources.
#pragma once

#include <stdint.h>
#include <string.h>

#include "crest_logmel_tables.h"

#ifndef CREST_LOGMEL_FFT_BITS
#define CREST_LOGMEL_FFT_BITS 16
#endif

#if CREST_LOGMEL_FFT_BITS == 16
#include "tensorflow/lite/experimental/microfrontend/lib/kiss_fft_int16.h"
namespace crest_kiss = kissfft_fixed16;
typedef int16_t crest_fft_in_t;
#define CREST_LOGMEL_FFT_IN_BITS 15
#define CREST_LOGMEL_FFT_STATE_BYTES 4096
#elif CREST_LOGMEL_FFT_BITS == 32
#include "tensorflow/lite/experimental/microfrontend/lib/kiss_fft_common.h"
#define FIXED_POINT 32
namespace kissfft_fixed32 {
#include "third_party/kissfft/kiss_fft.c"
#include "third_party/kissfft/tools/kiss_fftr.c"
}  // namespace kissfft_fixed32
#undef FIXED_POINT
namespace crest_kiss = kissfft_fixed32;
typedef int32_t crest_fft_in_t;
#define CREST_LOGMEL_FFT_IN_BITS 31
#define CREST_LOGMEL_FFT_STATE_BYTES 8192
#else
#error "CREST_LOGMEL_FFT_BITS must be 16 or 32"
#endif

// kissfft fixed-point FFTs scale by 1/2 per radix-2 stage (1/4 per radix-4),
// so a 512-point real FFT returns FFT/512. Verified by the host golden test.
#ifndef CREST_LOGMEL_FFT_GAIN_LOG2
#define CREST_LOGMEL_FFT_GAIN_LOG2 9
#endif

// Power values are shifted below 2^40 before the mel multiply-accumulate so
// sum(power * uint16 weight) over a bin's support cannot overflow uint64.
#define CREST_LOGMEL_POWER_BITS 40
// ln(2) in Q30.
#define CREST_LOGMEL_LN2_Q30 744261118LL

namespace crest_logmel {

static uint8_t g_fft_state[CREST_LOGMEL_FFT_STATE_BYTES] __attribute__((aligned(8)));
static crest_kiss::kiss_fftr_cfg g_fft_cfg = nullptr;
static int32_t g_windowed[CREST_LOGMEL_FRAME_LENGTH];
static crest_fft_in_t g_fft_in[CREST_LOGMEL_FFT_LENGTH];
static crest_kiss::kiss_fft_cpx g_fft_out[CREST_LOGMEL_NUM_BINS];
static uint64_t g_power[CREST_LOGMEL_NUM_BINS];

// Bytes kissfft needs for its state; 0 if g_fft_state is too small.
inline size_t init() {
  size_t needed = 0;
  crest_kiss::kiss_fftr_alloc(CREST_LOGMEL_FFT_LENGTH, 0, nullptr, &needed);
  if (needed > sizeof(g_fft_state)) {
    return 0;
  }
  size_t available = sizeof(g_fft_state);
  g_fft_cfg = crest_kiss::kiss_fftr_alloc(CREST_LOGMEL_FFT_LENGTH, 0, g_fft_state, &available);
  return g_fft_cfg ? needed : 0;
}

inline int bitlen32(uint32_t x) { return x ? 32 - __builtin_clz(x) : 0; }
inline int bitlen64(uint64_t x) { return x ? 64 - __builtin_clzll(x) : 0; }

// log2(x) in Q16 for x > 0 (LUT over the top 8 mantissa bits, linear
// interpolation on the next 16).
inline int32_t log2_q16(uint64_t x) {
  const int n = bitlen64(x) - 1;
  const uint64_t m = x << (63 - n);
  const uint32_t idx = (uint32_t)(m >> (63 - CREST_LOGMEL_LOG2_LUT_BITS)) & ((1u << CREST_LOGMEL_LOG2_LUT_BITS) - 1);
  const uint32_t rem = (uint32_t)(m >> (63 - CREST_LOGMEL_LOG2_LUT_BITS - 16)) & 0xFFFFu;
  const uint32_t a = kCrestLogmelLog2LutQ16[idx];
  const uint32_t b = kCrestLogmelLog2LutQ16[idx + 1];
  return (int32_t)(n * 65536) + (int32_t)a + (int32_t)(((b - a) * rem + 0x8000u) >> 16);
}

// One frame: pcm points at CREST_LOGMEL_FRAME_LENGTH int16 samples;
// out_ln_q16 receives CREST_LOGMEL_NUM_MELS natural-log values in Q16.
inline void frame(const int16_t* pcm, int32_t* out_ln_q16) {
  uint32_t peak = 0;
  for (int i = 0; i < CREST_LOGMEL_FRAME_LENGTH; ++i) {
    const int32_t y = (int32_t)pcm[i] * (int32_t)kCrestLogmelHannQ16[i];
    g_windowed[i] = y;
    const uint32_t mag = y < 0 ? (uint32_t)(-(int64_t)y) : (uint32_t)y;
    if (mag > peak) peak = mag;
  }
  if (peak == 0) {
    for (int m = 0; m < CREST_LOGMEL_NUM_MELS; ++m) out_ln_q16[m] = CREST_LOGMEL_LN_FLOOR_Q16;
    return;
  }
  // Block scaling: FFT input = y * 2^-shift_r, peak just below 2^IN_BITS.
  const int shift_r = bitlen32(peak) - CREST_LOGMEL_FFT_IN_BITS;
  const int32_t in_max = (int32_t)((1UL << CREST_LOGMEL_FFT_IN_BITS) - 1);
  for (int i = 0; i < CREST_LOGMEL_FRAME_LENGTH; ++i) {
    int32_t v;
    if (shift_r > 0) {
      v = (int32_t)(((int64_t)g_windowed[i] + (1LL << (shift_r - 1))) >> shift_r);
      if (v > in_max) v = in_max;
      if (v < -in_max) v = -in_max;
    } else {
      v = g_windowed[i] << (-shift_r);
    }
    g_fft_in[i] = (crest_fft_in_t)v;
  }
  for (int i = CREST_LOGMEL_FRAME_LENGTH; i < CREST_LOGMEL_FFT_LENGTH; ++i) g_fft_in[i] = 0;

  crest_kiss::kiss_fftr(g_fft_cfg, g_fft_in, g_fft_out);

  uint64_t power_peak = 0;
  for (int k = 0; k < CREST_LOGMEL_NUM_BINS; ++k) {
    const int64_t re = g_fft_out[k].r;
    const int64_t im = g_fft_out[k].i;
    const uint64_t p = (uint64_t)(re * re) + (uint64_t)(im * im);
    g_power[k] = p;
    if (p > power_peak) power_peak = p;
  }
  int shift_p = bitlen64(power_peak) - CREST_LOGMEL_POWER_BITS;
  if (shift_p < 0) shift_p = 0;

  // mel_true = acc * mel_scale * 2^shift_p * 2^(2 * (gain + shift_r - 31))
  const int32_t exponent_q16 =
      CREST_LOGMEL_LOG2_MEL_SCALE_Q16 + (int32_t)(shift_p + 2 * (CREST_LOGMEL_FFT_GAIN_LOG2 + shift_r - 31)) * 65536;
  for (int m = 0; m < CREST_LOGMEL_NUM_MELS; ++m) {
    const int start = kCrestLogmelMelStart[m];
    const int length = kCrestLogmelMelLength[m];
    const uint16_t* weights = &kCrestLogmelMelWeight[kCrestLogmelMelOffset[m]];
    uint64_t acc = 0;
    for (int j = 0; j < length; ++j) {
      acc += (g_power[start + j] >> shift_p) * (uint64_t)weights[j];
    }
    if (acc == 0) {
      out_ln_q16[m] = CREST_LOGMEL_LN_FLOOR_Q16;
      continue;
    }
    const int32_t log2_total = log2_q16(acc) + exponent_q16;
    // ln = log2 * ln(2), rounded to nearest in Q16.
    const int64_t ln_q46 = (int64_t)log2_total * CREST_LOGMEL_LN2_Q30;
    int32_t ln_q16 = (int32_t)((ln_q46 + (1LL << 29)) >> 30);
    if (ln_q16 < CREST_LOGMEL_LN_FLOOR_Q16) ln_q16 = CREST_LOGMEL_LN_FLOOR_Q16;
    out_ln_q16[m] = ln_q16;
  }
}

// int8 = clamp(rint(ln / scale) + zero_point), round half to even, with
// inv_scale_q24 = round(2^24 / scale).
inline int8_t quantize(int32_t ln_q16, int32_t inv_scale_q24, int32_t zero_point) {
  const int64_t v = (int64_t)ln_q16 * (int64_t)inv_scale_q24;  // Q40
  int64_t q = v >> 40;                                         // floor
  const uint64_t frac = (uint64_t)v & ((1ULL << 40) - 1);
  const uint64_t half = 1ULL << 39;
  if (frac > half || (frac == half && (q & 1))) q += 1;
  q += zero_point;
  if (q > 127) q = 127;
  if (q < -128) q = -128;
  return (int8_t)q;
}

// Number of frames for n samples (no centering).
inline int num_frames(int num_samples) {
  return num_samples < CREST_LOGMEL_FRAME_LENGTH
             ? 0
             : 1 + (num_samples - CREST_LOGMEL_FRAME_LENGTH) / CREST_LOGMEL_HOP_LENGTH;
}

}  // namespace crest_logmel
