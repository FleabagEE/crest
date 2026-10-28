// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
// Host build of sketches/common/crest_logmel.h, for the golden test.
//
// Usage: host_logmel IN.pcm OUT.bin INV_SCALE_Q24 ZERO_POINT
//   IN.pcm  raw little-endian int16 PCM (16 kHz mono)
//   OUT.bin int32 num_frames, int32 ln_q16[num_frames * 64], int8 q[num_frames * 64]
// Built with -DCREST_LOGMEL_FFT_BITS=16 or 32 by golden_logmel.py; the
// integer arithmetic is the same as on the RP2040.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "crest_logmel.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s IN.pcm OUT.bin INV_SCALE_Q24 ZERO_POINT\n", argv[0]);
    return 2;
  }
  std::FILE* in = std::fopen(argv[1], "rb");
  if (!in) return 3;
  std::vector<int16_t> pcm;
  int16_t sample;
  while (std::fread(&sample, sizeof(sample), 1, in) == 1) pcm.push_back(sample);
  std::fclose(in);
  const int32_t inv_scale_q24 = (int32_t)std::strtol(argv[3], nullptr, 10);
  const int32_t zero_point = (int32_t)std::strtol(argv[4], nullptr, 10);
  if (crest_logmel::init() == 0) {
    std::fprintf(stderr, "kissfft state buffer too small\n");
    return 4;
  }
  const int frames = crest_logmel::num_frames((int)pcm.size());
  std::vector<int32_t> ln((size_t)frames * CREST_LOGMEL_NUM_MELS);
  std::vector<int8_t> q(ln.size());
  for (int f = 0; f < frames; ++f) {
    crest_logmel::frame(&pcm[(size_t)f * CREST_LOGMEL_HOP_LENGTH], &ln[(size_t)f * CREST_LOGMEL_NUM_MELS]);
  }
  for (size_t i = 0; i < ln.size(); ++i) q[i] = crest_logmel::quantize(ln[i], inv_scale_q24, zero_point);
  std::FILE* out = std::fopen(argv[2], "wb");
  if (!out) return 5;
  const int32_t n = frames;
  std::fwrite(&n, sizeof(n), 1, out);
  std::fwrite(ln.data(), sizeof(int32_t), ln.size(), out);
  std::fwrite(q.data(), sizeof(int8_t), q.size(), out);
  std::fclose(out);
  return 0;
}
