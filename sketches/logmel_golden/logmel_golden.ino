// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
// On-device golden test and timing for the integer log-mel front end.
//
// Staged by analysis_scripts/rp2040_logmel/run_logmel_golden_board.py with:
//   common/crest_logmel.h, common/crest_logmel_tables.h  (front end)
//   golden_clips.h                                         (int16 test clips)
// After START it:
//   1. dumps every clip's log-mel: "LM <clip> <frame> <64 x int32 Q16, hex LE>"
//      and int8 input: "Q8 <clip> <frame> <64 x int8, hex>";
//   2. times one frame (one 10 ms hop) over a whole clip:
//      "TIMING frame_us n=.. min=.. median=.. max=..";
//   3. times full windows computed in batch (all frames + int8 quantization)
//      for 1 s (98 frames) and 2 s (198 frames):
//      "TIMING batch_1s_us runs=.. min=.. median=.. max=..";
// then prints DONE. PCM for timing is copied into RAM first (as mic DMA
// would deliver it); the dump reads clips from flash.
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>

#ifndef CREST_LOGMEL_FFT_BITS
#define CREST_LOGMEL_FFT_BITS 32
#endif

#include "common/crest_logmel.h"
#include "golden_clips.h"

#ifndef GOLDEN_TIMING_RUNS
#define GOLDEN_TIMING_RUNS 5
#endif

namespace {

constexpr int kMaxSamples2s = 2 * CREST_LOGMEL_SAMPLE_RATE;
constexpr int kMaxFrames2s = 1 + (kMaxSamples2s - CREST_LOGMEL_FRAME_LENGTH) / CREST_LOGMEL_HOP_LENGTH;

int16_t g_pcm[kMaxSamples2s];
int8_t g_input[kMaxFrames2s * CREST_LOGMEL_NUM_MELS];
int32_t g_ln[CREST_LOGMEL_NUM_MELS];
uint32_t g_samples_us[kMaxFrames2s];

void PrintHex(const uint8_t* data, size_t n) {
  static const char kHex[] = "0123456789abcdef";
  char buf[2 * 4 * CREST_LOGMEL_NUM_MELS + 1];
  for (size_t i = 0; i < n; ++i) {
    buf[2 * i] = kHex[data[i] >> 4];
    buf[2 * i + 1] = kHex[data[i] & 0xF];
  }
  buf[2 * n] = '\0';
  Serial.println(buf);
}

void SortU32(uint32_t* v, int n) {
  for (int i = 1; i < n; ++i) {
    uint32_t x = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > x) {
      v[j + 1] = v[j];
      --j;
    }
    v[j + 1] = x;
  }
}

void PrintStats(const char* label, uint32_t* v, int n, const char* count_key) {
  SortU32(v, n);
  Serial.print("TIMING ");
  Serial.print(label);
  Serial.print(" ");
  Serial.print(count_key);
  Serial.print("=");
  Serial.print(n);
  Serial.print(" min=");
  Serial.print(v[0]);
  Serial.print(" median=");
  Serial.print(n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2);
  Serial.print(" max=");
  Serial.println(v[n - 1]);
}

// All frames of the first num_samples of g_pcm, quantized into g_input.
void BatchWindow(int num_samples) {
  const int frames = crest_logmel::num_frames(num_samples);
  for (int f = 0; f < frames; ++f) {
    crest_logmel::frame(&g_pcm[f * CREST_LOGMEL_HOP_LENGTH], g_ln);
    int8_t* row = &g_input[f * CREST_LOGMEL_NUM_MELS];
    for (int m = 0; m < CREST_LOGMEL_NUM_MELS; ++m) {
      row[m] = crest_logmel::quantize(g_ln[m], GOLDEN_INV_SCALE_Q24, GOLDEN_ZERO_POINT);
    }
  }
}

bool WaitForStart() {
  const uint32_t start = millis();
  while (millis() - start < 5000) {
    if (Serial.available()) {
      String line = Serial.readStringUntil('\n');
      line.trim();
      if (line.equalsIgnoreCase("START")) return true;
    }
    delay(5);
  }
  return false;
}

}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  while (!Serial && millis() < 2000) {
    delay(10);
  }
  delay(500);
  do {
    Serial.println("DUT READY");
  } while (!WaitForStart());

  const size_t state_bytes = crest_logmel::init();
  Serial.print("INFO fft_bits=");
  Serial.print(CREST_LOGMEL_FFT_BITS);
  Serial.print(" fft_state_bytes=");
  Serial.print(state_bytes);
  Serial.print(" clips=");
  Serial.println(GOLDEN_NUM_CLIPS);
  if (state_bytes == 0) {
    Serial.println("ERROR fft state buffer too small");
    return;
  }

  // 1. Dump.
  for (int c = 0; c < GOLDEN_NUM_CLIPS; ++c) {
    const int frames = crest_logmel::num_frames(kGoldenClipLength[c]);
    for (int f = 0; f < frames; ++f) {
      crest_logmel::frame(&kGoldenClips[c][f * CREST_LOGMEL_HOP_LENGTH], g_ln);
      int8_t q[CREST_LOGMEL_NUM_MELS];
      for (int m = 0; m < CREST_LOGMEL_NUM_MELS; ++m) {
        q[m] = crest_logmel::quantize(g_ln[m], GOLDEN_INV_SCALE_Q24, GOLDEN_ZERO_POINT);
      }
      Serial.print("LM ");
      Serial.print(kGoldenClipName[c]);
      Serial.print(" ");
      Serial.print(f);
      Serial.print(" ");
      PrintHex(reinterpret_cast<const uint8_t*>(g_ln), sizeof(g_ln));
      Serial.print("Q8 ");
      Serial.print(kGoldenClipName[c]);
      Serial.print(" ");
      Serial.print(f);
      Serial.print(" ");
      PrintHex(reinterpret_cast<const uint8_t*>(q), sizeof(q));
    }
  }

  // RAM PCM for timing: the last clip (a real or noise clip) tiled to 2 s.
  const int src = GOLDEN_NUM_CLIPS - 1;
  for (int i = 0; i < kMaxSamples2s; ++i) {
    g_pcm[i] = kGoldenClips[src][i % kGoldenClipLength[src]];
  }

  // 2. Per frame (one hop): every frame of a 1 s window, timed individually.
  const int frames_1s = crest_logmel::num_frames(CREST_LOGMEL_SAMPLE_RATE);
  for (int f = 0; f < frames_1s; ++f) {
    const uint32_t t0 = micros();
    crest_logmel::frame(&g_pcm[f * CREST_LOGMEL_HOP_LENGTH], g_ln);
    g_samples_us[f] = micros() - t0;
  }
  PrintStats("frame_us", g_samples_us, frames_1s, "n");

  // 3. Batch windows.
  const int window_samples[2] = {CREST_LOGMEL_SAMPLE_RATE, kMaxSamples2s};
  const char* labels[2] = {"batch_1s_us", "batch_2s_us"};
  for (int w = 0; w < 2; ++w) {
    uint32_t runs[GOLDEN_TIMING_RUNS];
    for (int r = 0; r < GOLDEN_TIMING_RUNS; ++r) {
      const uint32_t t0 = micros();
      BatchWindow(window_samples[w]);
      runs[r] = micros() - t0;
    }
    PrintStats(labels[w], runs, GOLDEN_TIMING_RUNS, "runs");
  }
  Serial.println("DONE");
}

void loop() {
  delay(1000);
}
