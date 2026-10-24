// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
// CREST latency/memory-only Arduino sketch with per-run telemetry.
//
// Used by latency-only targets (currently ARDUINO_NANO_RP2040_CONNECT). It
// never touches harness GPIO or INA228 code: there is no energy measurement on
// these targets. Compared with crest_inference_no_energy.ino it:
//   - defers AllocateTensors() until after START so TFLM allocation errors
//     reach the host instead of being printed before the port is open;
//   - discards CREST_WARMUP_RUNS warm-up Invoke() calls;
//   - times each of CREST_INFERENCE_RUNS Invoke() calls individually with
//     micros() and buffers the results, so nothing is printed inside the
//     timed region;
//   - prints one key=value line per run, then the standard CREST summary
//     (`runs:`, placeholder power keys, `timer output:`) for the host parser.
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <cstdlib>
#include <cmath>

#include "model.h"

// Latency-only targets default to 100 timed runs. Host builds override this
// with -DCREST_INFERENCE_RUNS from device.measured_inference_runs.
#ifndef CREST_INFERENCE_RUNS
#define CREST_INFERENCE_RUNS 100
#endif

#include "common/crest_hil_config.h"
#include "common/crest_power.h"

#if CREST_USE_INA228
#error "crest_inference_latency.ino is latency-only; build without CREST_USE_INA228."
#endif

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Deployment constants rewritten by the host before each compile. Keep these
// three #define lines: arduino_base._patch_sketch_constants requires them.
#ifndef CREST_WINDOW_SIZE
#define CREST_WINDOW_SIZE 201
#endif

#ifndef CREST_NUM_CHANNELS
#define CREST_NUM_CHANNELS 64
#endif

#ifndef CREST_TENSOR_ARENA_BYTES
#define CREST_TENSOR_ARENA_BYTES (40 * 1024)
#endif

// Identifiers echoed on every telemetry line. The host rewrites
// CREST_MODEL_ID with the candidate directory name.
#ifndef CREST_BOARD_ID
#define CREST_BOARD_ID "nano_rp2040_connect"
#endif

#ifndef CREST_MODEL_ID
#define CREST_MODEL_ID "unnamed"
#endif

// Untimed Invoke() calls run before the timed window.
#ifndef CREST_WARMUP_RUNS
#define CREST_WARMUP_RUNS 5
#endif

// Log-mel features are computed on the host for UrbanSound8K, so this runner
// does no on-device preprocessing and reports preprocess_us=-1.
#ifndef CREST_PREPROCESS_ON_DEVICE
#define CREST_PREPROCESS_ON_DEVICE 0
#endif

#ifndef CREST_SKIP_SERIAL_WAIT
#define CREST_SKIP_SERIAL_WAIT 0
#endif

namespace {

constexpr int kWindowSize = CREST_WINDOW_SIZE;
constexpr int kNumChannels = CREST_NUM_CHANNELS;
constexpr size_t kTensorArenaSize = CREST_TENSOR_ARENA_BYTES;
constexpr int kTimedRuns = CREST_INFERENCE_RUNS;
constexpr int kWarmupRuns = CREST_WARMUP_RUNS;
constexpr uint32_t kSerialTimeoutMs = 50;
constexpr uint32_t kStartCommandTimeoutMs = 5000;
constexpr long kUnavailable = -1;

static_assert(kTimedRuns > 0, "CREST_INFERENCE_RUNS must be >= 1");
static_assert(kWarmupRuns >= 0, "CREST_WARMUP_RUNS must be >= 0");

// CMSIS-NN requires 16-byte alignment.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
// Per-run Invoke() durations, filled inside the timed window and printed after.
uint32_t invoke_us[kTimedRuns];

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
long arena_used_bytes = kUnavailable;

}  // namespace

bool WaitForStartCommand(uint32_t timeout_ms) {
  const uint32_t start_ms = millis();
  while (millis() - start_ms < timeout_ms) {
    if (!Serial.available()) {
      delay(5);
      continue;
    }
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.equalsIgnoreCase(CREST_CMD_START)) {
      return true;
    }
  }
  return false;
}

// Print the shared prefix of a telemetry line.
void PrintRunPrefix(int run) {
  Serial.print("board=");
  Serial.print(CREST_BOARD_ID);
  Serial.print(" model=");
  Serial.print(CREST_MODEL_ID);
  Serial.print(" run=");
  Serial.print(run);
}

void PrintRunLine(int run, long invoke, long preprocess) {
  PrintRunPrefix(run);
  Serial.print(" invoke_us=");
  Serial.print(invoke);
  Serial.print(" preprocess_us=");
  Serial.print(preprocess);
  Serial.print(" arena_used_bytes=");
  Serial.print(arena_used_bytes);
  Serial.println(" status=ok");
}

// Report an infeasible candidate and park the MCU. No `timer output:` line is
// printed, so the host classifies the attempt as a failure instead of
// recording a latency.
void ReportInfeasibleAndHalt(int run, const char* reason) {
  PrintRunPrefix(run);
  Serial.print(" invoke_us=-1 preprocess_us=-1 arena_used_bytes=");
  Serial.print(arena_used_bytes);
  Serial.print(" status=infeasible reason=");
  Serial.print(reason);
  Serial.print(" arena_bytes=");
  Serial.println(static_cast<unsigned long>(kTensorArenaSize));
  Serial.flush();
  while (true) {
    delay(1000);
  }
}

// Synthetic uniform input in [0, 5), matching crest_inference_no_energy.ino.
bool FillInputTensor() {
  const float scale = input->params.scale;
  const int32_t zero_point = input->params.zero_point;
  for (int sample = 0; sample < kWindowSize; ++sample) {
    for (int channel = 0; channel < kNumChannels; ++channel) {
      const int offset = sample * kNumChannels + channel;
      const float value =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 5.0f;
      if (input->type == kTfLiteFloat32) {
        input->data.f[offset] = value;
      } else if (input->type == kTfLiteInt8) {
        int32_t quantized = static_cast<int32_t>(roundf(value / scale)) + zero_point;
        quantized = min(127, max(-128, quantized));
        input->data.int8[offset] = static_cast<int8_t>(quantized);
      } else {
        return false;
      }
    }
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
#if !CREST_SKIP_SERIAL_WAIT
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif
  delay(1000);

  tflite::InitializeTarget();
  error_reporter = tflite::GetMicroErrorReporter();
  model = tflite::GetModel(g_model);

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, /*resource_variables=*/nullptr,
      /*profiler=*/nullptr);
  interpreter = &static_interpreter;

  Serial.println(CREST_DUT_READY);
  while (!WaitForStartCommand(kStartCommandTimeoutMs)) {
    Serial.println(CREST_DUT_READY);
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ReportInfeasibleAndHalt(0, "schema_mismatch");
  }
  // TFLM prints "Failed to allocate ... missing: N" itself, which the host
  // already maps to HIL_ERROR_UNDER_SIZED with a retry hint.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ReportInfeasibleAndHalt(0, "arena_too_small");
  }
  arena_used_bytes = static_cast<long>(interpreter->arena_used_bytes());

  input = interpreter->input(0);
  const size_t expected_elements = static_cast<size_t>(kWindowSize) * kNumChannels;
  const size_t element_bytes = input->type == kTfLiteFloat32 ? sizeof(float) : 1;
  if (input->bytes < expected_elements * element_bytes) {
    ReportInfeasibleAndHalt(0, "input_shape_mismatch");
  }
  if (!FillInputTensor()) {
    ReportInfeasibleAndHalt(0, "unsupported_input_type");
  }

  for (int i = 0; i < kWarmupRuns; ++i) {
    if (interpreter->Invoke() != kTfLiteOk) {
      ReportInfeasibleAndHalt(0, "warmup_invoke_failed");
    }
  }

  // Timed window: no serial I/O until every run has finished.
  int runs_completed = 0;
  bool invoke_failed = false;
  for (int i = 0; i < kTimedRuns; ++i) {
    const uint32_t start_us = micros();
    const TfLiteStatus status = interpreter->Invoke();
    invoke_us[i] = micros() - start_us;
    if (status != kTfLiteOk) {
      invoke_failed = true;
      break;
    }
    ++runs_completed;
  }

  uint64_t total_us = 0;
  for (int i = 0; i < runs_completed; ++i) {
    PrintRunLine(i + 1, static_cast<long>(invoke_us[i]), kUnavailable);
    total_us += invoke_us[i];
  }
  if (invoke_failed) {
    ReportInfeasibleAndHalt(runs_completed + 1, "invoke_failed");
  }

  // Standard CREST summary so arduino_base's parser works unchanged. Unlike
  // the BLE33 sketch, the mean excludes the warm-up runs.
  const float latency_s =
      (static_cast<float>(total_us) / runs_completed) / 1000000.0f;
  Serial.print("runs: ");
  Serial.println(runs_completed);
  crest_power::EmitPowerTelemetry(1, latency_s, runs_completed,
                                  crest_power::GetIdleBaselinePower(),
                                  crest_power::FlushEnergyWindow());
  Serial.print("timer output: ");
  Serial.println(latency_s, 6);
}

void loop() {
  delay(1000);
}
