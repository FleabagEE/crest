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
#if defined(ARDUINO_ARCH_ESP32)
// ESP32: TFLM comes prebuilt with the esp32 core (esp-tflite-micro, linked
// with ESP-NN's ESP32-S3-optimized kernels). Chirale is deliberately not
// included: its kernels would be linked first and replace ESP-NN.
#include <WiFi.h>
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#else
#include <Chirale_TensorFlowLite.h>
#endif
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

#if defined(ARDUINO_ARCH_ESP32)
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#else
#include "tensorflow/lite/micro/all_ops_resolver.h"
#endif
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
#if defined(ARDUINO_ARCH_ESP32)
#define CREST_BOARD_ID "xiao_esp32s3"
#else
#define CREST_BOARD_ID "nano_rp2040_connect"
#endif
#endif

// ESP32 only: place the tensor arena in PSRAM (heap_caps) instead of a
// static array in internal SRAM. Needs the board's PSRAM enabled.
#ifndef CREST_ARENA_IN_PSRAM
#define CREST_ARENA_IN_PSRAM 0
#endif
#if CREST_ARENA_IN_PSRAM && !defined(ARDUINO_ARCH_ESP32)
#error "CREST_ARENA_IN_PSRAM is only supported on ESP32 boards."
#endif

// ESP32 only: CPU clock set in setup() and reported in the env line.
#ifndef CREST_ESP32_CPU_MHZ
#define CREST_ESP32_CPU_MHZ 240
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

// Hold the on-board NINA-W102 Wi-Fi module in reset (RESET_N low) so its
// power state is defined instead of depending on pull resistors. Only the
// Nano RP2040 Connect has this module (and its NINA_RESETN pin).
#ifndef CREST_HOLD_NINA_RESET
#define CREST_HOLD_NINA_RESET 0
#endif

// Power-measurement mode (replaces the latency bench; never returns):
//   0 = off (latency bench, default)
//   1 = continuous inference: back-to-back Invoke() during every hold
//   2 = listening (P_listen in the energy model): the on-board microphone
//       captures 16 kHz mono PCM into a buffer, the CPU is otherwise idle,
//       no inference. RP2040: MP34DT06J via the PDM library (PIO/DMA capture,
//       PDM->PCM filter in its interrupt handler). ESP32-S3 Sense: the
//       expansion board's PDM mic via ESP_I2S PDM RX (clock GPIO42, data
//       GPIO41; hardware PDM->PCM, the CPU blocks on DMA).
//   3 = idle with the microphone off: CPU idle (delay(), i.e. RTOS idle /
//       WFI / WAITI), for the cost of the microphone itself (2 minus 3)
// After CREST_POWER_SETTLE_S, holds of CREST_POWER_HOLD_S repeat forever so a
// USB meter can be read several times. Each hold starts with one LED blink
// and a POWER_STATE_BEGIN line, and ends with a POWER_STATE_END line and two
// blinks. The LED is off during the hold so it does not add to the reading.
// No host handshake is needed.
#ifndef CREST_POWER_STATE
#define CREST_POWER_STATE 0
#endif
#ifndef CREST_POWER_HOLD_S
#define CREST_POWER_HOLD_S 60
#endif
#ifndef CREST_POWER_SETTLE_S
#define CREST_POWER_SETTLE_S 10
#endif
#if CREST_POWER_STATE < 0 || CREST_POWER_STATE > 3
#error "CREST_POWER_STATE must be 0, 1, 2 or 3"
#endif
#if CREST_POWER_STATE == 2
#if defined(ARDUINO_ARCH_ESP32)
#include <ESP_I2S.h>
#define CREST_MIC_PDM_CLK_PIN 42
#define CREST_MIC_PDM_DATA_PIN 41
#else
#include <PDM.h>
#endif
#define CREST_MIC_SAMPLE_RATE 16000
#endif
#if defined(ARDUINO_XIAO_ESP32S3)
// The XIAO ESP32-S3 user LED (GPIO21) is active-low.
#define CREST_LED_ON LOW
#define CREST_LED_OFF HIGH
#else
#define CREST_LED_ON HIGH
#define CREST_LED_OFF LOW
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
#if CREST_ARENA_IN_PSRAM
uint8_t* tensor_arena = nullptr;  // allocated from PSRAM in setup()
#else
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
#endif
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
        quantized = quantized > 127 ? 127 : (quantized < -128 ? -128 : quantized);
        input->data.int8[offset] = static_cast<int8_t>(quantized);
      } else {
        return false;
      }
    }
  }
  return true;
}

#if CREST_POWER_STATE == 2
// Microphone capture for the listening state. Samples are counted (and the
// last buffer's peak kept) so the END line shows the mic really ran.
int16_t g_mic_buffer[512];
volatile uint32_t g_mic_samples = 0;
volatile int32_t g_mic_peak = 0;

void TrackMicBuffer(int samples) {
  int32_t peak = 0;
  for (int i = 0; i < samples; ++i) {
    const int32_t v = g_mic_buffer[i] < 0 ? -static_cast<int32_t>(g_mic_buffer[i]) : g_mic_buffer[i];
    if (v > peak) peak = v;
  }
  g_mic_peak = peak;
  g_mic_samples += samples;
}

#if defined(ARDUINO_ARCH_ESP32)
I2SClass g_crest_mic_i2s;

bool StartMic() {
  g_crest_mic_i2s.setPinsPdmRx(CREST_MIC_PDM_CLK_PIN, CREST_MIC_PDM_DATA_PIN);
  return g_crest_mic_i2s.begin(I2S_MODE_PDM_RX, CREST_MIC_SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
}

// Blocks on the I2S DMA (CPU idle) for about 32 ms per call.
void CaptureMic() {
  const size_t bytes = g_crest_mic_i2s.readBytes(reinterpret_cast<char*>(g_mic_buffer), sizeof(g_mic_buffer));
  TrackMicBuffer(static_cast<int>(bytes / sizeof(int16_t)));
}
#else
void OnPdmData() {
  int bytes = PDM.available();
  if (bytes > static_cast<int>(sizeof(g_mic_buffer))) bytes = sizeof(g_mic_buffer);
  PDM.read(g_mic_buffer, bytes);
  TrackMicBuffer(bytes / static_cast<int>(sizeof(int16_t)));
}

bool StartMic() {
  PDM.onReceive(OnPdmData);
  return PDM.begin(1, CREST_MIC_SAMPLE_RATE) != 0;
}

// The PDM interrupt fills the buffer; the CPU sleeps in delay().
void CaptureMic() {
  delay(100);
}
#endif
#endif

#if CREST_POWER_STATE
void PowerMarkerBlink(int count) {
  for (int i = 0; i < count; ++i) {
    digitalWrite(LED_BUILTIN, CREST_LED_ON);
    delay(100);
    digitalWrite(LED_BUILTIN, CREST_LED_OFF);
    delay(150);
  }
}

void PrintPowerState(const char* tag, uint32_t cycle) {
  Serial.print(tag);
  Serial.print(" board=");
  Serial.print(CREST_BOARD_ID);
  Serial.print(" cycle=");
  Serial.print(cycle);
  Serial.print(" state=");
  Serial.print(CREST_POWER_STATE == 1 ? "inference" : (CREST_POWER_STATE == 2 ? "listen_mic_on" : "idle_mic_off"));
  Serial.print(" model=");
  Serial.print(CREST_MODEL_ID);
  Serial.print(" hold_s=");
  Serial.print(CREST_POWER_HOLD_S);
  Serial.print(" nina_reset=");
  Serial.print(CREST_HOLD_NINA_RESET);
#if defined(ARDUINO_ARCH_ESP32)
  Serial.print(" cpu_mhz=");
  Serial.print(getCpuFrequencyMhz());
  Serial.print(" wifi_mode=");
  Serial.print(static_cast<int>(WiFi.getMode()));
  Serial.print(" bt_started=");
  Serial.print(btStarted() ? 1 : 0);
#endif
  Serial.print(" t_ms=");
  Serial.print(millis());
}

void RunPowerStateForever() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, CREST_LED_OFF);
#if CREST_POWER_STATE == 2
  if (!StartMic()) {
    while (true) {
      Serial.println("POWER_STATE_ERROR mic_start_failed");
      PowerMarkerBlink(5);
      delay(1000);
    }
  }
#endif
  delay(static_cast<uint32_t>(CREST_POWER_SETTLE_S) * 1000UL);
  const uint32_t hold_ms = static_cast<uint32_t>(CREST_POWER_HOLD_S) * 1000UL;
  for (uint32_t cycle = 1;; ++cycle) {
    PowerMarkerBlink(1);
    PrintPowerState("POWER_STATE_BEGIN", cycle);
    Serial.println();
    Serial.flush();
    const uint32_t start_ms = millis();
    uint32_t invokes = 0;
#if CREST_POWER_STATE == 2
    const uint32_t mic_start = g_mic_samples;
#endif
    while (millis() - start_ms < hold_ms) {
#if CREST_POWER_STATE == 1
      interpreter->Invoke();
      ++invokes;
#elif CREST_POWER_STATE == 2
      CaptureMic();
#else
      delay(1000);
#endif
    }
    const uint32_t elapsed_ms = millis() - start_ms;
    PrintPowerState("POWER_STATE_END", cycle);
    Serial.print(" elapsed_ms=");
    Serial.print(elapsed_ms);
    Serial.print(" invokes=");
    Serial.print(invokes);
    if (invokes > 0) {
      Serial.print(" mean_invoke_us=");
      Serial.print(static_cast<unsigned long>(static_cast<uint64_t>(elapsed_ms) * 1000ULL / invokes));
    }
#if CREST_POWER_STATE == 2
    // Expect about CREST_MIC_SAMPLE_RATE * elapsed_ms / 1000 samples.
    Serial.print(" mic_samples=");
    Serial.print(g_mic_samples - mic_start);
    Serial.print(" mic_rate_hz=");
    Serial.print(static_cast<unsigned long>(static_cast<uint64_t>(g_mic_samples - mic_start) * 1000ULL / elapsed_ms));
    Serial.print(" mic_peak=");
    Serial.print(g_mic_peak);
#endif
    Serial.println();
    Serial.flush();
    PowerMarkerBlink(2);
  }
}
#endif

void setup() {
#if CREST_HOLD_NINA_RESET
#if !defined(ARDUINO_NANO_RP2040_CONNECT)
#error "CREST_HOLD_NINA_RESET is only supported on the Nano RP2040 Connect."
#endif
  pinMode(NINA_RESETN, OUTPUT);
  digitalWrite(NINA_RESETN, LOW);
#endif
  Serial.begin(115200);
  Serial.setTimeout(kSerialTimeoutMs);
#if !CREST_SKIP_SERIAL_WAIT
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif
  delay(1000);

#if defined(ARDUINO_ARCH_ESP32)
  // Radios off for measurement; fixed CPU clock.
  WiFi.mode(WIFI_OFF);
  btStop();
  setCpuFrequencyMhz(CREST_ESP32_CPU_MHZ);
#endif
#if CREST_ARENA_IN_PSRAM
  tensor_arena = static_cast<uint8_t*>(
      heap_caps_aligned_alloc(16, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
#endif

  tflite::InitializeTarget();
  error_reporter = tflite::GetMicroErrorReporter();
  model = tflite::GetModel(g_model);

#if defined(ARDUINO_ARCH_ESP32)
  // Every op used by the CREST DS-CNN exports (no AllOpsResolver in this TFLM).
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddPack();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddStridedSlice();
#else
  static tflite::AllOpsResolver resolver;
#endif
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, /*resource_variables=*/nullptr,
      /*profiler=*/nullptr);
  interpreter = &static_interpreter;

#if !CREST_POWER_STATE
  Serial.println(CREST_DUT_READY);
  while (!WaitForStartCommand(kStartCommandTimeoutMs)) {
    Serial.println(CREST_DUT_READY);
  }
#endif

#if CREST_ARENA_IN_PSRAM
  if (tensor_arena == nullptr) {
    ReportInfeasibleAndHalt(0, "psram_alloc_failed");
  }
#endif
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

#if CREST_POWER_STATE
  RunPowerStateForever();
#endif

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
#if defined(ARDUINO_ARCH_ESP32)
  // Measurement environment (after the timed window): clock, radios, and
  // where the arena actually lives (address range checked by the IDF).
  Serial.print("env cpu_mhz=");
  Serial.print(getCpuFrequencyMhz());
  Serial.print(" wifi_mode=");
  Serial.print(static_cast<int>(WiFi.getMode()));
  Serial.print(" bt_started=");
  Serial.print(btStarted() ? 1 : 0);
  Serial.print(" arena_memory=");
  Serial.print(esp_ptr_external_ram(tensor_arena) ? "psram" : "internal_sram");
  Serial.print(" arena_addr=0x");
  Serial.print(reinterpret_cast<uintptr_t>(tensor_arena), HEX);
  Serial.print(" psram_bytes=");
  Serial.print(ESP.getPsramSize());
  Serial.print(" free_internal_bytes=");
  Serial.println(static_cast<unsigned long>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
#endif
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
