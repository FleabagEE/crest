// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
#include "crest_dut_runner.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "network.h"
#include "network_data.h"
#include "stm32n6xx_hal_rtc.h"
#include "stm32n6xx_hal_rtc_ex.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "crest_dut_phase_config.h"

/*
 * This module owns the DUT-side HIL protocol for the CREST LRUN app.
 * It performs three jobs:
 *   - bootstraps the generated ST Edge AI network and optional external weights
 *   - speaks the host-visible START/READY/result text protocol over debug COM
 *   - runs either back-to-back or cadenced measurement loops and emits timing
 *     telemetry in a parseable format for the HIL harness
 */

AI_ALIGNED(4)
static ai_u8 s_network_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

enum
{
  kStartCommandTimeoutMs = 30000,
  kStartReadyHeartbeatMs = 5000,
  kUartPollTimeoutMs = 20,
  kArmHoldMs = 600,
  kRtcTimebaseHz = 32768,
  kRtcSyncPrediv = kRtcTimebaseHz - 1,
  kWakeupDivider = 16,
};

/* Memory-mapped external-weight address range: inclusive start, exclusive end. */
static const uintptr_t kExternalWeightsStart = 0x70000000UL;
static const uintptr_t kExternalWeightsEnd = 0x80000000UL;

/* PD0 is high during the harness measurement window. */
#define CREST_DUT_TRIGGER_GPIO_PORT   GPIOD
#define CREST_DUT_TRIGGER_GPIO_PIN    GPIO_PIN_0
/* PE9 arms the harness when low; high is the idle/disarmed state. */
#define CREST_DUT_ARM_GPIO_PORT       GPIOE
#define CREST_DUT_ARM_GPIO_PIN        GPIO_PIN_9

/* Telemetry label for system Stop entered with the WFI instruction. */
#define CREST_DUT_STOP_MODE_VARIANT "system_stop_wfi"
/* RTC windows, wrap-safe DWT cadenced active timing, and an awake final tail. */
#define CREST_DUT_MEASUREMENT_PROTOCOL "rtc_wall_v4_dwt_active_stop_evidence_awake_tail"

typedef struct
{
  uint64_t timer_per_inference_scaled_1e6;
  uint64_t timer_per_window_scaled_1e6;
  uint64_t dwt_cycles_per_inference_milli;
  uint64_t rtc_sleep_total_us;
  int64_t wake_recovery_total_us;
  int64_t wake_overshoot_total_us;
  uint32_t clock_hz;
  uint32_t deadline_miss_count;
  uint32_t slept_slots;
  uint32_t stop_entries;
  uint32_t runs_completed;
  bool have_timing;
  bool have_cycles;
} CrestDutMeasurement;

static RTC_HandleTypeDef s_rtc_handle;
static volatile bool s_rtc_wakeup_fired = false;
static const char *s_rtc_clock_source = "UNINITIALIZED";
static uint32_t s_rtc_clock_hz_nominal = 0U;
static const char *s_cadence_timing_quality = "unknown";
static bool s_external_weight_mapping_initialized = false;

void SystemClock_Config(void);

/* Convert a requested stop-sleep interval into the RTC wakeup counter domain. */
static uint32_t wakeup_counter_for_sleep_us(uint32_t sleep_us)
{
  uint64_t rtc_clock_hz = (uint64_t)s_rtc_clock_hz_nominal;
  uint64_t counter = 0ULL;

  if (rtc_clock_hz == 0ULL)
  {
    return 0U;
  }

  counter = ((uint64_t)sleep_us * rtc_clock_hz) / ((uint64_t)kWakeupDivider * 1000000ULL);
  if (counter > 0xFFFFU)
  {
    counter = 0xFFFFU;
  }

  return (uint32_t)counter;
}

/* All host-facing protocol output is line-oriented and flushed immediately. */
static void flush_stdout(void)
{
  fflush(stdout);
}

static void emit_line(const char *line)
{
  printf("%s\r\n", line);
  flush_stdout();
}

/* Emit the HIL harness telemetry fields in the exact textual format it parses. */
static void emit_float_scaled_line(const char *label, uint64_t scaled_1e6, bool valid)
{
  if (!valid)
  {
    printf("%s: -1\r\n", label);
    flush_stdout();
    return;
  }

  printf("%s: %lu.%06lu\r\n",
         label,
         (unsigned long)(scaled_1e6 / 1000000ULL),
         (unsigned long)(scaled_1e6 % 1000000ULL));
  flush_stdout();
}

static void emit_signed_us_line(const char *label, int64_t value_us, bool valid)
{
  if (!valid)
  {
    printf("%s: -1\r\n", label);
    flush_stdout();
    return;
  }

  printf("%s: %ld\r\n", label, (long)value_us);
  flush_stdout();
}

static void emit_cpu_clock_telemetry(uint32_t cpu_clock_hz)
{
  printf("clock hz output: %lu\r\n", (unsigned long)cpu_clock_hz);
  flush_stdout();
}

static void emit_cycle_telemetry(uint64_t cycles_per_inference_milli, bool valid)
{
  if (!valid)
  {
    emit_line("dwt cycles per inference output: -1");
    return;
  }

  printf("dwt cycles per inference output: %lu.%03lu\r\n",
         (unsigned long)(cycles_per_inference_milli / 1000ULL),
         (unsigned long)(cycles_per_inference_milli % 1000ULL));
  flush_stdout();
}

static uintptr_t generated_weights_address(void)
{
  return (uintptr_t)ai_network_data_weights_get();
}

static bool weights_use_external_flash(void)
{
  uintptr_t weights_addr = generated_weights_address();
  return (weights_addr >= kExternalWeightsStart) && (weights_addr < kExternalWeightsEnd);
}

/* Bring up XSPI2 only when ST Edge AI placed weights in the external-flash map. */
static bool configure_external_weight_xspi2_clocking(void)
{
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  HAL_PWREx_EnableVddIO3();
  HAL_PWREx_ConfigVddIORange(PWR_VDDIO3, PWR_VDDIO_RANGE_1V8);

  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
  periph_clk_init.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
  periph_clk_init.ICSelection[RCC_IC3].ClockSelection = RCC_ICCLKSOURCE_PLL1;
  periph_clk_init.ICSelection[RCC_IC3].ClockDivider = 24;

  if (HAL_RCCEx_PeriphCLKConfig(&periph_clk_init) != HAL_OK)
  {
    return false;
  }

  __HAL_RCC_XSPIM_CLK_ENABLE();
  return true;
}

/*
 * The generated network exposes a weights pointer but does not guarantee that
 * the NOR memory-mapped view is already active. Probe and enable that mapping
 * once before the first inference when external weights are in use.
 */
static bool ensure_external_weight_mapping_ready(void)
{
  BSP_XSPI_NOR_Init_t xspi_init = {0};
  uintptr_t weights_addr = generated_weights_address();
  volatile const uint32_t *probe_word = (const uint32_t *)weights_addr;
  uint32_t first_word = 0U;
  int32_t status = BSP_ERROR_NONE;

  if (!weights_use_external_flash())
  {
    return true;
  }

  if (s_external_weight_mapping_initialized)
  {
    return true;
  }

  xspi_init.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
  xspi_init.TransferRate = BSP_XSPI_NOR_DTR_TRANSFER;

  if (!configure_external_weight_xspi2_clocking())
  {
    printf("STM32_XSPI_INIT=FAIL stage=clock_config status=%d addr=0x%08lx\r\n",
           HAL_ERROR,
           (unsigned long)weights_addr);
    flush_stdout();
    return false;
  }

  status = BSP_XSPI_NOR_Init(0U, &xspi_init);
  if (status != BSP_ERROR_NONE)
  {
    printf("STM32_XSPI_INIT=FAIL stage=init status=%ld addr=0x%08lx\r\n",
           (long)status,
           (unsigned long)weights_addr);
    flush_stdout();
    return false;
  }

  status = BSP_XSPI_NOR_EnableMemoryMappedMode(0U);
  if (status != BSP_ERROR_NONE)
  {
    printf("STM32_XSPI_INIT=FAIL stage=memory_mapped status=%ld addr=0x%08lx\r\n",
           (long)status,
           (unsigned long)weights_addr);
    flush_stdout();
    return false;
  }

  __DSB();
  __ISB();
  first_word = *probe_word;
  printf("STM32_XSPI_INIT=OK addr=0x%08lx word0=0x%08lx\r\n",
         (unsigned long)weights_addr,
         (unsigned long)first_word);
  flush_stdout();
  s_external_weight_mapping_initialized = true;
  return true;
}

static bool restore_external_weight_clocking_after_stop(void)
{
  if (!weights_use_external_flash())
  {
    return true;
  }

  return configure_external_weight_xspi2_clocking();
}

static void set_harness_idle(void)
{
  HAL_GPIO_WritePin(CREST_DUT_TRIGGER_GPIO_PORT, CREST_DUT_TRIGGER_GPIO_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(CREST_DUT_ARM_GPIO_PORT, CREST_DUT_ARM_GPIO_PIN, GPIO_PIN_SET);
}

static void arm_harness_window(void)
{
  HAL_GPIO_WritePin(CREST_DUT_ARM_GPIO_PORT, CREST_DUT_ARM_GPIO_PIN, GPIO_PIN_RESET);
  HAL_Delay(kArmHoldMs);
  HAL_GPIO_WritePin(CREST_DUT_TRIGGER_GPIO_PORT, CREST_DUT_TRIGGER_GPIO_PIN, GPIO_PIN_SET);
}

static void disarm_harness_window(void)
{
  HAL_GPIO_WritePin(CREST_DUT_TRIGGER_GPIO_PORT, CREST_DUT_TRIGGER_GPIO_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(CREST_DUT_ARM_GPIO_PORT, CREST_DUT_ARM_GPIO_PIN, GPIO_PIN_SET);
}

static uint32_t read_cpu_clock_hz(void)
{
  SystemCoreClockUpdate();
  if (SystemCoreClock > 0U)
  {
    return SystemCoreClock;
  }
  return HAL_RCC_GetCpuClockFreq();
}

static bool uart_read_line(char *buffer, size_t buffer_size, uint32_t total_timeout_ms)
{
  uint32_t start_tick = HAL_GetTick();
  size_t used = 0;

  if (buffer_size == 0U)
  {
    return false;
  }
  buffer[0] = '\0';

  while ((HAL_GetTick() - start_tick) < total_timeout_ms)
  {
    uint8_t ch = 0;
    if (HAL_UART_Receive(&hcom_uart[COM1], &ch, 1, kUartPollTimeoutMs) != HAL_OK)
    {
      continue;
    }

    if ((ch == '\r') || (ch == '\n'))
    {
      if (used == 0U)
      {
        continue;
      }
      buffer[used] = '\0';
      return true;
    }

    if (used + 1U < buffer_size)
    {
      buffer[used++] = (char)ch;
      buffer[used] = '\0';
    }
  }

  return false;
}

static bool wait_for_start_command(void)
{
  uint32_t wait_started = HAL_GetTick();
  uint32_t last_ready_emit = 0U;
  char line[32];

  while ((HAL_GetTick() - wait_started) < kStartCommandTimeoutMs)
  {
    uint32_t now = HAL_GetTick();
    if ((last_ready_emit == 0U) || ((now - last_ready_emit) >= kStartReadyHeartbeatMs))
    {
      emit_line("DUT READY");
      last_ready_emit = now;
    }

    if (!uart_read_line(line, sizeof(line), kStartReadyHeartbeatMs))
    {
      continue;
    }

    if (strcmp(line, "START") == 0)
    {
      return true;
    }
  }

  return false;
}

static void fill_input_buffer(ai_buffer *input)
{
  memset((void *)input[0].data, 0, AI_NETWORK_IN_1_SIZE_BYTES);
}

static bool rtc_try_init_with_source(uint32_t oscillator_type,
                                     uint32_t rtc_clock_selection,
                                     const char *label,
                                     uint32_t nominal_hz)
{
  RCC_OscInitTypeDef osc_init = {0};
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};
  RTC_TimeTypeDef rtc_time = {0};
  RTC_DateTypeDef rtc_date = {0};
  RTC_PrivilegeStateTypeDef privilege_state = {0};
  RTC_SecureStateTypeDef secure_state = {0};
  HAL_StatusTypeDef status = HAL_OK;

  __HAL_RCC_PWR_CLK_ENABLE();
  HAL_PWR_EnableBkUpAccess();
  printf("STM32_RTC_STAGE=backup_access source=%s\r\n", label);
  flush_stdout();

  osc_init.OscillatorType = oscillator_type;
  osc_init.PLL1.PLLState = RCC_PLL_NONE;
  osc_init.PLL2.PLLState = RCC_PLL_NONE;
  osc_init.PLL3.PLLState = RCC_PLL_NONE;
  osc_init.PLL4.PLLState = RCC_PLL_NONE;
  if (oscillator_type == RCC_OSCILLATORTYPE_LSE)
  {
    osc_init.LSEState = RCC_LSE_ON;
  }
  else
  {
    osc_init.LSIState = RCC_LSI_ON;
  }

  emit_line("STM32_RTC_STAGE=osc_config_start");
  status = HAL_RCC_OscConfig(&osc_init);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=osc_config_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=osc_config_ok");

  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_RTC;
  periph_clk_init.RTCClockSelection = rtc_clock_selection;
  emit_line("STM32_RTC_STAGE=periph_clk_start");
  status = HAL_RCCEx_PeriphCLKConfig(&periph_clk_init);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=periph_clk_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=periph_clk_ok");

  emit_line("STM32_RTC_STAGE=rtc_clock_enable_start");
  __HAL_RCC_RTC_ENABLE();
  emit_line("STM32_RTC_STAGE=rtc_clock_enable_ok");

  emit_line("STM32_RTC_STAGE=rtc_handle_setup_start");
  s_rtc_handle.Instance = RTC;
  s_rtc_handle.Init.HourFormat = RTC_HOURFORMAT_24;
  s_rtc_handle.Init.AsynchPrediv = 0U;
  /* Keep the RTC calendar and wake-up timer on the same 32.768 kHz basis so
   * subsecond timestamps and WUT sleep requests use one consistent timebase. */
  s_rtc_handle.Init.SynchPrediv = kRtcSyncPrediv;
  s_rtc_handle.Init.OutPut = RTC_OUTPUT_DISABLE;
  s_rtc_handle.Init.OutPutPolarity = RTC_OUTPUT_POLARITY_HIGH;
  s_rtc_handle.Init.OutPutType = RTC_OUTPUT_TYPE_OPENDRAIN;
  emit_line("STM32_RTC_STAGE=rtc_handle_setup_ok");

  emit_line("STM32_RTC_STAGE=rtc_init_start");
  status = HAL_RTC_Init(&s_rtc_handle);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=rtc_init_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=rtc_init_ok");

  privilege_state.rtcPrivilegeFull = RTC_PRIVILEGE_FULL_NO;
  privilege_state.backupRegisterPrivZone = RTC_PRIVILEGE_BKUP_ZONE_NONE;
  privilege_state.backupRegisterStartZone2 = RTC_BKP_DR0;
  privilege_state.backupRegisterStartZone3 = RTC_BKP_DR0;
  emit_line("STM32_RTC_STAGE=privilege_start");
  status = HAL_RTCEx_PrivilegeModeSet(&s_rtc_handle, &privilege_state);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=privilege_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=privilege_ok");

  secure_state.rtcSecureFull = RTC_SECURE_FULL_YES;
  secure_state.backupRegisterStartZone2 = RTC_BKP_DR0;
  secure_state.backupRegisterStartZone3 = RTC_BKP_DR0;
  emit_line("STM32_RTC_STAGE=secure_start");
  status = HAL_RTCEx_SecureModeSet(&s_rtc_handle, &secure_state);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=secure_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=secure_ok");

  rtc_time.Hours = 0U;
  rtc_time.Minutes = 0U;
  rtc_time.Seconds = 0U;
  rtc_time.DayLightSaving = RTC_DAYLIGHTSAVING_NONE;
  rtc_time.StoreOperation = RTC_STOREOPERATION_RESET;
  emit_line("STM32_RTC_STAGE=set_time_start");
  status = HAL_RTC_SetTime(&s_rtc_handle, &rtc_time, RTC_FORMAT_BIN);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=set_time_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=set_time_ok");

  rtc_date.WeekDay = RTC_WEEKDAY_MONDAY;
  rtc_date.Month = RTC_MONTH_JANUARY;
  rtc_date.Date = 1U;
  rtc_date.Year = 24U;
  emit_line("STM32_RTC_STAGE=set_date_start");
  status = HAL_RTC_SetDate(&s_rtc_handle, &rtc_date, RTC_FORMAT_BIN);
  if (status != HAL_OK)
  {
    printf("STM32_RTC_STAGE=set_date_fail status=%d source=%s\r\n", (int)status, label);
    flush_stdout();
    return false;
  }
  emit_line("STM32_RTC_STAGE=set_date_ok");

  s_rtc_clock_source = label;
  s_rtc_clock_hz_nominal = nominal_hz;
  s_cadence_timing_quality = (strcmp(label, "LSE") == 0) ? "crystal" : "rc_coarse";
  HAL_RCCEx_WakeUpStopCLKConfig(RCC_STOP_WAKEUPCLOCK_HSI);
  return true;
}

static bool rtc_init(void)
{
  /* Cadenced timing is validated against real wall-clock cadence, so use the
   * board's 32.768 kHz crystal-backed RTC source rather than the coarse LSI RC
   * oscillator. Fail the run if LSE bring-up does not succeed. */
  return rtc_try_init_with_source(RCC_OSCILLATORTYPE_LSE, RCC_RTCCLKSOURCE_LSE, "LSE", kRtcTimebaseHz);
}

static uint64_t rtc_time_to_us(const RTC_TimeTypeDef *time)
{
  uint32_t seconds_of_day = ((uint32_t)time->Hours * 3600U) +
                            ((uint32_t)time->Minutes * 60U) +
                            (uint32_t)time->Seconds;
  uint32_t fraction_ticks = 0U;
  uint64_t rtc_clock_hz = (uint64_t)s_rtc_clock_hz_nominal;

  if (time->SecondFraction >= time->SubSeconds)
  {
    fraction_ticks = time->SecondFraction - time->SubSeconds;
  }

  if (rtc_clock_hz == 0ULL)
  {
    rtc_clock_hz = (uint64_t)kRtcTimebaseHz;
  }

  return ((uint64_t)seconds_of_day * 1000000ULL) +
         (((uint64_t)fraction_ticks * 1000000ULL) / rtc_clock_hz);
}

static uint64_t rtc_datetime_to_us(const RTC_TimeTypeDef *time, const RTC_DateTypeDef *date)
{
  uint64_t day_offset_seconds = 0ULL;

  /* The cadence windows in this DUT are only a few seconds long, so using the
   * day-of-month as the coarse day offset is sufficient to keep timestamps
   * monotonic across second and midnight rollovers without implementing full
   * calendar conversion logic. */
  if (date->Date > 0U)
  {
    day_offset_seconds = ((uint64_t)date->Date - 1ULL) * 86400ULL;
  }

  return (day_offset_seconds * 1000000ULL) + rtc_time_to_us(time);
}

static uint64_t rtc_now_us(void)
{
  RTC_TimeTypeDef first_time = {0};
  RTC_TimeTypeDef second_time = {0};
  RTC_DateTypeDef first_date = {0};
  RTC_DateTypeDef second_date = {0};
  uint64_t first_us = 0ULL;
  uint64_t second_us = 0ULL;
  int attempt = 0;

  /* Read RTC time/date twice and keep the second sample only when it is
   * monotonic relative to the first. This avoids mixing a pre-rollover time
   * sample with a post-rollover date sample when the cadence window crosses a
   * second boundary. */
  for (attempt = 0; attempt < 4; ++attempt)
  {
    HAL_RTC_GetTime(&s_rtc_handle, &first_time, RTC_FORMAT_BIN);
    HAL_RTC_GetDate(&s_rtc_handle, &first_date, RTC_FORMAT_BIN);
    HAL_RTC_GetTime(&s_rtc_handle, &second_time, RTC_FORMAT_BIN);
    HAL_RTC_GetDate(&s_rtc_handle, &second_date, RTC_FORMAT_BIN);

    first_us = rtc_datetime_to_us(&first_time, &first_date);
    second_us = rtc_datetime_to_us(&second_time, &second_date);
    if (second_us >= first_us)
    {
      return second_us;
    }
  }

  return second_us;
}

static bool restore_runtime_clocks(void)
{
  RCC_PeriphCLKInitTypeDef periph_clk_init = {0};

  SystemClock_Config();

  periph_clk_init.PeriphClockSelection = RCC_PERIPHCLK_USART1;
  periph_clk_init.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  if (HAL_RCCEx_PeriphCLKConfig(&periph_clk_init) != HAL_OK)
  {
    return false;
  }

  if (weights_use_external_flash())
  {
    if (!restore_external_weight_clocking_after_stop())
    {
      return false;
    }
  }

  SystemCoreClockUpdate();
  return true;
}

static bool stop_sleep_for_us(uint32_t sleep_us, int64_t *wake_recovery_us,
                              CrestDutMeasurement *measurement)
{
  uint32_t wakeup_counter = 0U;
  uint64_t after_wake_us = 0ULL;
  uint64_t after_restore_us = 0ULL;

  if (sleep_us < CREST_DUT_MIN_SLEEP_US)
  {
    return false;
  }

  wakeup_counter = wakeup_counter_for_sleep_us(sleep_us);
  if (wakeup_counter < 2U)
  {
    return false;
  }
  wakeup_counter -= 1U;

  s_rtc_wakeup_fired = false;
  HAL_NVIC_DisableIRQ(RTC_S_IRQn);
  NVIC_ClearPendingIRQ(RTC_S_IRQn);
  HAL_NVIC_SetPriority(RTC_S_IRQn, 0U, 0U);
  HAL_NVIC_EnableIRQ(RTC_S_IRQn);
  HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
  if (HAL_RTCEx_SetWakeUpTimer_IT(&s_rtc_handle,
                                  wakeup_counter,
                                  RTC_WAKEUPCLOCK_RTCCLK_DIV16,
                                  0U) != HAL_OK)
  {
    return false;
  }

  __HAL_PWR_CLEAR_FLAG(PWR_FLAG_STOPF);
  measurement->slept_slots++;
  HAL_SuspendTick();
  HAL_PWR_EnterSTOPMode(PWR_MAINREGULATOR_ON, PWR_STOPENTRY_WFI);
  if (__HAL_PWR_GET_FLAG(PWR_FLAG_STOPF) != RESET)
  {
    measurement->stop_entries++;
    __HAL_PWR_CLEAR_FLAG(PWR_FLAG_STOPF);
  }

  /* HAL synchronization has a tick-based timeout: resume SysTick first.
   * STOPF is entry evidence only; it does not measure Stop residency. */
  HAL_ResumeTick();
  __HAL_RTC_WRITEPROTECTION_DISABLE(&s_rtc_handle);
  HAL_StatusTypeDef sync_status = HAL_RTC_WaitForSynchro(&s_rtc_handle);
  __HAL_RTC_WRITEPROTECTION_ENABLE(&s_rtc_handle);
  if (sync_status != HAL_OK)
  {
    HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
    HAL_NVIC_DisableIRQ(RTC_S_IRQn);
    /* The caller emits a UART integrity failure. Restore the runtime clock
     * even when the RTC cannot synchronize, so that report has the right baud.
     * A clock-restore failure itself remains a best-effort reporting case. */
    (void)restore_runtime_clocks();
    return false;
  }
  after_wake_us = rtc_now_us();
  if (!restore_runtime_clocks())
  {
    HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
    HAL_NVIC_DisableIRQ(RTC_S_IRQn);
    return false;
  }

  after_restore_us = rtc_now_us();
  HAL_ResumeTick();
  HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
  HAL_NVIC_DisableIRQ(RTC_S_IRQn);

  if (wake_recovery_us != NULL)
  {
    *wake_recovery_us = (int64_t)(after_restore_us - after_wake_us);
  }
  /* STOPF proves entry only. An unrelated interrupt cannot justify counting
   * the requested sleep interval as completed; invalidate this measurement. */
  return s_rtc_wakeup_fired;
}

static bool rtc_wakeup_irq_self_test(void)
{
  const uint32_t self_test_counter = 100U;
  const uint64_t timeout_us = 250000ULL;
  uint64_t deadline_us = rtc_now_us() + timeout_us;

  s_rtc_wakeup_fired = false;
  HAL_NVIC_DisableIRQ(RTC_S_IRQn);
  NVIC_ClearPendingIRQ(RTC_S_IRQn);
  HAL_NVIC_SetPriority(RTC_S_IRQn, 0U, 0U);
  HAL_NVIC_EnableIRQ(RTC_S_IRQn);
  HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
  if (HAL_RTCEx_SetWakeUpTimer_IT(&s_rtc_handle,
                                  self_test_counter,
                                  RTC_WAKEUPCLOCK_RTCCLK_DIV16,
                                  0U) != HAL_OK)
  {
    emit_line("STM32_RTC_WUT_SELFTEST=ARM_FAIL");
    return false;
  }

  while ((!s_rtc_wakeup_fired) && (rtc_now_us() < deadline_us))
  {
  }

  HAL_RTCEx_DeactivateWakeUpTimer(&s_rtc_handle);
  HAL_NVIC_DisableIRQ(RTC_S_IRQn);

  printf("STM32_RTC_WUT_SELFTEST=%s fired=%d smISR=0x%08lx misr=0x%08lx sr=0x%08lx seccfgr=0x%08lx privcfgr=0x%08lx\r\n",
         s_rtc_wakeup_fired ? "OK" : "FAIL",
         s_rtc_wakeup_fired ? 1 : 0,
         (unsigned long)RTC->SMISR,
         (unsigned long)RTC->MISR,
         (unsigned long)RTC->SR,
         (unsigned long)RTC->SECCFGR,
         (unsigned long)RTC->PRIVCFGR);
  flush_stdout();
  return s_rtc_wakeup_fired;
}

static void emit_phase_header(void)
{
  const char *phase_label =
      (CREST_DUT_SELECTED_PHASE == CREST_DUT_PHASE_CADENCED) ? "cadenced" : "back_to_back";

  /* Snapshot after the loader has disconnected, before any measured window.
   * This describes this boot, not the historical experiment or an unattached boot. */
  printf("STM32_POWER_REGISTERS cpucr=0x%08lx dbgcr=0x%08lx stopcr=0x%08lx\r\n",
         (unsigned long)PWR->CPUCR, (unsigned long)DBGMCU->CR,
         (unsigned long)RCC->STOPCR);
  printf("measurement protocol output: %s\r\n", CREST_DUT_MEASUREMENT_PROTOCOL);
  printf("phase output: %s\r\n", phase_label);
  printf("rtc clock source output: %s\r\n", s_rtc_clock_source);
  printf("rtc clock hz nominal output: %lu\r\n", (unsigned long)s_rtc_clock_hz_nominal);
  printf("cadence timing quality output: %s\r\n", s_cadence_timing_quality);
  printf("stop mode variant output: %s\r\n", CREST_DUT_STOP_MODE_VARIANT);
  printf("cadence budget (ms): %d\r\n", CREST_DUT_LATENCY_BUDGET_MS);
  flush_stdout();
}

static void emit_measurement_telemetry(const CrestDutMeasurement *measurement)
{
  bool have_means = (measurement->runs_completed > 0U) && measurement->have_timing;
  bool cadenced_phase = (CREST_DUT_SELECTED_PHASE == CREST_DUT_PHASE_CADENCED);
  int64_t wake_recovery_mean_us = -1;
  int64_t wake_overshoot_mean_us = -1;

  if (measurement->runs_completed > 0U)
  {
    printf("runs: %lu\r\n", (unsigned long)measurement->runs_completed);
  }
  else
  {
    emit_line("runs: 0");
  }

  emit_cpu_clock_telemetry(measurement->clock_hz);
  emit_cycle_telemetry(measurement->dwt_cycles_per_inference_milli, measurement->have_cycles);
  emit_float_scaled_line("timer per inference output",
                         measurement->timer_per_inference_scaled_1e6,
                         have_means);
  emit_float_scaled_line("timer per window output",
                         measurement->timer_per_window_scaled_1e6,
                         measurement->have_timing);
  if (CREST_DUT_SELECTED_PHASE == CREST_DUT_PHASE_BACK_TO_BACK)
  {
    emit_float_scaled_line("timer output",
                           measurement->timer_per_inference_scaled_1e6,
                           have_means);
  }

  if (measurement->runs_completed > 0U)
  {
    wake_overshoot_mean_us = measurement->wake_overshoot_total_us / (int64_t)measurement->runs_completed;
  }
  if (cadenced_phase && (measurement->runs_completed > 0U))
  {
    wake_recovery_mean_us = measurement->wake_recovery_total_us / (int64_t)measurement->runs_completed;
  }

  if (cadenced_phase)
  {
    emit_signed_us_line("wake recovery us",
                        wake_recovery_mean_us,
                        measurement->runs_completed > 0U);
    emit_signed_us_line("wake overshoot us",
                        wake_overshoot_mean_us,
                        measurement->runs_completed > 0U);
    printf("rtc sleep total ms: %lu.%03lu\r\n",
           (unsigned long)(measurement->rtc_sleep_total_us / 1000ULL),
           (unsigned long)(measurement->rtc_sleep_total_us % 1000ULL));
    printf("slept slots: %lu\r\n", (unsigned long)measurement->slept_slots);
    printf("stop entries: %lu\r\n", (unsigned long)measurement->stop_entries);
    printf("cadence deadline misses: %lu\r\n", (unsigned long)measurement->deadline_miss_count);
  }
  emit_line("inference seq output: 1");

  printf("STM32_AI_RUN=OK\r\n");
  if (measurement->have_cycles)
  {
    printf("STM32_AI_LATENCY_CYCLES=%lu\r\n",
           (unsigned long)(measurement->dwt_cycles_per_inference_milli / 1000ULL));
  }
  else
  {
    emit_line("STM32_AI_LATENCY_CYCLES=0");
  }
  flush_stdout();
}

static bool run_back_to_back_window(ai_handle network,
                                    ai_buffer *input,
                                    ai_buffer *output,
                                    CrestDutMeasurement *measurement)
{
  uint64_t active_cycles_total = 0ULL;
  uint64_t phase_start_us = 0ULL;
  uint64_t phase_end_us = 0ULL;
  bool cycles_valid = true;
  ai_i32 batch = 0;

  fill_input_buffer(input);
  measurement->clock_hz = read_cpu_clock_hz();
  arm_harness_window();
  phase_start_us = rtc_now_us();

  for (int run_idx = 0; run_idx < CREST_DUT_MEASURED_RUNS; ++run_idx)
  {
    uint32_t invoke_start_cycles = DWT->CYCCNT;
    batch = ai_network_run(network, input, output);
    uint32_t invoke_end_cycles = DWT->CYCCNT;
    if (batch != 1)
    {
      disarm_harness_window();
      printf("STM32_AI_RUN=FAIL batch=%ld run=%d\r\n", (long)batch, run_idx + 1);
      flush_stdout();
      return false;
    }
    active_cycles_total += (uint32_t)(invoke_end_cycles - invoke_start_cycles);
    measurement->runs_completed++;
  }

  phase_end_us = rtc_now_us();
  disarm_harness_window();

  /* Keep RTC reads outside this energy window's invoke loop. A short whole
   * window proves every invoke wrap-safe; its average cannot prove that for
   * a long window, so conservatively discard optional DWT telemetry then. */
  cycles_valid = ((phase_end_us - phase_start_us + 62ULL) * measurement->clock_hz) <
                 (0x100000000ULL * 1000000ULL);
  measurement->have_cycles = cycles_valid && (measurement->runs_completed > 0U);
  measurement->have_timing = measurement->runs_completed > 0U;
  if (measurement->runs_completed > 0U)
  {
    measurement->dwt_cycles_per_inference_milli =
        (active_cycles_total * 1000ULL) / measurement->runs_completed;
    measurement->timer_per_inference_scaled_1e6 =
        (phase_end_us - phase_start_us) / measurement->runs_completed;
    measurement->timer_per_window_scaled_1e6 = phase_end_us - phase_start_us;
  }

  return true;
}

static bool run_cadenced_window(ai_handle network,
                                ai_buffer *input,
                                ai_buffer *output,
                                CrestDutMeasurement *measurement)
{
  uint64_t phase_start_us = 0ULL;
  uint64_t next_release_us = 0ULL;
  uint64_t phase_end_us = 0ULL;
  uint64_t active_cycles_total = 0ULL;
  uint64_t active_us_total = 0ULL;
  bool cycles_valid = true;
  const uint64_t cadence_us = (uint64_t)CREST_DUT_LATENCY_BUDGET_MS * 1000ULL;

  fill_input_buffer(input);
  arm_harness_window();
  phase_start_us = rtc_now_us();
  next_release_us = phase_start_us;
  measurement->clock_hz = read_cpu_clock_hz();

  for (int run_idx = 0; run_idx < CREST_DUT_MEASURED_RUNS; ++run_idx)
  {
    uint64_t now_us = rtc_now_us();
    int64_t wake_recovery_us = 0;

    if (now_us < next_release_us)
    {
      uint64_t slack_us = next_release_us - now_us;
      if (slack_us > (uint64_t)(CREST_DUT_WAKE_MARGIN_US + CREST_DUT_MIN_SLEEP_US))
      {
        uint32_t requested_sleep_us = (uint32_t)(slack_us - (uint64_t)CREST_DUT_WAKE_MARGIN_US);
        if (stop_sleep_for_us(requested_sleep_us, &wake_recovery_us, measurement))
        {
          measurement->rtc_sleep_total_us += (uint64_t)requested_sleep_us;
          measurement->wake_recovery_total_us += wake_recovery_us;
        }
        else
        {
          disarm_harness_window();
          emit_line("STM32_AI_RUN=FAIL measurement_integrity:stop_sleep");
          flush_stdout();
          return false;
        }
      }
      while ((now_us = rtc_now_us()) < next_release_us)
      {
      }
    }

    uint64_t inference_start_us = rtc_now_us();
    uint32_t invoke_start_cycles = DWT->CYCCNT;
    measurement->wake_overshoot_total_us += (int64_t)(inference_start_us - next_release_us);
    ai_i32 batch = ai_network_run(network, input, output);
    uint32_t invoke_end_cycles = DWT->CYCCNT;
    uint64_t inference_end_us = rtc_now_us();
    if (batch != 1)
    {
      disarm_harness_window();
      printf("STM32_AI_RUN=FAIL batch=%ld run=%d\r\n", (long)batch, run_idx + 1);
      flush_stdout();
      return false;
    }

    uint64_t active_us = inference_end_us - inference_start_us;
    active_us_total += active_us;
    if (((active_us + 62ULL) * measurement->clock_hz) >=
        (0x100000000ULL * 1000000ULL))
    {
      cycles_valid = false;
    }
    active_cycles_total += (uint32_t)(invoke_end_cycles - invoke_start_cycles);
    measurement->runs_completed++;
    next_release_us += cadence_us;
    if (rtc_now_us() > next_release_us)
    {
      measurement->deadline_miss_count++;
    }
  }

  /* Protocol v2 deliberately retains the historical awake final-slot tail. */
  while ((phase_end_us = rtc_now_us()) < next_release_us)
  {
  }
  disarm_harness_window();

  measurement->clock_hz = read_cpu_clock_hz();
  measurement->have_cycles = cycles_valid && (measurement->clock_hz > 0U) &&
                             (measurement->runs_completed > 0U);
  measurement->have_timing = measurement->runs_completed > 0U;
  if (measurement->runs_completed > 0U)
  {
    measurement->dwt_cycles_per_inference_milli =
        (active_cycles_total * 1000ULL) / (uint64_t)measurement->runs_completed;
    /* RTC tick quantization is phase-locked to each cadence release. Prefer
     * DWT for active latency when every invoke is provably below one wrap;
     * RTC still measures the wall window and is the long-invoke fallback. */
    if (measurement->have_cycles)
    {
      measurement->timer_per_inference_scaled_1e6 =
          (active_cycles_total * 1000000ULL) /
          ((uint64_t)measurement->clock_hz * measurement->runs_completed);
    }
    else
    {
      measurement->timer_per_inference_scaled_1e6 =
          active_us_total / (uint64_t)measurement->runs_completed;
    }
    measurement->timer_per_window_scaled_1e6 = (phase_end_us - phase_start_us);
  }

  return true;
}

void crest_dut_harness_gpio_init(void)
{
  GPIO_InitTypeDef gpio_init = {0};

  HAL_PWREx_EnableVddIO5();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
  gpio_init.Pull = GPIO_NOPULL;
  gpio_init.Speed = GPIO_SPEED_FREQ_LOW;

  gpio_init.Pin = CREST_DUT_ARM_GPIO_PIN;
  HAL_GPIO_Init(CREST_DUT_ARM_GPIO_PORT, &gpio_init);

  gpio_init.Pin = CREST_DUT_TRIGGER_GPIO_PIN;
  HAL_GPIO_Init(CREST_DUT_TRIGGER_GPIO_PORT, &gpio_init);

  set_harness_idle();
}

void crest_dut_rtc_irq_handler(void)
{
  HAL_RTCEx_WakeUpTimerIRQHandler(&s_rtc_handle);
}

void HAL_RTCEx_WakeUpTimerEventCallback(RTC_HandleTypeDef *hrtc)
{
  if (hrtc->Instance == RTC)
  {
    s_rtc_wakeup_fired = true;
  }
}

int crest_dut_run_once(void)
{
  ai_handle network = AI_HANDLE_NULL;
  CrestDutMeasurement measurement = {0};

  emit_line("STM32_BOOT=START");
  emit_line("STM32_RTC_INIT=START");
  if (!rtc_init())
  {
    emit_line("STM32_RTC_INIT=FAIL");
    emit_line("STM32_AI_INIT=FAIL reason=rtc_init");
    return -6;
  }
  emit_line("STM32_RTC_INIT=OK");
  if (!rtc_wakeup_irq_self_test())
  {
    if (CREST_DUT_SELECTED_PHASE == CREST_DUT_PHASE_CADENCED)
    {
      emit_line("STM32_AI_INIT=FAIL reason=rtc_wakeup_selftest");
      return -7;
    }
  }

  if (!ensure_external_weight_mapping_ready())
  {
    emit_line("STM32_AI_INIT=FAIL reason=external_weight_mapping");
    return -5;
  }

  const ai_handle activations[] = {s_network_activations};
  ai_error err = ai_network_create_and_init(&network, activations, NULL);
  if (err.type != AI_ERROR_NONE)
  {
    printf("STM32_AI_INIT=FAIL type=%d code=%d\r\n", err.type, err.code);
    flush_stdout();
    return -1;
  }

  printf("STM32_AI_INIT=OK\r\n");
  flush_stdout();
  emit_phase_header();

  ai_buffer *input = ai_network_inputs_get(network, NULL);
  ai_buffer *output = ai_network_outputs_get(network, NULL);

  fill_input_buffer(input);
  ai_i32 batch = ai_network_run(network, input, output);
  if (batch != 1)
  {
    printf("STM32_AI_RUN=FAIL batch=%ld\r\n", (long)batch);
    flush_stdout();
    ai_network_destroy(network);
    return -2;
  }

  if (!wait_for_start_command())
  {
    emit_line("STM32_AI_START=TIMEOUT");
    ai_network_destroy(network);
    return -3;
  }

  if (CREST_DUT_SELECTED_PHASE == CREST_DUT_PHASE_CADENCED)
  {
    if (!run_cadenced_window(network, input, output, &measurement))
    {
      ai_network_destroy(network);
      return -4;
    }
  }
  else
  {
    if (!run_back_to_back_window(network, input, output, &measurement))
    {
      ai_network_destroy(network);
      return -4;
    }
  }

  emit_measurement_telemetry(&measurement);
  ai_network_destroy(network);
  return 0;
}
