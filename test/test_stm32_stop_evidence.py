"""Offline tests for Stop-entry evidence and wake synchronization."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from crest.microcontrollers.stm32_measurement import validate_measurement
from crest.microcontrollers import stm32_runtime


def test_stop_count_acceptance():
    """Missing, fractional or mismatched entry counts invalidate instrumentation."""
    from test.test_stm32_measurement_acceptance import sample
    args = sample('cadenced')
    for slots, entries, valid in [(9, 9, True), (9, 8, False), (None, None, False),
                                   (0, 0, True), (9.5, 9.5, False), (10, 10, False)]:
        args['metrics'].update(slept_slots=slots, stop_entries=entries)
        result = validate_measurement(**args)
        assert result['instrumentation_valid'] is valid


def test_real_stop_function_resynchronizes_before_reading():
    """Compile the actual stop helper against HAL fakes, including sync failure."""
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('C compiler unavailable')
    import tempfile
    source = (Path(__file__).resolve().parents[1] / 'sketches/stm32/crest_stm32_lrun/Appli/Src/crest_dut_runner.c').read_text()
    measurement = re.search(r'typedef struct\n\{.*?\} CrestDutMeasurement;', source, re.S).group()
    function = source[source.index('static bool stop_sleep_for_us'):source.index('static bool rtc_wakeup_irq_self_test')]
    fake = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#define CREST_DUT_MIN_SLEEP_US 5000
#define RTC_S_IRQn 1
#define RTC_WAKEUPCLOCK_RTCCLK_DIV16 1
#define PWR_MAINREGULATOR_ON 1
#define PWR_STOPENTRY_WFI 1
#define PWR_FLAG_STOPF 1
#define RESET 0
#define HAL_OK 0
typedef int HAL_StatusTypeDef;
static int write_protected=1;
#define __HAL_RTC_WRITEPROTECTION_DISABLE(h) (write_protected=0)
#define __HAL_RTC_WRITEPROTECTION_ENABLE(h) (write_protected=1)
static int s_rtc_handle, flag, synced, tick, sync_fail, simulate_stop=1, reads, disabled;
static bool s_rtc_wakeup_fired;
static int restore_calls, restore_fail, runtime_clock=1, simulate_rtc_wake=1;
static uint32_t wakeup_counter_for_sleep_us(uint32_t us) { return us/500; }
static void HAL_NVIC_DisableIRQ(int n) { ++disabled; }
static void NVIC_ClearPendingIRQ(int n) {}
static void HAL_NVIC_SetPriority(int n,int a,int b) {}
static void HAL_NVIC_EnableIRQ(int n) {}
static void HAL_RTCEx_DeactivateWakeUpTimer(int *h) {}
static int HAL_RTCEx_SetWakeUpTimer_IT(int *h,int c,int d,int e) { return HAL_OK; }
static void HAL_SuspendTick(void) { tick=0; }
static void HAL_ResumeTick(void) { tick=1; }
static void HAL_PWR_EnterSTOPMode(int a,int b) {
 assert(flag==0); flag=simulate_stop; synced=0; runtime_clock=0;
 s_rtc_wakeup_fired=simulate_rtc_wake;
}
#define __HAL_PWR_CLEAR_FLAG(f) (flag=0)
#define __HAL_PWR_GET_FLAG(f) (flag)
static int HAL_RTC_WaitForSynchro(int *h) { assert(tick && !write_protected); synced=!sync_fail; return sync_fail; }
static uint64_t rtc_now_us(void) { assert(synced && write_protected); return ++reads; }
static bool restore_runtime_clocks(void) {
 ++restore_calls; runtime_clock=!restore_fail; return !restore_fail;
}
'''
    checks = r'''
int main(void) {
 CrestDutMeasurement m={0}; int64_t recovery=0;
 flag=1; assert(stop_sleep_for_us(10000,&recovery,&m));
 assert(m.slept_slots==1 && m.stop_entries==1 && reads==2 && recovery==1);
 assert(runtime_clock && restore_calls==1);
 simulate_stop=0; assert(stop_sleep_for_us(10000,&recovery,&m));
 assert(m.slept_slots==2 && m.stop_entries==1);
 sync_fail=1; int before=reads; int previous_disabled=disabled;
 assert(!stop_sleep_for_us(10000,&recovery,&m));
 assert(reads==before && tick && write_protected && disabled>=previous_disabled+2);
 assert(runtime_clock && restore_calls==3); /* sync failure still restores UART clock */
 sync_fail=0; restore_fail=1; before=reads;
 assert(!stop_sleep_for_us(10000,&recovery,&m));
 assert(reads==before+1 && restore_calls==4 && tick && write_protected);
 /* Entry alone cannot prove the requested sleep completed after an early wake. */
 restore_fail=0; simulate_stop=1; simulate_rtc_wake=0;
 assert(!stop_sleep_for_us(10000,&recovery,&m));
 assert(runtime_clock && restore_calls==5 && m.slept_slots==5 && m.stop_entries==2);
 return 0;
}
'''
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path/'test.c').write_text(fake+measurement+function+checks)
        subprocess.run([cc, '-std=c11', '-Werror=implicit-function-declaration', str(path/'test.c'), '-o', str(path/'test')], check=True, capture_output=True)
        subprocess.run([str(path/'test')], check=True, capture_output=True)
