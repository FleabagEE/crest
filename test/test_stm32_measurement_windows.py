"""Execute the actual STM32 window functions with deterministic clock/HIL fakes."""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'sketches/stm32/crest_stm32_lrun/Appli/Src/crest_dut_runner.c'


class MeasurementWindowTests(unittest.TestCase):
    def test_wall_timing_and_cycle_wraps(self):
        """Long batches and single invokes retain valid wall timing, without UART."""
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('C compiler unavailable')
        source = RUNNER.read_text()
        measurement = re.search(r'typedef struct\n\{.*?\} CrestDutMeasurement;', source, re.S).group()
        windows = source[source.index('static bool run_back_to_back_window'):source.index('void crest_dut_harness_gpio_init')]
        fake = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
typedef void *ai_handle;
typedef int ai_buffer;
typedef int ai_i32;
#define CREST_DUT_MEASURED_RUNS 10
#define CREST_DUT_LATENCY_BUDGET_MS 200
#define CREST_DUT_WAKE_MARGIN_US 5000
#define CREST_DUT_MIN_SLEEP_US 5000
static struct { uint32_t CYCCNT; } dwt;
#define DWT (&dwt)
static uint64_t now, duration;
static bool armed;
static int sleeps, rtc_reads;
static uint32_t read_cpu_clock_hz(void) { return 800000000; }
static uint64_t rtc_now_us(void) { ++rtc_reads; now += 31; return now; }
static void fill_input_buffer(ai_buffer *p) { (void)p; }
static void arm_harness_window(void) { armed = true; }
static void disarm_harness_window(void) { armed = false; }
static void flush_stdout(void) { assert(!armed); }
static int measured_printf(const char *fmt, ...) { assert(!armed); return 0; }
#define printf measured_printf
static int ai_network_run(ai_handle n, ai_buffer *i, ai_buffer *o) {
  now += duration; dwt.CYCCNT += (uint32_t)(duration * 800); return 1;
}
'''
        stop_fake = r'''
static bool fail_stop;
static const char *last_line;
static void emit_line(const char *s) { assert(!armed); last_line=s; }
static bool stop_sleep_for_us(uint32_t us, int64_t *recovery, CrestDutMeasurement *m) {
 if (fail_stop) return false;
 ++sleeps; ++m->slept_slots; ++m->stop_entries; now += us; *recovery = 0; return true;
}
'''
        checks = r'''
int main(void) {
 CrestDutMeasurement m = {0};
 duration = 1000000; /* batch wraps repeatedly, individual invokes do not */
 assert(run_back_to_back_window(0,0,0,&m));
 assert(m.have_timing && !m.have_cycles);
 assert(rtc_reads==2);
 assert(m.timer_per_inference_scaled_1e6 >= duration);
 assert(m.timer_per_inference_scaled_1e6 < duration + 100);
 assert(m.dwt_cycles_per_inference_milli == 800000000000ULL);
 memset(&m,0,sizeof m); duration=20000; rtc_reads=0;
 assert(run_back_to_back_window(0,0,0,&m));
 assert(m.have_timing && m.have_cycles && rtc_reads==2);
 memset(&m,0,sizeof m); duration = 6000000; /* single invoke wraps at 800 MHz */
 assert(run_back_to_back_window(0,0,0,&m));
 assert(m.have_timing && !m.have_cycles);
 assert(m.timer_per_inference_scaled_1e6 >= duration);
 memset(&m,0,sizeof m); now=0; duration=20000; sleeps=0;
 assert(run_cadenced_window(0,0,0,&m));
 assert(m.have_cycles && m.runs_completed==10 && m.deadline_miss_count==0);
 assert(m.timer_per_window_scaled_1e6 >= 2000000);
 assert(m.timer_per_window_scaled_1e6 < 2000200);
 assert(sleeps==9 && m.slept_slots==9 && m.stop_entries==9); /* historical final slot remains awake */
 assert(m.timer_per_inference_scaled_1e6 == duration);
 /* Each fake RTC read advances by a tick: DWT must avoid this fixed bias. */
 memset(&m,0,sizeof m); now=0; duration=100;
 assert(run_cadenced_window(0,0,0,&m));
 assert(m.have_cycles && m.timer_per_inference_scaled_1e6 == 100);
 assert(m.timer_per_window_scaled_1e6 >= 2000000);
 memset(&m,0,sizeof m); duration=6000000;
 assert(run_cadenced_window(0,0,0,&m));
 assert(!m.have_cycles && m.have_timing && m.deadline_miss_count==10);
 assert(m.timer_per_inference_scaled_1e6 >= duration);
 memset(&m,0,sizeof m); now=0; duration=20000; fail_stop=true;
 assert(!run_cadenced_window(0,0,0,&m));
 assert(!armed && strcmp(last_line,"STM32_AI_RUN=FAIL measurement_integrity:stop_sleep")==0);
 return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'test.c').write_text(fake + measurement + stop_fake + windows + checks)
            subprocess.run([compiler, '-std=c11', '-Werror=implicit-function-declaration', str(path/'test.c'), '-o', str(path/'test')], check=True, capture_output=True)
            subprocess.run([str(path/'test')], check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
