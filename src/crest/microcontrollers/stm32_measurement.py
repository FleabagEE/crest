"""Pure acceptance rules for the repaired STM32 measurement protocol."""
import math

PROTOCOL = 'rtc_wall_v4_dwt_active_stop_evidence_awake_tail'


def firmware_period_ms(period_ms):
    """Resolve the integer cadence emitted in the generated firmware header.

    Parameters
    ----------
    period_ms : float
        Requested positive cadence in milliseconds.

    Returns
    -------
    int
        Rounded cadence, clamped to the firmware minimum of one millisecond.
    """
    return max(1, int(round(period_ms)))


def positive(value):
    """Check whether a telemetry value is finite and strictly positive.

    Parameters
    ----------
    value : object
        Candidate scalar; booleans and nonnumeric values are rejected.

    Returns
    -------
    bool
        Whether the value is a finite positive integer or float.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_measurement(metrics, *, latency_s, phase, clock_mhz, runs, period_ms,
                         dut_log, harness_log, require_harness):
    """Separate instrumentation failures from deployment infeasibility.

    Parameters
    ----------
    metrics : dict
        Parsed timing, clock, energy, and Stop-entry telemetry.
    latency_s : float
        Active invocation latency in seconds.
    phase : str
        Requested back-to-back or cadenced measurement phase.
    clock_mhz : int
        Requested CPU clock in MHz.
    runs : int
        Expected number of measured invocations.
    period_ms : float
        Requested cadence and active-latency budget in milliseconds.
    dut_log : str or sequence of str
        DUT output containing protocol identity and cadence budget.
    harness_log : str or sequence of str
        Raw harness output used to establish presence of external evidence.
    require_harness : bool
        Whether harness timing and energy must also pass acceptance.

    Returns
    -------
    dict
        Acceptance flags, individual checks, failure reasons, and per-release
        and aggregate energy in millijoules.

    Notes
    -----
    The 2% timing agreement band detects gross mismatch, not meter accuracy.
    Two RTC ticks (62 us) bound cadence-window endpoint quantization.
    Window duration follows the rounded firmware cadence; the active-latency
    constraint retains the requested budget. Clock telemetry reports the integer
    configured frequency, not a physical oscillator measurement.
    Requested sleep is never interpreted as measured Stop residency.
    """
    effective_period_ms = firmware_period_ms(period_ms)
    reasons = []
    checks = {
        'phase_matches': metrics.get('phase') == phase,
        'clock_matches': metrics.get('clock_hz') == clock_mhz * 1_000_000,
        'runs_match': metrics.get('runs') == runs,
        'latency_positive_finite': positive(latency_s),
        'protocol_matches': f'measurement protocol output: {PROTOCOL}' in dut_log,
        'period_matches': f'cadence budget (ms): {effective_period_ms}' in dut_log,
        'rtc_window_positive_finite': positive(metrics.get('timer_per_window_s')),
        'rtc_lse': metrics.get('rtc_clock_source') == 'LSE' and metrics.get('rtc_clock_hz_nominal') == 32768,
    }
    rtc = metrics.get('timer_per_window_s')
    if require_harness:
        harness = metrics.get('harness_latency_s')
        checks.update(raw_harness_present=bool(harness_log),
                      energy_positive_finite=positive(metrics.get('energy_mj_per_inference')),
                      harness_window_positive_finite=positive(harness))
        checks['window_sources_agree'] = (positive(rtc) and positive(harness)
                                          and abs(harness * runs - rtc) / rtc <= .02)
    if phase == 'cadenced':
        slots, entries = metrics.get('slept_slots'), metrics.get('stop_entries')
        def valid_count(value):
            """Check a Stop count against the available inter-release sleep slots.

            Parameters
            ----------
            value : object
                Reported count, required to be an integer between zero and runs minus one.

            Returns
            -------
            bool
                Whether the count is finite, integral, and within the slot limit.
            """
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 and int(value) == value and value <= max(0, runs - 1)
        checks['stop_counts_valid'] = valid_count(slots) and valid_count(entries)
        checks['stop_entries_match_slots'] = checks['stop_counts_valid'] and entries == slots
        misses = metrics.get('deadline_miss_count')
        checks['deadline_count_valid'] = isinstance(misses, (int, float)) and not isinstance(misses, bool) and math.isfinite(misses) and misses >= 0 and int(misses) == misses
    reasons = [name for name, passed in checks.items() if not passed]
    eligibility = []
    if positive(latency_s) and latency_s * 1000 > period_ms:
        eligibility.append('active_latency_exceeds_budget')
    if phase == 'cadenced':
        if checks['deadline_count_valid'] and metrics['deadline_miss_count'] > 0:
            eligibility.append('deadline_miss')
        if positive(rtc) and abs(rtc - runs * effective_period_ms / 1000) > .000062:
            eligibility.append('cadence_window_outside_budget')
    return {'instrumentation_valid': not reasons, 'checks': checks,
            'instrumentation_reasons': reasons, 'deployment_eligible': not reasons and not eligibility,
            'infeasibility_reasons': eligibility, 'energy_unit': 'mJ_per_release',
            'energy_mj_per_release': metrics.get('energy_mj_per_inference'),
            'energy_mj_aggregate': metrics.get('energy_mj_per_inference', 0) * runs if positive(metrics.get('energy_mj_per_inference')) else None}
