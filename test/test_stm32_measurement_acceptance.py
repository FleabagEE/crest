"""No hardware: exercise production acceptance and durable failure records."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from crest.microcontrollers.stm32_measurement import validate_measurement, PROTOCOL
from crest.microcontrollers.stm32_nucleo_n657x0 import STM32NucleoN657X0QDevice
from crest.devices import DeviceMetrics
from crest.errors import HIL_ERROR_OK, HIL_ERROR_LATENCY, HIL_ERROR_INTEGRITY


def sample(phase='back_to_back'):
    """Build valid phase telemetry and matching raw evidence for acceptance tests."""
    window = 5.0 if phase == 'cadenced' else .1
    return dict(metrics={'phase': phase, 'clock_hz': 600000000, 'runs': 10,
                         'timer_per_window_s': window, 'harness_latency_s': window / 10,
                         'energy_mj_per_inference': 1.25, 'rtc_clock_source': 'LSE',
                         'rtc_clock_hz_nominal': 32768, 'deadline_miss_count': 0, 'stop_entries': 9, 'slept_slots': 9},
                latency_s=.01, phase=phase, clock_mhz=600, runs=10, period_ms=500,
                dut_log=[f'measurement protocol output: {PROTOCOL}', 'cadence budget (ms): 500'],
                harness_log=['DONE'], require_harness=True)


def test_both_phases_accept_units():
    """Both phases must accept valid telemetry and retain consistent energy units."""
    for phase in ('back_to_back', 'cadenced'):
        result = validate_measurement(**sample(phase))
        assert result['instrumentation_valid'] and result['deployment_eligible']
        assert result['energy_mj_aggregate'] == 12.5


@pytest.mark.parametrize('field,value,reason', [
    ('clock_hz', 400000000, 'clock_matches'), ('runs', 9, 'runs_match'),
    ('phase', 'cadenced', 'phase_matches'), ('energy_mj_per_inference', float('nan'), 'energy_positive_finite'),
    ('energy_mj_per_inference', 0, 'energy_positive_finite'),
    ('timer_per_window_s', -1, 'rtc_window_positive_finite'),
    ('harness_latency_s', .02, 'window_sources_agree'),
    ('rtc_clock_source', 'LSI', 'rtc_lse')])
def test_invalid_telemetry_rejected(field, value, reason):
    """Malformed telemetry must report the corresponding instrumentation check."""
    args = sample(); args['metrics'][field] = value
    assert reason in validate_measurement(**args)['instrumentation_reasons']


def test_old_firmware_and_wrong_budget_rejected():
    """Protocol identity and requested cadence budget must both be present."""
    args = sample(); args['dut_log'] = []
    assert {'protocol_matches', 'period_matches'} <= set(validate_measurement(**args)['instrumentation_reasons'])


def test_deadline_infeasible_is_not_instrumentation_failure():
    """Valid deadline misses must remain infeasibility rather than instrumentation failure."""
    args = sample('cadenced'); args['metrics']['deadline_miss_count'] = 1
    args['latency_s'] = .6
    result = validate_measurement(**args)
    assert result['instrumentation_valid'] and not result['deployment_eligible']
    assert set(result['infeasibility_reasons']) == {'active_latency_exceeds_budget', 'deadline_miss'}
    args['metrics']['timer_per_window_s'] = 6; args['metrics']['harness_latency_s'] = .6
    assert 'cadence_window_outside_budget' in validate_measurement(**args)['infeasibility_reasons']


def test_missing_deadline_telemetry_is_instrumentation_failure():
    """Invalid deadline counts must invalidate measurement instrumentation."""
    args = sample('cadenced'); args['metrics']['deadline_miss_count'] = -1
    assert 'deadline_count_valid' in validate_measurement(**args)['instrumentation_reasons']


def test_unique_raw_records_accept_and_reject(tmp_path, monkeypatch):
    """Accepted and rejected attempts must retain separate raw evidence records."""
    monkeypatch.chdir(tmp_path)
    device = STM32NucleoN657X0QDevice(device_options={'latency_budget_ms': 500})
    args = sample()
    def unchecked(**kwargs):
        """Supply deterministic telemetry and populate the attempt evidence record."""
        kwargs['_record'].update(expected_runs=10, dut_log=args['dut_log'], harness_log=args['harness_log'])
        return DeviceMetrics(1, 2, .01, 3, HIL_ERROR_OK, power_metrics=args['metrics'])
    with patch.object(device, '_evaluate_single_phase_unchecked', side_effect=unchecked):
        good = device._evaluate_single_phase(dirpath=tmp_path, phase='back_to_back', harness_serial_port='fake')
        args['metrics']['clock_hz'] = 200000000
        bad = device._evaluate_single_phase(dirpath=tmp_path, phase='back_to_back', harness_serial_port='fake')
    assert good.error_code == HIL_ERROR_OK and bad.error_code == HIL_ERROR_INTEGRITY
    paths = list((tmp_path/'outputs/hil_measurements').glob('*.json')); assert len(paths) == 2
    records = [json.loads(path.read_text()) for path in paths]
    assert all(record['dut_log'] and record['harness_log'] for record in records)
    assert {record['acceptance']['instrumentation_valid'] for record in records} == {True, False}


def test_cadence_failure_propagates_without_base_latency_fallback():
    """Cadence failure must propagate instead of reusing base-phase latency."""
    device = STM32NucleoN657X0QDevice(device_options={'runtime_mode': 'cadenced'})
    base = DeviceMetrics(1, 2, .01, 3, HIL_ERROR_OK, power_metrics={})
    failed = DeviceMetrics(1, 2, -1, 3, HIL_ERROR_LATENCY,
                           power_metrics={'backend_error_kind': 'measurement_integrity', 'measurement_deployment_eligible': False})
    with patch.object(device, '_evaluate_single_phase', side_effect=[base, failed]):
        result = device.evaluate(dirpath='unused', arena_kb=-1, window_size=200, num_channels=10)
    assert result.error_code == HIL_ERROR_LATENCY and result.latency_s == -1
    assert result.power_metrics['measurement_deployment_eligible'] is False


def test_cadence_keeps_both_records_and_base_infeasibility():
    """Cadence results must retain both phase records and any base infeasibility."""
    device = STM32NucleoN657X0QDevice(device_options={'runtime_mode': 'cadenced'})
    base = DeviceMetrics(1, 2, .6, 3, HIL_ERROR_OK,
                        power_metrics={'measurement_deployment_eligible': False,
                                       'measurement_infeasibility_reasons': ['active_latency_exceeds_budget'],
                                       'measurement_record_paths': ['base.json']})
    cadence = DeviceMetrics(1, 2, .4, 3, HIL_ERROR_OK,
                           power_metrics={'measurement_deployment_eligible': True,
                                          'measurement_infeasibility_reasons': [],
                                          'measurement_record_paths': ['cadence.json']})
    with patch.object(device, '_evaluate_single_phase', side_effect=[base, cadence]):
        result = device.evaluate(dirpath='unused', arena_kb=-1, window_size=200, num_channels=10)
    assert result.error_code == HIL_ERROR_OK
    assert result.power_metrics['measurement_record_paths'] == ['base.json', 'cadence.json']
    assert result.power_metrics['measurement_deployment_eligible'] is False
    assert result.power_metrics['measurement_infeasibility_reasons'] == ['active_latency_exceeds_budget']


def test_no_harness_uses_rtc_mean_not_aggregate(tmp_path, monkeypatch):
    """Without a harness, reported latency must be the RTC mean per invocation."""
    monkeypatch.chdir(tmp_path)
    device = STM32NucleoN657X0QDevice(device_options={'latency_budget_ms': 500})
    args = sample()
    def unchecked(**kwargs):
        """Supply deterministic telemetry and populate the attempt evidence record."""
        kwargs['_record'].update(expected_runs=10, dut_log=args['dut_log'], harness_log=[])
        return DeviceMetrics(1, 2, .1, 3, HIL_ERROR_OK, power_metrics=args['metrics'])
    with patch.object(device, '_evaluate_single_phase_unchecked', side_effect=unchecked):
        result = device._evaluate_single_phase(dirpath=tmp_path, phase='back_to_back')
    assert result.error_code == HIL_ERROR_OK and result.latency_s == .01


def test_integrity_remeasure_once_same_compile_and_keeps_both_records():
    """One integrity retry must reuse the compiled candidate and preserve both records."""
    device = STM32NucleoN657X0QDevice()
    compiled = object()
    failed = DeviceMetrics(1, 2, -1, 3, HIL_ERROR_INTEGRITY,
                           power_metrics={'measurement_record_paths': ['first.json']})
    success = DeviceMetrics(1, 2, .01, 3, HIL_ERROR_OK,
                            power_metrics={'measurement_record_paths': ['second.json']})
    calls = []
    def attempt(**kwargs):
        """Fail the first attempt and require the retry to reuse its compiled build."""
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs['_retry_state']['compile_result'] = compiled
            return failed
        assert kwargs['_reuse_compile_result'] is compiled
        return success
    with patch.object(device, '_evaluate_phase_attempt', side_effect=attempt):
        result = device._evaluate_single_phase(dirpath='unused')
    assert len(calls) == 2 and result.error_code == HIL_ERROR_OK
    assert not hasattr(device, '_phase_compile_result')
    assert result.power_metrics['measurement_record_paths'] == ['first.json', 'second.json']
    assert result.power_metrics['measurement_remeasure_count'] == 1


def test_exhausted_integrity_stops_at_two_and_infeasible_is_not_retried():
    """Integrity recovery must stop after two attempts and skip valid infeasible results."""
    device = STM32NucleoN657X0QDevice()
    failure = DeviceMetrics(1, 2, -1, 3, HIL_ERROR_INTEGRITY, power_metrics={})
    def fail(**kwargs):
        """Simulate exhausted instrumentation recovery after a successful compile."""
        kwargs['_retry_state']['compile_result'] = object()
        return failure
    with patch.object(device, '_evaluate_phase_attempt', side_effect=fail) as mock:
        assert device._evaluate_single_phase(dirpath='unused').error_code == HIL_ERROR_INTEGRITY
        assert mock.call_count == 2
    infeasible = DeviceMetrics(1, 2, .7, 3, HIL_ERROR_OK,
                               power_metrics={'measurement_deployment_eligible': False})
    with patch.object(device, '_evaluate_phase_attempt', return_value=infeasible) as mock:
        assert device._evaluate_single_phase(dirpath='unused') is infeasible
        assert mock.call_count == 1


def test_explicit_record_directory_and_original_exception_survives_write_failure(tmp_path, monkeypatch):
    """A final record write failure must preserve the original candidate exception."""
    records = tmp_path / 'server-models' / 'hil_measurements'
    device = STM32NucleoN657X0QDevice(device_options={'measurement_records_dir': records})
    real_open = Path.open
    def open_with_final_failure(path, mode='r', *args, **kwargs):
        """Fail record finalization while allowing initial evidence creation."""
        if path.parent == records and mode == 'w':
            raise OSError('disk full')
        return real_open(path, mode, *args, **kwargs)
    with patch.object(device, '_evaluate_single_phase_unchecked', side_effect=ValueError('original candidate error')), patch.object(Path, 'open', open_with_final_failure):
        with pytest.raises(ValueError, match='original candidate error'):
            device._evaluate_single_phase(dirpath='unused')
    assert len(list(records.glob('*.json'))) == 1


@pytest.mark.parametrize('token,expected_kind,expected_code,attempts', [
    ('STM32_AI_RUN=FAIL measurement_integrity:stop_sleep', 'measurement_integrity', HIL_ERROR_INTEGRITY, 2),
    ('STM32_AI_START=TIMEOUT', 'runtime_start_timeout', HIL_ERROR_INTEGRITY, 2),
    ('STM32_AI_RUN=FAIL batch=0', 'runtime_model', HIL_ERROR_LATENCY, 1),
    ('STM32_AI_RUN=FAIL measurement_integrity:stop_sleep_extra', 'runtime_model', HIL_ERROR_LATENCY, 1),
])
def test_real_parser_failure_routes_through_backend_recovery(tmp_path, token, expected_kind, expected_code, attempts):
    """Parsed firmware failures must trigger only the matching backend recovery path."""
    from types import SimpleNamespace
    from crest.microcontrollers import stm32_runtime
    device = STM32NucleoN657X0QDevice(device_options={'measurement_records_dir': tmp_path/'records'})
    compiled = SimpleNamespace(success=True, log='ok', flash_bytes=2, ram_bytes=1,
                               overflow_kind=None, build_dir=tmp_path/'Debug', arena_bytes=3)
    def parse_failure(*args, **kwargs):
        """Exercise production parsing of a firmware failure token."""
        return stm32_runtime.parse_stm32_runtime_lines(['STM32_AI_INIT=OK', 'DUT READY', token])
    module = 'crest.microcontrollers.stm32_nucleo_n657x0.'
    with patch.object(device, 'compile', return_value=compiled) as compile_mock, \
         patch(module+'stm32_cube_clt.resolve_elf_path', return_value=tmp_path/'app.elf'), \
         patch(module+'stm32_runtime.SerialMonitor'), \
         patch(module+'stm32_cube_clt.debug_load_elf', return_value='ok'), \
         patch(module+'stm32_runtime.execute_runtime_session', side_effect=parse_failure) as runtime_mock:
        result = device.evaluate(dirpath=tmp_path/'unstaged', arena_kb=-1, window_size=200, num_channels=10, serial_port='fake')
    assert result.error_code == expected_code
    assert result.power_metrics['backend_error_kind'] == expected_kind
    assert compile_mock.call_count == 1
    assert runtime_mock.call_count == attempts
    assert len(result.power_metrics['measurement_record_paths']) == attempts
    for filename in result.power_metrics['measurement_record_paths']:
        assert token in json.loads(Path(filename).read_text())['dut_log']


@pytest.mark.parametrize('kind', ['runtime_io', 'harness_ready_timeout', 'harness_done_timeout', 'runtime_start_timeout'])
def test_transport_failures_retry_same_build_and_keep_evidence(tmp_path, kind):
    """Transport failures must retry the same build once and retain both evidence files."""
    from types import SimpleNamespace
    from crest.microcontrollers import stm32_runtime
    device = STM32NucleoN657X0QDevice(device_options={'measurement_records_dir': tmp_path/'records'})
    compiled = SimpleNamespace(success=True, log='ok', flash_bytes=2, ram_bytes=1,
                               overflow_kind=None, build_dir=tmp_path/'Debug', arena_bytes=3)
    module = 'crest.microcontrollers.stm32_nucleo_n657x0.'
    failure = (RuntimeError('serial transport failed') if kind == 'runtime_io' else
               stm32_runtime.STM32RuntimeProtocolError(kind=kind, detail='transport failed', serial_log=['evidence']))
    with patch.object(device, 'compile', return_value=compiled) as compile_mock, \
         patch(module+'stm32_cube_clt.resolve_elf_path', return_value=tmp_path/'app.elf'), \
         patch(module+'stm32_runtime.SerialMonitor'), \
         patch(module+'stm32_cube_clt.debug_load_elf', return_value='ok'), \
         patch(module+'stm32_runtime.execute_runtime_session', side_effect=failure) as runtime_mock:
        result = device.evaluate(dirpath=tmp_path/'unstaged', arena_kb=-1, window_size=200, num_channels=10, serial_port='fake')
    assert result.error_code == HIL_ERROR_INTEGRITY
    assert compile_mock.call_count == 1 and runtime_mock.call_count == 2
    assert len(result.power_metrics['measurement_record_paths']) == 2
    assert result.power_metrics['backend_error_kind'] == kind


@pytest.mark.parametrize('misses', ['0.5', '-0.5'])
def test_fractional_deadline_telemetry_is_rejected_after_parsing(misses):
    """Fractional deadline counts must remain invalid after runtime parsing."""
    from crest.microcontrollers import stm32_runtime
    args = sample('cadenced')
    lines = args['dut_log'] + [
        'phase output: cadenced', 'runs: 10', 'clock hz output: 600000000',
        'rtc clock source output: LSE', 'rtc clock hz nominal output: 32768',
        'timer per inference output: 0.010000', 'timer per window output: 5.000000',
        'slept slots: 9', 'stop entries: 9',
        f'cadence deadline misses: {misses}', 'STM32_AI_RUN=OK',
    ]
    telemetry = stm32_runtime.parse_stm32_runtime_lines(lines)
    args['metrics'].update(telemetry.power_metrics)
    args['dut_log'] = lines
    result = validate_measurement(**args)
    assert args['metrics']['deadline_miss_count'] == float(misses)
    assert not result['instrumentation_valid']
    assert not result['deployment_eligible']
    assert 'deadline_count_valid' in result['instrumentation_reasons']


@pytest.mark.parametrize('period_ms,effective_ms', [(33.333, 33), (33.7, 34), (0.25, 1)])
def test_cadence_window_uses_firmware_period(period_ms, effective_ms):
    """Fractional and sub-millisecond budgets must match the programmed cadence."""
    args = sample('cadenced')
    args.update(period_ms=period_ms, latency_s=0.0001)
    window_s = args['runs'] * effective_ms / 1000
    args['metrics'].update(timer_per_window_s=window_s,
                           harness_latency_s=window_s / args['runs'])
    args['dut_log'][1] = f'cadence budget (ms): {effective_ms}'
    result = validate_measurement(**args)
    assert result['instrumentation_valid'] and result['deployment_eligible']
    # Rounding the cadence must not relax the independently requested active deadline.
    args['latency_s'] = period_ms / 1000 + 0.00001
    assert 'active_latency_exceeds_budget' in validate_measurement(**args)['infeasibility_reasons']
    args['metrics']['timer_per_window_s'] += 0.0001
    args['metrics']['harness_latency_s'] = args['metrics']['timer_per_window_s'] / args['runs']
    assert 'cadence_window_outside_budget' in validate_measurement(**args)['infeasibility_reasons']


@pytest.mark.parametrize('clock_mhz', [200, 300, 400, 600, 800])
def test_configured_integer_clock_is_exact_after_parsing(clock_mhz):
    """Supported integer clock telemetry needs no oscillator-drift tolerance."""
    from crest.microcontrollers import stm32_runtime
    args = sample()
    args['clock_mhz'] = clock_mhz
    telemetry = stm32_runtime.parse_stm32_runtime_lines([
        'phase output: back_to_back', 'runs: 10',
        f'clock hz output: {clock_mhz * 1000000}',
        'timer per inference output: 0.010000', 'STM32_AI_RUN=OK'])
    args['metrics']['clock_hz'] = telemetry.power_metrics['clock_hz']
    assert validate_measurement(**args)['checks']['clock_matches']
    args['metrics']['clock_hz'] -= 1
    assert not validate_measurement(**args)['checks']['clock_matches']
