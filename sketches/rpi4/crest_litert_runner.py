# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""CREST latency/memory runner for Linux boards (Raspberry Pi 4).

This is the Linux counterpart of ``sketches/crest_inference_latency.ino``.
The host copies it next to a ``.tflite`` model over SSH and runs it. It:

1. imports a TFLite interpreter (``ai_edge_litert``, then ``tflite_runtime``,
   then ``tensorflow.lite``) and records the process RSS;
2. loads the model, allocates tensors, and fills the input with synthetic
   uniform values in [0, 5), quantized for int8 inputs (the same dummy input
   as the MCU sketch);
3. runs ``--warmup`` untimed ``invoke()`` calls, then ``--runs`` timed calls
   with ``time.perf_counter_ns()``, printing nothing inside the timed loop;
4. prints one line per run in the RP2040 format
   (``board=... model=... run=<k> invoke_us=<int> preprocess_us=-1
   arena_used_bytes=-1 status=ok``), then ``key=value`` environment and memory
   lines, then ``timer output: <mean seconds>`` for CREST's parser.

Failures print ``status=infeasible reason=<reason>`` and exit non-zero
without a ``timer output:`` line.

Memory: ``peak_rss_delta_bytes`` is the peak RSS between model load and the
last timed run, minus the RSS just before the model is loaded (after the
runtime import). The peak is ``VmHWM`` after resetting it with
``/proc/self/clear_refs``; ``ru_maxrss`` is the fallback when that is not
allowed (then the import peak can mask the model peak, reported as
``peak_window=ru_maxrss``). It approximates model + interpreter + activation
memory.

Standard library + numpy only; Python 3.9+.
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

BOARD_ID = "rpi4"


def load_interpreter_class():
    """Return ``(Interpreter class, runtime label)`` from the first available package."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        import ai_edge_litert

        return Interpreter, f"ai_edge_litert {getattr(ai_edge_litert, '__version__', '?')}"
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        import tflite_runtime

        return Interpreter, f"tflite_runtime {getattr(tflite_runtime, '__version__', '?')}"
    except ImportError:
        pass
    import tensorflow as tf  # noqa: F401  (last resort; heavy)

    return tf.lite.Interpreter, f"tensorflow.lite {tf.__version__}"


def proc_status_bytes(field: str) -> int:
    """Return a ``/proc/self/status`` memory field (``VmRSS``, ``VmHWM``) in bytes."""
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return -1


def reset_peak_rss() -> bool:
    """Reset ``VmHWM`` to the current RSS (``clear_refs`` value 5, no root needed)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        return False
    return True


def read_text(path: str) -> str:
    """Read a small sysfs file, or return ``na``."""
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return "na"


def soc_temp_c() -> str:
    """Return the SoC temperature in degrees C, or ``na``."""
    raw = read_text("/sys/class/thermal/thermal_zone0/temp")
    return f"{int(raw) / 1000:.1f}" if raw.isdigit() else "na"


def throttled_flags() -> str:
    """Return ``vcgencmd get_throttled`` (e.g. ``0x0``), or ``na``."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "na"
    return out.stdout.strip().split("=")[-1] or "na"


def fill_input(interpreter, numpy) -> None:
    """Fill input 0 with uniform [0, 5) values, quantized for integer inputs."""
    detail = interpreter.get_input_details()[0]
    rng = numpy.random.default_rng(0)
    values = rng.uniform(0.0, 5.0, size=detail["shape"]).astype(numpy.float32)
    dtype = detail["dtype"]
    if dtype == numpy.float32:
        data = values
    elif dtype in (numpy.int8, numpy.uint8):
        scale, zero_point = detail["quantization"]
        info = numpy.iinfo(dtype)
        data = numpy.clip(numpy.round(values / (scale or 1.0)) + zero_point, info.min, info.max).astype(dtype)
    else:
        raise TypeError(f"unsupported input dtype {dtype}")
    interpreter.set_tensor(detail["index"], data)


def emit(line: str) -> None:
    """Print one output line and flush, so SSH sees it immediately."""
    print(line, flush=True)


def infeasible(model_id: str, reason: str, detail: str = "") -> int:
    """Print an infeasible line and return the exit code."""
    emit(
        f"board={BOARD_ID} model={model_id} run=0 invoke_us=-1 preprocess_us=-1 "
        f"arena_used_bytes=-1 status=infeasible reason={reason}"
    )
    if detail:
        emit(f"error_detail={detail.replace(chr(10), ' ')[:300]}")
    return 2


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Path to the .tflite model.")
    parser.add_argument("--model-id", default=None, help="Identifier echoed on every line.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-xnnpack", action="store_true", help="Disable the default XNNPACK delegate.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the benchmark and print results; return the process exit code."""
    args = parse_args(argv)
    model_id = args.model_id or os.path.splitext(os.path.basename(args.model))[0]
    if args.runs < 1 or args.warmup < 0 or args.threads < 1:
        return infeasible(model_id, "bad_arguments")

    import numpy

    try:
        interpreter_cls, runtime = load_interpreter_class()
    except ImportError as exc:
        return infeasible(model_id, "no_tflite_runtime", str(exc))

    rss_before = proc_status_bytes("VmRSS")
    # Start the peak-RSS window here so the runtime import is not counted.
    peak_window = "vmhwm" if reset_peak_rss() else "ru_maxrss"
    temp_before = soc_temp_c()
    kwargs = {"model_path": args.model, "num_threads": args.threads}
    if args.no_xnnpack:
        # Experimental op resolver without default delegates (XNNPACK off).
        kwargs["experimental_op_resolver_type"] = _builtin_without_defaults(interpreter_cls)
    try:
        interpreter = interpreter_cls(**kwargs)
        interpreter.allocate_tensors()
    except Exception as exc:  # noqa: BLE001 - report any load/allocation failure
        return infeasible(model_id, "load_or_allocate_failed", repr(exc))
    try:
        fill_input(interpreter, numpy)
        for _ in range(args.warmup):
            interpreter.invoke()
    except Exception as exc:  # noqa: BLE001
        return infeasible(model_id, "warmup_invoke_failed", repr(exc))

    samples_ns = [0] * args.runs
    completed = 0
    failure = None
    perf_counter_ns = time.perf_counter_ns
    invoke = interpreter.invoke
    for i in range(args.runs):
        start = perf_counter_ns()
        try:
            invoke()
        except Exception as exc:  # noqa: BLE001
            failure = repr(exc)
            break
        samples_ns[i] = perf_counter_ns() - start
        completed += 1

    if peak_window == "vmhwm":
        peak_rss = proc_status_bytes("VmHWM")
    else:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    for i in range(completed):
        emit(
            f"board={BOARD_ID} model={model_id} run={i + 1} invoke_us={samples_ns[i] // 1000} "
            f"preprocess_us=-1 arena_used_bytes=-1 status=ok"
        )
    if failure is not None:
        return infeasible(model_id, "invoke_failed", failure)

    cpu = "/sys/devices/system/cpu/cpu0/cpufreq/"
    emit(f"runtime={runtime.replace(' ', '@')}")
    emit(f"num_threads={args.threads} xnnpack={'off' if args.no_xnnpack else 'default'}")
    emit(f"python={platform.python_version()} machine={platform.machine()} kernel={platform.release()}")
    emit(
        f"cpu_governor={read_text(cpu + 'scaling_governor')} cpu_cur_khz={read_text(cpu + 'scaling_cur_freq')} "
        f"cpu_max_khz={read_text(cpu + 'scaling_max_freq')}"
    )
    emit(f"temp_before_c={temp_before} temp_after_c={soc_temp_c()} throttled={throttled_flags()}")
    emit(
        f"rss_before_load_bytes={rss_before} peak_rss_bytes={peak_rss} "
        f"peak_rss_delta_bytes={peak_rss - rss_before} peak_window={peak_window} "
        f"model_file_bytes={os.path.getsize(args.model)}"
    )
    mean_s = statistics.fmean(samples_ns) / 1e9
    emit(f"runs: {completed}")
    emit(f"timer output: {mean_s:.9f}")
    return 0


def _builtin_without_defaults(interpreter_cls):
    """Return the ``BUILTIN_WITHOUT_DEFAULT_DELEGATES`` resolver for this runtime."""
    module = sys.modules[interpreter_cls.__module__]
    return module.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES


if __name__ == "__main__":
    sys.exit(main())
