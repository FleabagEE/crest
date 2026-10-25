# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compile, upload and time one staged candidate on a Nano RP2040 Connect.

Uses the same ``ArduinoNanoRP2040Device`` compile/upload/measure calls as a
NAS HIL run, then writes a results file with the raw per-run telemetry lines
plus median and p95 ``invoke_us``.

With ``--arena-kb auto`` (the default) arenas from the RP2040 spec are tried
in ascending order until AllocateTensors() succeeds, so the result also
reports the smallest feasible arena in that sweep.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.microcontrollers.arduino_nano_rp2040 import (  # noqa: E402
    BOARD_ID,
    ArduinoNanoRP2040Device,
    parse_run_lines,
)

DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--arena-kb",
        default="auto",
        help="Arena size in KiB, or 'auto' to sweep the RP2040 spec upward (default).",
    )
    parser.add_argument("--runs", type=int, default=100, help="Timed runs (default 100).")
    parser.add_argument("--warmup", type=int, default=5, help="Discarded warm-up runs (default 5).")
    parser.add_argument("--window-size", type=int, default=201, help="Input frames (default 201).")
    parser.add_argument("--num-channels", type=int, default=64, help="Mel bins (default 64).")
    parser.add_argument("--serial-timeout-s", type=float, default=120.0)
    parser.add_argument("--dut-ready-timeout-s", type=float, default=12.0)
    parser.add_argument("--upload-retries", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true", help="Stop after compile (no board needed).")
    parser.add_argument("--output", type=Path, default=None, help="Results file path.")
    return parser.parse_args()


def percentile_nearest_rank(values: Sequence[int], pct: float) -> int:
    """Return the nearest-rank percentile of ``values``.

    Parameters
    ----------
    values : Sequence[int]
        Samples.
    pct : float
        Percentile in ``(0, 100]``.

    Returns
    -------
    int
        Sample at rank ``ceil(pct / 100 * n)``.
    """
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def infeasible_line(model_id: str, reason: str, arena_bytes: int) -> str:
    """Format a host-side infeasible line in the device telemetry format.

    Parameters
    ----------
    model_id : str
        Candidate identifier.
    reason : str
        Failure reason token.
    arena_bytes : int
        Arena size of the attempt.

    Returns
    -------
    str
        Telemetry-formatted line.
    """
    return (
        f"board={BOARD_ID} model={model_id} run=0 invoke_us=-1 preprocess_us=-1 "
        f"arena_used_bytes=-1 status=infeasible reason={reason} arena_bytes={arena_bytes}"
    )


_SUMMARY_FIELD_RE = re.compile(r"\b(\w+)=([-\w.]+)")


def parse_results_file(path: Path) -> Dict[str, Any]:
    """Parse a results file written by :func:`write_results`.

    Parameters
    ----------
    path : Path
        Results file.

    Returns
    -------
    dict[str, Any]
        ``model``, ``timed_runs``, ``warmup_runs``, the ``# summary:`` fields
        (``status``, ``reason``, ``arena_kb``, ``arena_used_bytes``,
        ``flash_bytes``, ``ram_bytes``, ``n``, ``median_invoke_us``,
        ``p95_invoke_us``, ...; numbers converted) and ``invoke_us``, the
        per-run samples from the raw ``status=ok`` lines.
    """
    text = path.read_text(encoding="utf-8")
    parsed: Dict[str, Any] = {"path": str(path)}
    header = re.search(r"^# board: (\S+)\s+model: (\S+)", text, re.MULTILINE)
    if header:
        parsed["board"], parsed["model"] = header.group(1), header.group(2)
    runs = re.search(r"^# timed_runs: (\d+)\s+warmup_runs: (\d+)", text, re.MULTILINE)
    if runs:
        parsed["timed_runs"], parsed["warmup_runs"] = int(runs.group(1)), int(runs.group(2))
    summary = text.split("# summary:", 1)[1].split("# raw", 1)[0] if "# summary:" in text else ""
    for key, value in _SUMMARY_FIELD_RE.findall(summary):
        try:
            parsed[key] = float(value) if "." in value else int(value)
        except ValueError:
            parsed[key] = None if value == "None" else value
    records = [record for record in parse_run_lines(text.splitlines()) if record.get("status") == "ok"]
    parsed["invoke_us"] = [int(record["invoke_us"]) for record in records]
    return parsed


def run_attempt(
    device: ArduinoNanoRP2040Device,
    args: argparse.Namespace,
    arena_kb: int,
) -> Dict[str, Any]:
    """Compile, upload and measure one arena size.

    Parameters
    ----------
    device : ArduinoNanoRP2040Device
        Configured device wrapper.
    args : argparse.Namespace
        Parsed arguments.
    arena_kb : int
        Arena size in KiB.

    Returns
    -------
    dict[str, Any]
        Attempt summary with ``status`` (``ok`` or ``infeasible``), memory
        figures, serial log and parsed run records.
    """
    sketch_path = args.candidate_dir.resolve()
    model_id = sketch_path.name
    arena_bytes = arena_kb * 1024
    build_defines = {"CREST_INFERENCE_RUNS": args.runs}
    build_defines.update(device.runtime_mode_build_defines())
    print(f"[arena {arena_kb} KiB] compiling ...", flush=True)
    compile_result = device.compile(
        sketch_path=sketch_path,
        arena_kb=arena_kb,
        window_size=args.window_size,
        num_channels=args.num_channels,
        build_defines=build_defines,
    )
    summary: Dict[str, Any] = {
        "arena_kb": arena_kb,
        "flash_bytes": compile_result.flash_bytes,
        "ram_bytes": compile_result.ram_bytes,
        "serial_log": [],
        "records": [],
        "upload_log": "",
        "compile_log": compile_result.log,
    }
    overflow = compile_result.overflow_kind
    if compile_result.ram_bytes is not None and compile_result.ram_bytes > device.spec.max_ram_bytes:
        overflow = "ram"
    if overflow is not None or not compile_result.success:
        reason = f"{overflow}_overflow" if overflow else "compile_failed"
        summary.update(status="infeasible", reason=reason, infeasible_line=infeasible_line(model_id, reason, arena_bytes))
        return summary
    if args.compile_only:
        summary.update(status="compiled", reason=None)
        return summary

    print(f"[arena {arena_kb} KiB] uploading to {args.port} ...", flush=True)
    started = time.monotonic()
    upload_result = device.upload(sketch_path=sketch_path, build_dir=compile_result.build_dir, serial_port=args.port)
    summary["upload_log"] = upload_result.log
    summary["upload_s"] = round(time.monotonic() - started, 1)
    if not upload_result.success:
        summary.update(status="infeasible", reason="upload_failed", infeasible_line=infeasible_line(model_id, "upload_failed", arena_bytes))
        return summary

    print(f"[arena {arena_kb} KiB] measuring ({args.warmup} warm-up + {args.runs} timed runs) ...", flush=True)
    measure_result = device.measure(
        serial_port=args.port,
        baud_rate=115200,
        serial_timeout_s=args.serial_timeout_s,
        measured_inference_runs=args.runs,
        dut_ready_timeout_s=args.dut_ready_timeout_s,
    )
    records = parse_run_lines(measure_result.serial_log)
    summary["serial_log"] = list(measure_result.serial_log)
    summary["records"] = records
    summary["latency_s"] = measure_result.latency_s
    ok_records = [record for record in records if record.get("status") == "ok"]
    if measure_result.latency_s is not None and ok_records:
        summary.update(status="ok", reason=None)
        return summary
    device_infeasible = [record for record in records if record.get("status") == "infeasible"]
    if device_infeasible:
        summary.update(status="infeasible", reason=device_infeasible[0].get("reason"), infeasible_line=device_infeasible[0]["raw"])
    elif measure_result.arena_error_line:
        summary.update(status="infeasible", reason="arena_too_small", infeasible_line=infeasible_line(model_id, "arena_too_small", arena_bytes))
    else:
        summary.update(status="infeasible", reason="timeout_no_telemetry", infeasible_line=infeasible_line(model_id, "timeout_no_telemetry", arena_bytes))
    return summary


def write_results(path: Path, args: argparse.Namespace, attempts: List[Dict[str, Any]]) -> None:
    """Write a plain-text results file.

    Parameters
    ----------
    path : Path
        Destination file.
    args : argparse.Namespace
        Parsed arguments, echoed in the header.
    attempts : list[dict[str, Any]]
        Attempt summaries in execution order.
    """
    final = attempts[-1]
    model_id = args.candidate_dir.resolve().name
    lines: List[str] = [
        "# CREST Nano RP2040 Connect latency/memory result",
        f"# generated_utc: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"# board: {BOARD_ID}  model: {model_id}  port: {args.port}",
        f"# timed_runs: {args.runs}  warmup_runs: {args.warmup}  preprocess_on_device: no (preprocess_us=-1)",
        "#",
        "# arena sweep:",
    ]
    for attempt in attempts:
        lines.append(
            f"#   arena_kb={attempt['arena_kb']} status={attempt['status']} reason={attempt.get('reason')} "
            f"flash_bytes={attempt.get('flash_bytes')} ram_bytes={attempt.get('ram_bytes')} "
            f"upload_s={attempt.get('upload_s')}"
        )
    lines.append("#")
    if final["status"] == "ok":
        ok_records = [record for record in final["records"] if record.get("status") == "ok"]
        invoke = [int(record["invoke_us"]) for record in ok_records]
        arena_used = ok_records[0].get("arena_used_bytes")
        lines.extend(
            [
                "# summary:",
                f"#   status=ok arena_kb={final['arena_kb']} arena_used_bytes={arena_used}",
                f"#   flash_bytes={final['flash_bytes']} ram_bytes={final['ram_bytes']} (arena included in ram_bytes)",
                f"#   n={len(invoke)} median_invoke_us={statistics.median(invoke):.1f} "
                f"p95_invoke_us={percentile_nearest_rank(invoke, 95)} "
                f"min_invoke_us={min(invoke)} max_invoke_us={max(invoke)} "
                f"mean_invoke_us={statistics.fmean(invoke):.1f}",
                f"#   crest_timer_output_s={final.get('latency_s')}",
                "#   p95 uses the nearest-rank method.",
                "#",
                "# raw per-run lines:",
            ]
        )
        lines.extend(record["raw"] for record in final["records"])
    elif final["status"] == "compiled":
        lines.append("# summary:")
        lines.append(
            f"#   status=compiled (compile-only) arena_kb={final['arena_kb']} "
            f"flash_bytes={final['flash_bytes']} ram_bytes={final['ram_bytes']}"
        )
    else:
        lines.append("# summary:")
        lines.append(f"#   status=infeasible reason={final.get('reason')}")
        lines.append("#")
        lines.append("# raw lines:")
        lines.append(final.get("infeasible_line", ""))
    lines.append("")
    lines.append("# full serial log of the final attempt:")
    lines.extend(f"#   {line}" for line in final.get("serial_log", []))
    if final.get("reason") in {"upload_failed", "compile_failed", "flash_overflow", "ram_overflow"}:
        lines.append("# tool log of the final attempt:")
        tool_log = final.get("upload_log") or final.get("compile_log") or ""
        lines.extend(f"#   {line}" for line in tool_log.splitlines()[-60:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the bench and write the results file.

    Returns
    -------
    int
        ``0`` on a feasible measurement (or successful compile-only run),
        ``1`` otherwise.
    """
    args = parse_args()
    device = ArduinoNanoRP2040Device(
        serial_port=args.port,
        device_options={"warmup_runs": args.warmup, "upload_retries": args.upload_retries},
    )
    if str(args.arena_kb).strip().lower() == "auto":
        arena_sweep: List[int] = [int(value) for value in device.spec.arena_sizes_kb]
    else:
        arena_sweep = [int(args.arena_kb)]

    attempts: List[Dict[str, Any]] = []
    for arena_kb in arena_sweep:
        attempt = run_attempt(device, args, arena_kb)
        attempts.append(attempt)
        print(f"[arena {arena_kb} KiB] -> {attempt['status']} {attempt.get('reason') or ''}", flush=True)
        # Only an undersized arena is worth retrying with a bigger one.
        if attempt.get("reason") != "arena_too_small":
            break

    output: Optional[Path] = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_RESULTS_DIR / f"{args.candidate_dir.resolve().name}_{stamp}.txt"
    write_results(output, args, attempts)
    print(f"results: {output}")
    return 0 if attempts[-1]["status"] in {"ok", "compiled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
