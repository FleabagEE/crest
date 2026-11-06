# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Offline triage of the course-project model grid (no hardware).

For every ``<run>/subnets/<subnet>/exports/<1s|2s>/int8.tflite`` under
``--models-root`` (the course repo's ``models/`` directory), this writes one
CSV row with:

- metadata from ``report.json`` (architecture, conv type, mel bins, width,
  window, parameters, int8 validation/test accuracy when present);
- the graph: input shape and quantization, operator list, MACs (per-layer,
  from the .tflite, cross-checked against the report's ``estimated_macs``);
- operator support for each runtime: Chirale TFLM's ``AllOpsResolver`` (RP2040),
  the esp32 core's esp-tflite-micro library and the op resolver registered
  in ``sketches/crest_inference_latency.ino`` (ESP32-S3), and optionally a real
  LiteRT load + invoke (``--litert-python``; Raspberry Pi 4 runtime);
- memory predictions: the RP2040-calibrated arena estimator
  (``rp2040_latency_bench/analyze_tflite_memory.py``) and the ESP-NN
  arena/estimate ratio range measured on the ESP32-S3 (0.99-2.08);
- a latency TRIAGE ESTIMATE per board, from MAC/s measured on the closest
  already-measured model (same width and window, standard-conv DS-CNN). It is
  not a result: depthwise layers get a range between the standard-conv rate
  (optimistic) and the overhead-bound depthwise baseline (pessimistic);
- a measurement plan (run / skip, timed and warm-up runs, minutes).

It also writes ``model_grid_summary.md`` (counts, op support, memory and
latency classes, plan with per-board time, 128-mel front-end notes).

Run in the ``crest`` env (TF 2.20 for the .tflite parser, librosa for the
128-mel notes).
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BENCH_DIR = REPO_ROOT / "analysis_scripts" / "rp2040_latency_bench"
for _path in (REPO_ROOT / "src", BENCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analyze_tflite_memory import (  # noqa: E402
    ARENA_MARGIN,
    DEFAULT_STATIC_RAM_BASE,
    analyze_tflite,
    recommended_arena_kb,
)

WINDOWS = ("1s", "2s")
WIDTHS = {"1": 1.0, "5": 0.5, "25": 0.25}

CHIRALE_ALL_OPS = (
    REPO_ROOT / "tools/arduino-user/libraries/Arduino_TensorFlowLite/src/tensorflow/lite/micro/all_ops_resolver.cpp"
)
ESP_TFLM_RESOLVER = (
    REPO_ROOT
    / "tools/arduino-data/packages/esp32/tools/esp32s3-libs/3.3.12/include/espressif__esp-tflite-micro"
    / "tensorflow/lite/micro/micro_mutable_op_resolver.h"
)
LATENCY_SKETCH = REPO_ROOT / "sketches" / "crest_inference_latency.ino"

# --- Memory constants (memory.md sections 9 and 14) ------------------------
RP2040_RAM_CAP = 225_000  # DeviceSpec.max_ram_bytes (CREST cap)
RP2040_RAM_HW = 270_336  # upload.maximum_data_size (linker limit)
ESP32_STATIC_BASE = 45_780  # latency sketch without the arena, measured 2026-11-02
ESP32_RAM_LIMIT = 327_680  # upload.maximum_data_size (internal SRAM)
ESP_NN_RATIO = (0.99, 2.08)  # measured arena_used / RP2040 estimate, 7 models

# --- Latency anchors: measured MAC/s (median), int8 -------------------------
# Standard-conv DS-CNN (DSCNN_64_SC_<w>), keyed by (width token, window).
# RP2040: rp2040_summary.csv 2026-10-25; ESP32-S3: esp32s3_summary.csv
# 2026-11-02; Pi 4: rpi4_summary.csv 2026-10-26 (1 thread, XNNPACK default).
SC_RATE = {
    "rp2040": {("25", "1s"): 7.317623e6, ("25", "2s"): 7.315612e6, ("5", "1s"): 7.568609e6,
               ("5", "2s"): 7.601195e6, ("1", "1s"): 6.927652e6, ("1", "2s"): 6.944542e6},
    "esp32s3": {("25", "1s"): 85.593734e6, ("25", "2s"): 84.370179e6, ("5", "1s"): 152.476136e6,
                ("5", "2s"): 155.479765e6, ("1", "1s"): 254.143222e6, ("1", "2s"): 266.046441e6},
    "rpi4_1t": {("25", "1s"): 3.937826e9, ("25", "2s"): 3.900379e9, ("5", "1s"): 5.981826e9,
                ("5", "2s"): 5.926762e9, ("1", "1s"): 7.761924e9, ("1", "2s"): 7.477674e9},
}
# Pessimistic rate for depthwise layers: the smallest CREST DS-CNN baseline
# (depthwise with 1 input channel, overhead-bound). Pi: XNNPACK was off.
DW_PESSIMISTIC_RATE = {"rp2040": 1.575609e6, "esp32s3": 4.809564e6, "rpi4_1t": 0.101172e9}
# Streaming log-mel CPU duty (frame time / 10 ms hop), 64 mels, q31.
LOGMEL_DUTY = {"rp2040": 0.4626, "esp32s3": 0.0624}

# Graph signatures (ops + tensor shapes, see graph_details) of the exports
# already measured on all three boards (course repo models/1s_2s/
# DSCNN_64_SC_all/20261023T213435Z_774e54f8_ffb532, memory.md sections 9-14).
# A retrained export with the same signature runs the same kernels on the same
# shapes, so its latency and arena are already known; only weights differ.
MEASURED_GRAPHS = {
    "110b008bf8bd663b": "DSCNN_64_SC_1_1s",
    "c3c8a3724206df7f": "DSCNN_64_SC_1_2s",
    "42cdf0a02f86f393": "DSCNN_64_SC_25_1s",
    "ed51e6634508dd42": "DSCNN_64_SC_25_2s",
    "ca85b69b7ef7fc34": "DSCNN_64_SC_5_1s",
    "19a1ab4f47a91157": "DSCNN_64_SC_5_2s",
}

# --- Plan timing (minutes; from the 2026-10-25 / 2026-11-02 bench logs) -----
SETUP_MIN = {"rp2040": 4.5, "esp32s3": 1.5, "rpi4": 0.5}  # compile + upload + probe (Pi: scp + probe)
FULL_OVERHEAD_MIN = {"rp2040": 1.0, "esp32s3": 0.5, "rpi4": 0.25}
ARENA_RETRY_MIN = {"rp2040": 4.0, "esp32s3": 1.5}


def camel(op: str) -> str:
    """``REDUCE_MAX`` -> ``ReduceMax`` (TFLM ``Add<Op>`` naming)."""
    return "".join(part.title() for part in op.split("_"))


def resolver_ops(path: Path, pattern: str) -> Optional[set]:
    """Return the ``Add<Op>`` names matched in ``path``, or ``None`` if missing."""
    if not path.exists():
        return None
    return set(re.findall(pattern, path.read_text(encoding="utf-8", errors="replace")))


def sketch_esp32_resolver(path: Path) -> Tuple[set, int]:
    """Ops registered for ESP32 in the latency sketch and the resolver capacity."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"MicroMutableOpResolver<(\d+)>\s+resolver;(.*?)#else", text, re.S)
    if not match:
        return set(), 0
    return set(re.findall(r"resolver\.Add(\w+)\(\)", match.group(2))), int(match.group(1))


def find_models(root: Path) -> List[Tuple[Path, Path, Path, str]]:
    """Return ``(tflite, subnet_dir, run_dir, window)`` for every 1 s / 2 s int8 export."""
    found = []
    for tflite in sorted(root.glob("*/subnets/*/exports/*/int8.tflite")):
        window = tflite.parent.name
        if window in WINDOWS:
            subnet_dir = tflite.parents[2]
            found.append((tflite, subnet_dir, subnet_dir.parents[1], window))
    return found


def load_reports(subnet_dir: Path, run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(run_report, subnet_report)``; subnet reports may be nested in the run report."""
    run = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    own = subnet_dir / "report.json"
    if own.exists():
        return run, json.loads(own.read_text(encoding="utf-8"))
    return run, run.get("subnets", {}).get(f"subnets/{subnet_dir.name}", {})


def int8_eval(sub: Dict[str, Any], window: str, split: str) -> Tuple[Optional[float], Optional[float]]:
    """int8 TFLite accuracy and macro-F1 for one split, or ``(None, None)``."""
    entry = ((sub.get("evaluation") or {}).get(window) or {}).get(split) or {}
    result = entry.get("int8_tflite") if isinstance(entry, dict) else None
    if not isinstance(result, dict) or result.get("accuracy") is None:
        return None, None
    return float(result["accuracy"]), float(result.get("macro_f1")) if result.get("macro_f1") is not None else None


def graph_details(path: Path) -> Dict[str, Any]:
    """Input quantization, largest CONV_2D filter, graph signature (ops + shapes)."""
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()
    inp = interpreter.get_input_details()[0]
    scale, zero_point = inp["quantization"]
    max_filter = 0
    signature = []
    for op in ops:
        shapes = [tensors[i]["shape"].tolist() for i in list(op["inputs"]) + list(op["outputs"]) if i >= 0]
        signature.append([op["op_name"], shapes])
        if op["op_name"] == "CONV_2D":
            max_filter = max(max_filter, int(np.prod(tensors[op["inputs"][1]]["shape"])))
    return {
        "input_dtype": np.dtype(inp["dtype"]).name,
        "input_scale": float(scale),
        "input_zero_point": int(zero_point),
        "max_conv_filter_bytes": max_filter,
        "signature": hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16],
    }


LITERT_PROBE = r"""
import collections, json, sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter
out = {}
for path in sys.argv[1:]:
    try:
        it = Interpreter(model_path=path, num_threads=1)
        it.allocate_tensors()
        d = it.get_input_details()[0]
        it.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
        it.invoke()
        ops = collections.Counter(o["op_name"] for o in it._get_ops_details())
        out[path] = {"ok": True, "delegate_partitions": ops.get("DELEGATE", 0)}
    except Exception as exc:
        out[path] = {"ok": False, "error": str(exc)[:200]}
print("LITERT_JSON " + json.dumps(out))
"""


def litert_probe(python: Path, paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Load + invoke every model with LiteRT (XNNPACK default) in ``python``."""
    proc = subprocess.run(
        [str(python), "-I", "-c", LITERT_PROBE, *map(str, paths)], capture_output=True, text=True, check=False
    )
    for line in proc.stdout.splitlines():
        if line.startswith("LITERT_JSON "):
            return json.loads(line[len("LITERT_JSON "):])
    raise RuntimeError(f"LiteRT probe failed: {proc.stderr[-500:]}")


def latency_range(board: str, width: str, window: str, conv_macs: int, dw_macs: int) -> Tuple[float, float]:
    """Optimistic / pessimistic latency (s): DW layers at the SC rate / baseline rate."""
    rate = SC_RATE[board][(width, window)]
    optimistic = (conv_macs + dw_macs) / rate
    pessimistic = conv_macs / rate + dw_macs / DW_PESSIMISTIC_RATE[board]
    return optimistic, pessimistic


def triage(optimistic: float, pessimistic: float, budget: float) -> str:
    """``miss`` if even the optimistic estimate exceeds the budget."""
    if optimistic > budget:
        return "miss"
    if pessimistic > budget:
        return "at_risk"
    return "meets"


def run_counts(latency_s: float) -> Tuple[int, int]:
    """(timed, warm-up) runs: 100 below 0.3 s, ~60 s of runs to 1 s, 10-30 above 1 s."""
    if latency_s < 0.3:
        return 100, 5
    if latency_s < 1.0:
        return max(30, min(100, math.ceil(60.0 / latency_s))), 5
    return max(10, min(30, math.ceil(30.0 / latency_s))), 2


def plan_minutes(board: str, latency_s: float, timed: int, warmup: int, retries: float) -> float:
    """Wall-clock minutes for one model on one board (probe + full run)."""
    probe_s = 12 * latency_s
    return (
        SETUP_MIN[board] + probe_s / 60.0 + FULL_OVERHEAD_MIN[board]
        + (timed + warmup) * latency_s / 60.0 + retries * ARENA_RETRY_MIN.get(board, 0.0)
    )


def classify_rp2040(arena_estimate: int) -> Tuple[int, int, str]:
    """Recommended arena KiB, predicted static RAM and fits / borderline / no."""
    arena_kb = recommended_arena_kb(arena_estimate, static_ram_base=DEFAULT_STATIC_RAM_BASE, ram_cap=RP2040_RAM_CAP)
    ram = DEFAULT_STATIC_RAM_BASE + arena_kb * 1024
    if DEFAULT_STATIC_RAM_BASE + arena_estimate * ARENA_MARGIN <= RP2040_RAM_CAP:
        status = "fits"
    elif DEFAULT_STATIC_RAM_BASE + arena_estimate <= RP2040_RAM_HW:
        status = "borderline"
    else:
        status = "no"
    return arena_kb, ram, status


def classify_esp32(arena_estimate: int, max_filter: int) -> Dict[str, Any]:
    """ESP-NN arena range from the measured ratio range; fits if the high end fits."""
    low = math.ceil(arena_estimate * ESP_NN_RATIO[0])
    high = math.ceil(arena_estimate * ESP_NN_RATIO[1])
    if ESP32_STATIC_BASE + high <= ESP32_RAM_LIMIT:
        status = "fits"
    elif ESP32_STATIC_BASE + low <= ESP32_RAM_LIMIT:
        status = "borderline"
    else:
        status = "no (PSRAM arena possible)"
    return {
        "esp32_arena_low_b": low,
        "esp32_arena_high_b": high,
        "esp32_arena_weightcopy_b": arena_estimate + max_filter,
        "esp32_static_ram_low_b": ESP32_STATIC_BASE + low,
        "esp32_static_ram_high_b": ESP32_STATIC_BASE + high,
        "esp32_mem": status,
    }


def analyze_model(tflite: Path, subnet_dir: Path, run_dir: Path, window: str, support: Dict[str, Any]) -> Dict[str, Any]:
    """Build one CSV row."""
    run, sub = load_reports(subnet_dir, run_dir)
    model_cfg = run.get("config", {}).get("model", {})
    audio_cfg = run.get("config", {}).get("audio", {})
    name = subnet_dir.name
    width_token = str((sub.get("metadata") or {}).get("effective_width") or name.rsplit("_", 1)[-1])
    analysis = analyze_tflite(tflite)
    details = graph_details(tflite)
    ops = collections.Counter(layer["op"] for layer in analysis["layers"])
    conv_macs = sum(l["macs"] for l in analysis["layers"] if l["op"] in ("CONV_2D", "FULLY_CONNECTED"))
    dw_macs = sum(l["macs"] for l in analysis["layers"] if l["op"] == "DEPTHWISE_CONV_2D")
    export_arch = (((sub.get("exports") or {}).get(window) or {}).get("int8") or {}).get("architecture") or {}
    val_acc, val_f1 = int8_eval(sub, window, "val")
    test_acc, test_f1 = int8_eval(sub, window, "test")
    stride = float(window.rstrip("s"))
    conv_type = model_cfg.get("conv_type") or model_cfg.get("mobile_spatial_conv")

    row: Dict[str, Any] = {
        "model": f"{name}_{window}",
        "run": run_dir.name,
        "architecture": model_cfg.get("architecture"),
        "conv_type": conv_type,
        "pooling": (sub.get("architecture") or {}).get("pooling") or model_cfg.get("pooling"),
        "n_mels": audio_cfg.get("n_mels") or analysis["channels"],
        "width": WIDTHS.get(width_token),
        "window_s": stride,
        "input_shape": "x".join(map(str, analysis["input_shape"])),
        "input_dtype": details["input_dtype"],
        "input_scale": round(details["input_scale"], 6),
        "input_zero_point": details["input_zero_point"],
        "params": export_arch.get("num_parameters") or (sub.get("architecture") or {}).get("num_parameters"),
        "macs": analysis["total_macs"],
        "macs_report": export_arch.get("estimated_macs"),
        "conv_fc_macs": conv_macs,
        "dw_macs": dw_macs,
        "n_ops": len(analysis["layers"]),
        "ops": "; ".join(f"{op} x{count}" for op, count in sorted(ops.items())),
        "tflite_bytes": tflite.stat().st_size,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "graph_signature": details["signature"],
        "same_graph_as_measured": MEASURED_GRAPHS.get(details["signature"], ""),
    }

    # Operator support.
    op_names = set(ops)
    chirale, esp_lib, (esp_sketch, esp_cap) = support["chirale"], support["esp_lib"], support["esp_sketch"]
    row["rp2040_ops"] = "n/a" if chirale is None else (
        "ok" if all(camel(o) in chirale for o in op_names)
        else "MISSING " + ",".join(sorted(o for o in op_names if camel(o) not in chirale)))
    row["esp32_ops_runtime"] = "n/a" if esp_lib is None else (
        "ok" if all(camel(o) in esp_lib for o in op_names)
        else "MISSING " + ",".join(sorted(o for o in op_names if camel(o) not in esp_lib)))
    missing_sketch = sorted(o for o in op_names if camel(o) not in esp_sketch)
    row["esp32_sketch_resolver_missing"] = ",".join(missing_sketch) or "none"
    litert = support["litert"].get(str(tflite)) if support["litert"] is not None else None
    row["litert_load_invoke"] = "not checked" if litert is None else ("ok" if litert["ok"] else "FAIL " + litert["error"])
    row["litert_xnnpack_partitions"] = None if not litert or not litert.get("ok") else litert["delegate_partitions"]

    # Memory.
    calibrated = row["architecture"] == "dscnn" and dw_macs == 0
    arena_kb, rp_ram, rp_status = classify_rp2040(analysis["arena_estimate_bytes"])
    row.update(
        arena_estimate_b=analysis["arena_estimate_bytes"],
        peak_activation_b=analysis["peak_activation_bytes"],
        estimator_calibrated_for_graph=calibrated,
        rp2040_arena_kb=arena_kb,
        rp2040_static_ram_b=rp_ram,
        rp2040_mem=rp_status,
    )
    row.update(classify_esp32(analysis["arena_estimate_bytes"], details["max_conv_filter_bytes"]))

    # Latency triage (estimate, not a result).
    for board, key in (("rp2040", "rp2040"), ("esp32s3", "esp32"), ("rpi4_1t", "pi1t")):
        optimistic, pessimistic = latency_range(board, width_token, window, conv_macs, dw_macs)
        row[f"{key}_t_opt_ms"] = round(optimistic * 1e3, 2)
        row[f"{key}_t_pess_ms"] = round(pessimistic * 1e3, 2)
        row[f"{key}_triage"] = triage(optimistic, pessimistic, stride)
        if board in LOGMEL_DUTY:
            row[f"{key}_triage_logmel_same_core"] = triage(optimistic, pessimistic, stride * (1 - LOGMEL_DUTY[board]))
    return row


def add_plan(row: Dict[str, Any]) -> None:
    """Add run/skip decisions, run counts and minutes per board."""
    uncalibrated = not row["estimator_calibrated_for_graph"]
    for board, key in (("rp2040", "rp2040"), ("esp32s3", "esp32"), ("rpi4", "pi1t")):
        t_mid = math.sqrt(row[f"{key}_t_opt_ms"] * row[f"{key}_t_pess_ms"]) / 1e3
        skip = None
        if row["same_graph_as_measured"]:
            skip = f"already measured (same graph as {row['same_graph_as_measured']})"
        elif board == "rp2040":
            if row["rp2040_mem"] == "no":
                skip = "predicted RAM overflow"
            elif row["rp2040_triage"] == "miss":
                skip = "predicted too slow (optimistic > stride)"
        if board == "rpi4" and row["litert_load_invoke"].startswith("FAIL"):
            skip = "LiteRT load failed"
        if skip:
            row[f"plan_{board}"] = f"skip: {skip}"
            row[f"plan_{board}_min"] = 0.0
            continue
        timed, warmup = run_counts(t_mid)
        if board == "rpi4":
            minutes = 2 * plan_minutes(board, t_mid, timed, warmup, 0)  # 1 and 4 threads
            row[f"plan_{board}"] = f"run t1+t4 N={timed} warm={warmup}"
        else:
            retries = 1.0 if (board == "esp32s3" or uncalibrated) else 0.0
            if board == "rp2040" and row["rp2040_mem"] == "borderline":
                retries += 1.0
            minutes = plan_minutes(board, t_mid, timed, warmup, retries)
            psram = board == "esp32s3" and row["esp32_mem"].startswith("no")
            row[f"plan_{board}"] = f"run N={timed} warm={warmup}" + (" (PSRAM arena)" if psram else "")
        row[f"plan_{board}_min"] = round(minutes, 1)


def logmel_128_notes() -> List[str]:
    """Table sizes for 64 vs 128 Slaney mel bins (training front end parameters)."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        return ["(librosa not available: table statistics skipped)"]
    lines = ["| n_mels | nonzero weights | 1-bin filters | max weight | min weight / max | sparse table bytes |",
             "|---|---|---|---|---|---|"]
    for n_mels in (64, 128):
        weights = librosa.filters.mel(sr=16000, n_fft=512, n_mels=n_mels, fmin=0, fmax=8000, htk=False, norm="slaney")
        nonzero = int((weights > 0).sum())
        one_bin = int(sum(1 for row in weights if (row > 0).sum() == 1))
        table_bytes = 3 * 2 * n_mels + 2 * nonzero  # start/length/offset uint16 + uint16 weights
        lines.append(
            f"| {n_mels} | {nonzero} | {one_bin} | {weights.max():.4f} | "
            f"{weights[weights > 0].min() / weights.max():.2e} | {table_bytes:,} |"
        )
    return lines


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write rows with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_acc(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def write_summary(rows: List[Dict[str, Any]], path: Path, extras: Dict[str, Any]) -> None:
    """Short markdown summary + measurement plan."""
    out: List[str] = ["# Model-grid triage (offline, no hardware)", ""]
    out.append(f"Models: {len(rows)} int8 exports ({extras['skipped_windows']} other-window exports ignored). "
               f"Source: `{extras['models_root']}`.")
    out.append("")
    out.append("Latency columns are a TRIAGE ESTIMATE from measured MAC/s of the standard-conv DS-CNN with the same "
               "width and window; depthwise layers are bracketed between that rate and the overhead-bound depthwise "
               "baseline. They are not measurements.")
    out.append("")
    out.append("## Operator support")
    unsupported = [r["model"] for r in rows if r["rp2040_ops"] != "ok" or r["esp32_ops_runtime"] != "ok"
                   or r["litert_load_invoke"].startswith("FAIL")]
    out.append(f"- Unsupported in a runtime: {', '.join(unsupported) if unsupported else 'none'}.")
    sketch_missing = sorted({op for r in rows for op in r["esp32_sketch_resolver_missing"].split(",") if op != "none"})
    out.append(f"- ESP32 latency sketch resolver (`MicroMutableOpResolver<{extras['esp_cap']}>`) lacks: "
               f"{', '.join(sketch_missing) or 'nothing'} -> register them before measuring (runtime has them).")
    out.append("")
    out.append("## Classes")
    for column in ("rp2040_mem", "esp32_mem", "rp2040_triage", "rp2040_triage_logmel_same_core",
                   "esp32_triage", "pi1t_triage"):
        counts = collections.Counter(r[column] for r in rows)
        out.append(f"- {column}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    out.append("")
    out.append("## Table")
    out.append("| model | val acc | MACs | RP2040 mem (static RAM) | ESP32 mem (static RAM range) | "
               "RP2040 ms (opt-pess) | triage | ESP32 ms | Pi 1T ms | plan RP2040 | plan ESP32 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['model']} | {fmt_acc(r['val_acc'])} | {r['macs']:,} | {r['rp2040_mem']} ({r['rp2040_static_ram_b']:,}) | "
            f"{r['esp32_mem']} ({r['esp32_static_ram_low_b']:,}-{r['esp32_static_ram_high_b']:,}) | "
            f"{r['rp2040_t_opt_ms']:,.0f}-{r['rp2040_t_pess_ms']:,.0f} | {r['rp2040_triage']} | "
            f"{r['esp32_t_opt_ms']:,.1f}-{r['esp32_t_pess_ms']:,.1f} | {r['pi1t_t_opt_ms']:.2f}-{r['pi1t_t_pess_ms']:.2f} | "
            f"{r['plan_rp2040']} | {r['plan_esp32s3']} |"
        )
    out.append("")
    out.append("## Measurement plan")
    for board in ("rp2040", "esp32s3", "rpi4"):
        run_rows = [r for r in rows if not r[f"plan_{board}"].startswith("skip")]
        minutes = sum(r[f"plan_{board}_min"] for r in rows)
        out.append(f"- **{board}**: run {len(run_rows)}, skip {len(rows) - len(run_rows)}; "
                   f"~{minutes / 60:.1f} h ({minutes:.0f} min) of bench time.")
        reasons = collections.Counter(r[f"plan_{board}"].split(" (same graph")[0]
                                      for r in rows if r[f"plan_{board}"].startswith("skip"))
        for reason, count in sorted(reasons.items()):
            out.append(f"  - {reason}: {count}")
    out.append("")
    out.append("## 128 mel bins: front-end table sizes (training parameters, Slaney)")
    out.extend(logmel_128_notes())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-root", type=Path, required=True,
                        help="Course repo models/ directory (e.g. a `git archive origin/main models` extract).")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--litert-python", type=Path, default=None,
                        help="Python with ai-edge-litert installed; loads + invokes every model with XNNPACK.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = find_models(args.models_root)
    if not models:
        print(f"no */subnets/*/exports/<1s|2s>/int8.tflite under {args.models_root}", file=sys.stderr)
        return 1
    skipped = len(list(args.models_root.glob("*/subnets/*/exports/*/int8.tflite"))) - len(models)
    esp_sketch, esp_cap = sketch_esp32_resolver(LATENCY_SKETCH)
    support = {
        "chirale": resolver_ops(CHIRALE_ALL_OPS, r"\bAdd(\w+)\(\);"),
        "esp_lib": resolver_ops(ESP_TFLM_RESOLVER, r"TfLiteStatus Add(\w+)\("),
        "esp_sketch": (esp_sketch, esp_cap),
        "litert": litert_probe(args.litert_python, [m[0] for m in models]) if args.litert_python else None,
    }
    rows = []
    for tflite, subnet_dir, run_dir, window in models:
        row = analyze_model(tflite, subnet_dir, run_dir, window, support)
        add_plan(row)
        rows.append(row)
        print(f"{row['model']:32s} MACs={row['macs']:>12,} rp2040={row['rp2040_mem']:10s} {row['rp2040_triage']:8s} "
              f"esp32={row['esp32_mem']}")
    write_csv(rows, args.out_dir / "model_grid_triage.csv")
    write_summary(rows, args.out_dir / "model_grid_summary.md",
                  {"skipped_windows": skipped, "models_root": args.models_root, "esp_cap": esp_cap})
    print(f"csv: {args.out_dir / 'model_grid_triage.csv'}\nsummary: {args.out_dir / 'model_grid_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
