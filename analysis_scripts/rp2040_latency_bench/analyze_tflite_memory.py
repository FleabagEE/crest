# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Per-layer MACs and a predicted TFLM arena / static RAM for .tflite models.

Run this before flashing to predict which candidates fit the RP2040 arena and
RAM limits. The arena estimate is

    peak live activations + conv scratch + 8 B per conv output channel + 2 KiB

- Peak live activations: the largest sum of activation tensors alive at one
  op. A tensor is alive from the op that produces it to its last consumer.
  Weights and other constants live in flash and are excluded.
- Conv scratch: CMSIS-NN ``arm_convolve_s8`` on a core without MVE needs
  ``2 * in_ch * kh * kw`` int16 values, planned with the peak op.
- 8 B per output channel: per-channel requantization multiplier and shift.
- 2 KiB: fixed interpreter overhead.

On the Nano RP2040 Connect this was within 0.5-3.5% of the measured
``arena_used_bytes`` (always slightly above) for seven models. Predicted
static RAM is ``--static-ram-base`` (mbed + TFLM + sketch, measured
2026-10-24 with 100 timed runs) plus the arena.

Arguments are candidate directories (containing ``g_model.tflite``) or
``.tflite`` files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FIXED_OVERHEAD_BYTES = 2048
PER_CHANNEL_QPARAM_BYTES = 8
DEFAULT_STATIC_RAM_BASE = 50_512
ARENA_MARGIN = 1.10


def _resolve_tflite(path: Path) -> Path:
    """Return the .tflite file for a candidate directory or file path.

    Parameters
    ----------
    path : Path
        Candidate directory or ``.tflite`` file.

    Returns
    -------
    Path
        The ``.tflite`` file.
    """
    return path / "g_model.tflite" if path.is_dir() else path


def analyze_tflite(path: Path) -> Dict[str, Any]:
    """Compute per-layer MACs, peak activations and the arena estimate.

    Parameters
    ----------
    path : Path
        Candidate directory or ``.tflite`` file.

    Returns
    -------
    dict[str, Any]
        Model summary. ``window`` and ``channels`` are input dims 1 and 2
        (frames and mel bins for the audio models).
    """
    import numpy as np
    import tensorflow as tf

    tflite_path = _resolve_tflite(Path(path))
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    ops = interpreter._get_ops_details()
    graph_inputs = {d["index"] for d in interpreter.get_input_details()}
    graph_outputs = {d["index"] for d in interpreter.get_output_details()}
    activations = {o for op in ops for o in op["outputs"]} | graph_inputs

    def nbytes(index: int) -> int:
        tensor = tensors[index]
        return int(np.prod(tensor["shape"])) * np.dtype(tensor["dtype"]).itemsize

    first_use = {index: -1 for index in graph_inputs}
    last_use: Dict[int, int] = {index: len(ops) for index in graph_outputs}
    for k, op in enumerate(ops):
        for index in op["outputs"]:
            first_use.setdefault(index, k)
        for index in op["inputs"]:
            if index in activations:
                last_use[index] = max(last_use.get(index, k), k)

    layers: List[Dict[str, Any]] = []
    peak_bytes, peak_op = 0, 0
    for k, op in enumerate(ops):
        name = op["op_name"]
        out_shape = [int(v) for v in tensors[op["outputs"][0]]["shape"]]
        in_shape = [int(v) for v in tensors[op["inputs"][0]]["shape"]]
        macs, detail, scratch = 0, "", 0
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            weights = tensors[op["inputs"][1]]["shape"]
            kh, kw = int(weights[1]), int(weights[2])
            out_h, out_w, out_c = out_shape[1:4]
            per_output = kh * kw * (int(weights[3]) if name == "CONV_2D" else 1)
            macs = out_h * out_w * out_c * per_output
            detail = f"{kh}x{kw}"
            if name == "CONV_2D":
                scratch = 2 * in_shape[-1] * kh * kw * 2
        elif name == "FULLY_CONNECTED":
            weights = tensors[op["inputs"][1]]["shape"]
            macs = int(weights[0]) * int(weights[1])
            detail = f"{int(weights[1])}->{int(weights[0])}"
        live = sum(nbytes(t) for t in activations if first_use.get(t, len(ops)) <= k <= last_use.get(t, -1))
        if live + scratch > peak_bytes:
            peak_bytes, peak_op = live + scratch, k
        layers.append(
            {"index": k, "op": name, "in": in_shape, "out": out_shape, "detail": detail,
             "macs": macs, "live_bytes": live, "scratch_bytes": scratch}
        )

    conv_channels = sum(layer["out"][-1] for layer in layers if layer["op"] in ("CONV_2D", "DEPTHWISE_CONV_2D"))
    input_shape = [int(v) for v in interpreter.get_input_details()[0]["shape"]]
    peak_layer = layers[peak_op]
    estimate = (
        peak_layer["live_bytes"]
        + peak_layer["scratch_bytes"]
        + PER_CHANNEL_QPARAM_BYTES * conv_channels
        + FIXED_OVERHEAD_BYTES
    )
    return {
        "tflite": str(tflite_path),
        "input_shape": input_shape,
        "window": input_shape[1],
        "channels": input_shape[2],
        "total_macs": sum(layer["macs"] for layer in layers),
        "peak_activation_bytes": peak_layer["live_bytes"],
        "peak_op": peak_op,
        "arena_estimate_bytes": estimate,
        "layers": layers,
    }


def recommended_arena_kb(arena_estimate_bytes: int, *, static_ram_base: int, ram_cap: int) -> int:
    """Pick an arena size: estimate + 10% margin, clipped to the RAM cap.

    Parameters
    ----------
    arena_estimate_bytes : int
        Output of :func:`analyze_tflite`.
    static_ram_base : int
        Static RAM without the arena.
    ram_cap : int
        ``DeviceSpec.max_ram_bytes``.

    Returns
    -------
    int
        Arena size in KiB. Never below the unmargined estimate; may exceed
        what the RAM cap allows, in which case the caller reports a predicted
        RAM overflow.
    """
    largest_kb = (ram_cap - static_ram_base) // 1024
    minimum_kb = math.ceil(arena_estimate_bytes / 1024)
    return max(minimum_kb, min(math.ceil(arena_estimate_bytes * ARENA_MARGIN / 1024), largest_kb))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", type=Path, help="Candidate directories or .tflite files.")
    parser.add_argument("--static-ram-base", type=int, default=DEFAULT_STATIC_RAM_BASE)
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--quiet-layers", action="store_true", help="Print only the per-model summary line.")
    return parser.parse_args()


def main() -> int:
    """Analyze each model and print per-layer and summary tables.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    from crest.microcontrollers.arduino_nano_rp2040 import BOARD_DEFAULT_SPEC

    ram_cap = int(BOARD_DEFAULT_SPEC.max_ram_bytes)
    results: Dict[str, Any] = {}
    for model_path in args.models:
        name = model_path.name if model_path.is_dir() else model_path.stem
        result = analyze_tflite(model_path)
        arena_kb = recommended_arena_kb(
            result["arena_estimate_bytes"], static_ram_base=args.static_ram_base, ram_cap=ram_cap
        )
        predicted_ram = args.static_ram_base + arena_kb * 1024
        result.update(
            recommended_arena_kb=arena_kb,
            predicted_static_ram_bytes=predicted_ram,
            predicted_fits_ram=predicted_ram <= ram_cap,
        )
        results[name] = result
        print(
            f"\n### {name}: input={result['input_shape']} MACs={result['total_macs']:,} "
            f"peak_activations={result['peak_activation_bytes']:,} B (op {result['peak_op']}) "
            f"arena_estimate={result['arena_estimate_bytes']:,} B -> arena {arena_kb} KiB, "
            f"static RAM ~{predicted_ram:,} B ({'fits' if result['predicted_fits_ram'] else 'RAM overflow'} "
            f"vs {ram_cap:,})"
        )
        if not args.quiet_layers:
            for layer in result["layers"]:
                print(
                    f"  {layer['index']:2d} {layer['op']:18s} {str(layer['in']):20s} -> {str(layer['out']):18s} "
                    f"{layer['detail']:8s} MACs={layer['macs']:>12,} live={layer['live_bytes']:>9,}"
                )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"\njson: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
