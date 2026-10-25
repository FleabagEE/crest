# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Stage an existing .tflite as an RP2040 bench candidate directory.

The candidate gets ``g_model.tflite``, ``model.cc``/``model.h``, the staged
``crest_inference_latency.ino`` and a ``candidate.json``, the same layout
``export_smallest_dscnn.py`` produces.

For exports laid out as ``<subnet>/exports/<1s|2s>/int8.tflite`` with a
``<subnet>/run.json`` (the course-project training repo), ``candidate.json``
also records the subnet name, width token, channels, parameter count and the
validation accuracy / macro-F1 of the matching int8 export. Other layouts are
staged without that metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CANDIDATE_ROOT = REPO_ROOT / "models" / "rp2040_bench"


def run_json_metadata(tflite_path: Path) -> Optional[Dict[str, Any]]:
    """Read subnet metadata from ``<subnet>/run.json`` next to an export.

    Parameters
    ----------
    tflite_path : Path
        ``<subnet>/exports/<duration>/<variant>.tflite``.

    Returns
    -------
    dict[str, Any] | None
        Metadata, or ``None`` when the layout or file does not match.
    """
    duration_dir = tflite_path.parent
    if duration_dir.parent.name != "exports":
        return None
    run_json = duration_dir.parent.parent / "run.json"
    if not run_json.exists():
        return None
    run = json.loads(run_json.read_text(encoding="utf-8"))
    architecture = run.get("architecture", {})
    metadata = run.get("metadata", {})
    variant = f"{tflite_path.stem}_tflite"
    evaluation = run.get("evaluation", {}).get(duration_dir.name, {}).get("val", {}).get(variant, {})
    # Subnet run.json files omit the test-split flag; the parent run records it.
    parent_run_json = run_json.parent.parent.parent / "run.json"
    parent_state = (
        json.loads(parent_run_json.read_text(encoding="utf-8")).get("state", {}) if parent_run_json.exists() else {}
    )
    return {
        "source_run_json": str(run_json),
        "subnet": metadata.get("canonical_name", duration_dir.parent.parent.name),
        "width_token": metadata.get("effective_width"),
        "block_channels": architecture.get("channels"),
        "params": architecture.get("num_parameters"),
        "duration": duration_dir.name,
        "variant": tflite_path.stem,
        "val_accuracy": evaluation.get("accuracy"),
        "val_macro_f1": evaluation.get("macro_f1"),
        "held_out_test_evaluated": run.get("state", {}).get(
            "held_out_test_evaluated", parent_state.get("held_out_test_evaluated")
        ),
    }


def default_candidate_name(tflite_path: Path) -> str:
    """Derive ``<subnet>_<duration>_<variant>`` for the exports layout.

    Parameters
    ----------
    tflite_path : Path
        Source model.

    Returns
    -------
    str
        Candidate directory name.
    """
    if tflite_path.parent.parent.name == "exports":
        subnet = tflite_path.parent.parent.parent.name
        return f"{subnet}_{tflite_path.parent.name}_{tflite_path.stem}"
    return tflite_path.stem


def stage(tflite_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Stage one model.

    Parameters
    ----------
    tflite_path : Path
        Source ``.tflite``.
    output_dir : Path
        Candidate directory to create or refresh.

    Returns
    -------
    dict[str, Any]
        The written ``candidate.json`` content.
    """
    from crest.devices import _sync_arduino_sketch_variant_for_config
    from crest.hardware import convert_to_cpp_model
    from crest.microcontrollers.arduino_nano_rp2040 import BOARD_NAME

    from analyze_tflite_memory import analyze_tflite

    output_dir.mkdir(parents=True, exist_ok=True)
    staged = output_dir / "g_model.tflite"
    shutil.copyfile(tflite_path, staged)
    convert_to_cpp_model(tflite_path=staged, output_dir=output_dir)
    _sync_arduino_sketch_variant_for_config(
        {"device": {"name": BOARD_NAME}, "training": {"energy_aware": False, "input_mode": "uniform"}},
        output_dir,
    )
    analysis = analyze_tflite(staged)
    candidate: Dict[str, Any] = {
        "model_id": output_dir.name,
        "source_tflite": str(tflite_path),
        "source_sha256": hashlib.sha256(tflite_path.read_bytes()).hexdigest(),
        "tflite_bytes": staged.stat().st_size,
        "input_shape": analysis["input_shape"],
        "window": analysis["window"],
        "channels": analysis["channels"],
        "total_macs": analysis["total_macs"],
        "arena_estimate_bytes": analysis["arena_estimate_bytes"],
    }
    metadata = run_json_metadata(tflite_path)
    if metadata:
        candidate.update(metadata)
    (output_dir / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    return candidate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tflite", nargs="+", type=Path, help="Source .tflite files.")
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
        help="Parent directory for candidates (default: models/rp2040_bench).",
    )
    return parser.parse_args()


def main() -> int:
    """Stage every given model.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    for tflite_path in args.tflite:
        tflite_path = tflite_path.resolve()
        output_dir = args.candidate_root.resolve() / default_candidate_name(tflite_path)
        candidate = stage(tflite_path, output_dir)
        print(
            f"{candidate['model_id']}: {candidate['tflite_bytes']} B, input {candidate['input_shape']}, "
            f"MACs {candidate['total_macs']:,}, arena estimate {candidate['arena_estimate_bytes']:,} B "
            f"-> {output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
