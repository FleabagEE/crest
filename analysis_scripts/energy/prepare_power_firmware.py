# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Build (and later flash) the USB power-measurement firmware variants.

Every variant is the staged latency sketch with ``CREST_POWER_STATE`` set
(see ``sketches/crest_inference_latency.ino``), running the SC_5 1s model:

| variant                         | board        | state        | NINA       |
|---------------------------------|--------------|--------------|------------|
| rp2040_inference                | RP2040       | inference    | free       |
| rp2040_listen                   | RP2040       | listen idle  | free       |
| rp2040_inference_nina_reset     | RP2040       | inference    | in reset   |
| rp2040_listen_nina_reset        | RP2040       | listen idle  | in reset   |
| esp32s3_inference               | ESP32-S3     | inference    | -          |
| esp32s3_listen                  | ESP32-S3     | listen idle  | -          |

The ESP32-S3 variants run at 240 MHz with Wi-Fi and Bluetooth off (the
backend's board options and the sketch's ESP32 setup).

Default: compile all variants and write ``results/power_firmware/manifest.json``.
``--upload VARIANT --port PORT`` flashes one compiled variant; nothing is
flashed without it. See README.md, "Power measurement".
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BENCH_DIR = REPO_ROOT / "analysis_scripts" / "rp2040_latency_bench"
for _path in (REPO_ROOT / "src", BENCH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bench_boards import make_device  # noqa: E402

MODEL = "DSCNN_64_SC_5_1s_int8"
CANDIDATE = REPO_ROOT / "models" / "rp2040_bench" / MODEL
OUT_DIR = SCRIPT_DIR / "results" / "power_firmware"
WINDOW, CHANNELS = 98, 64
# Arena sizes that worked for SC_5 1s in the latency bench (memory.md 9, 14).
ARENA_KB = {"rp2040": 46, "xiao_esp32s3": 51}
STATE_IDS = {"inference": 1, "listen": 2}
VARIANTS: Dict[str, Dict[str, Any]] = {
    "rp2040_inference": {"board": "rp2040", "state": "inference", "options": {}},
    "rp2040_listen": {"board": "rp2040", "state": "listen", "options": {}},
    "rp2040_inference_nina_reset": {"board": "rp2040", "state": "inference", "options": {"hold_nina_reset": True}},
    "rp2040_listen_nina_reset": {"board": "rp2040", "state": "listen", "options": {"hold_nina_reset": True}},
    "esp32s3_inference": {"board": "xiao_esp32s3", "state": "inference", "options": {"cpu_mhz": 240}},
    "esp32s3_listen": {"board": "xiao_esp32s3", "state": "listen", "options": {"cpu_mhz": 240}},
}


def stage(variant: str) -> Path:
    """Copy the SC_5 1s candidate into its own sketch folder for ``variant``."""
    target = OUT_DIR / variant
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(CANDIDATE, target, ignore=shutil.ignore_patterns(".arduino-build", "*.ino"))
    shutil.copyfile(REPO_ROOT / "sketches" / "crest_inference_latency.ino", target / f"{variant}.ino")
    return target


def build(variant: str, hold_s: int, settle_s: int) -> Dict[str, Any]:
    """Compile one variant; return its manifest entry."""
    spec = VARIANTS[variant]
    device = make_device(spec["board"], device_options=dict(spec["options"]))
    sketch = stage(variant)
    defines = dict(device.runtime_mode_build_defines())
    defines.update(CREST_POWER_STATE=STATE_IDS[spec["state"]], CREST_POWER_HOLD_S=hold_s,
                   CREST_POWER_SETTLE_S=settle_s, CREST_INFERENCE_RUNS=1)
    result = device.compile(sketch_path=sketch, arena_kb=ARENA_KB[spec["board"]], window_size=WINDOW,
                            num_channels=CHANNELS, build_defines=defines)
    return {
        "variant": variant, "board": spec["board"], "state": spec["state"], "model": MODEL,
        "nina_reset": bool(spec["options"].get("hold_nina_reset", False)),
        "arena_kb": ARENA_KB[spec["board"]], "defines": defines,
        "board_options": device._board_options, "fqbn": device.spec.fqbn,
        "compile_success": result.success, "flash_bytes": result.flash_bytes, "ram_bytes": result.ram_bytes,
        "sketch_dir": str(sketch.relative_to(REPO_ROOT)),
        "build_dir": str(result.build_dir.relative_to(REPO_ROOT)) if result.build_dir else None,
        "log_tail": "" if result.success else result.log[-1500:],
    }


def main() -> int:
    """Compile all variants, or flash one with ``--upload``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--hold-s", type=int, default=60)
    parser.add_argument("--settle-s", type=int, default=10)
    parser.add_argument("--upload", choices=sorted(VARIANTS), default=None,
                        help="Flash this already-compiled variant (needs --port).")
    parser.add_argument("--port", default=None)
    args = parser.parse_args()
    manifest_path = OUT_DIR / "manifest.json"

    if args.upload:
        if not args.port:
            parser.error("--upload needs --port")
        entry = json.loads(manifest_path.read_text())[args.upload]
        spec = VARIANTS[args.upload]
        device = make_device(spec["board"], serial_port=args.port, device_options=dict(spec["options"]))
        result = device.upload(sketch_path=REPO_ROOT / entry["sketch_dir"], build_dir=REPO_ROOT / entry["build_dir"],
                               serial_port=args.port)
        print(f"upload {args.upload}: success={result.success}")
        return 0 if result.success else 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for variant in args.variants:
        entry = build(variant, args.hold_s, args.settle_s)
        manifest[variant] = entry
        print(f"{variant:30s} compile={entry['compile_success']} flash={entry['flash_bytes']} ram={entry['ram_bytes']}",
              flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 0 if all(manifest[v]["compile_success"] for v in args.variants) else 1


if __name__ == "__main__":
    raise SystemExit(main())
