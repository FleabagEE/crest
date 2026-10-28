# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Build, flash and run ``sketches/logmel_golden`` on a Nano RP2040 Connect.

Stages the golden sketch with ``sketches/common`` and a generated
``golden_clips.h`` (the same clips ``golden_logmel.py`` uses), compiles it
with ``CREST_LOGMEL_FFT_BITS``, uploads it through ``ArduinoNanoRP2040Device``
(timeout + retries), captures the serial output until ``DONE``, saves it to
``results/board_dump_q<bits-1>_<timestamp>.txt``, prints the ``TIMING`` lines
and the compile memory figures, then runs ``golden_logmel.py --board-dump``
on the capture.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import serial

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (REPO_ROOT / "src", SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from crest.microcontrollers import arduino_base  # noqa: E402
from crest.microcontrollers.arduino_nano_rp2040 import (  # noqa: E402
    BOARD_FQBN,
    ArduinoNanoRP2040Device,
    wait_for_serial_port,
)
import golden_logmel  # noqa: E402

SKETCH_DIR = REPO_ROOT / "sketches" / "logmel_golden"


def stage(build_dir: Path, seconds: float, wavs: List[Path], training_repo: Path) -> Path:
    """Stage the golden sketch; return its directory."""
    target = build_dir / "logmel_golden"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copyfile(SKETCH_DIR / "logmel_golden.ino", target / "logmel_golden.ino")
    shutil.copytree(REPO_ROOT / "sketches" / "common", target / "common")
    clips = golden_logmel.synthetic_clips(seconds)
    for wav in wavs:
        clips[f"wav_{wav.stem}"] = golden_logmel.wav_clip(wav, seconds, training_repo)
    golden_logmel.write_clip_header(target / "golden_clips.h", clips)
    return target


def capture(port: str, timeout_s: float) -> List[str]:
    """Send START and read lines until DONE (or ERROR / timeout)."""
    lines: List[str] = []
    with serial.Serial(port, 115200, timeout=0.2) as dut:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            line = dut.readline().decode(errors="replace").strip()
            if line == "DUT READY":
                break
        dut.write(b"START\n")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = dut.readline().decode(errors="replace").strip()
            if not line or line == "DUT READY":
                continue
            lines.append(line)
            if line == "DONE" or line.startswith("ERROR"):
                break
    return lines


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--bits", type=int, default=32, choices=[16, 32])
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--wav", type=Path, nargs="*", default=[])
    parser.add_argument("--training-repo", type=Path, default=golden_logmel.DEFAULT_TRAINING_REPO)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Stage, build, flash, capture and compare."""
    args = parse_args()
    build_dir = golden_logmel.DEFAULT_BUILD_DIR / f"golden_q{args.bits - 1}"
    sketch = stage(build_dir, args.seconds, args.wav, args.training_repo)
    result = arduino_base.compile_sketch(sketch_path=sketch, fqbn=BOARD_FQBN,
                                         build_defines={"CREST_LOGMEL_FFT_BITS": args.bits})
    print(f"compile success={result.success} flash_bytes={result.flash_bytes} ram_bytes={result.ram_bytes}")
    if not result.success:
        print(result.log[-3000:])
        return 1
    if args.compile_only:
        return 0
    device = ArduinoNanoRP2040Device(serial_port=args.port)
    upload = device.upload(sketch_path=sketch, build_dir=result.build_dir, serial_port=args.port)
    if not upload.success:
        print(upload.log[-2000:])
        return 1
    wait_for_serial_port(args.port, 15)
    lines = capture(args.port, args.timeout_s)
    golden_logmel.DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dump = golden_logmel.DEFAULT_RESULTS_DIR / f"board_dump_q{args.bits - 1}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        if line.startswith(("INFO", "TIMING", "ERROR", "DONE")):
            print(line)
    print(f"dump: {dump} ({len(lines)} lines)")
    cmd = [sys.executable, str(SCRIPT_DIR / "golden_logmel.py"), "--board-dump", str(dump), "--bits", str(args.bits),
           "--seconds", str(args.seconds), "--training-repo", str(args.training_repo),
           "--json", str(dump.with_suffix(".json"))]
    for wav in args.wav:
        cmd += ["--wav", str(wav)]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
