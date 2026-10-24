# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Upload the same sketch to a Nano RP2040 Connect N times and log each try.

The sketch is compiled once; each iteration then runs ``arduino-cli upload``
with a hard timeout and records success, duration, whether the CDC port came
back, and the tail of the tool output.

``--mode plain`` (default) runs bare ``arduino-cli upload`` (which already
performs the core's 1200-bps touch) with no retries, to characterise the raw
behaviour. ``--mode robust`` uses ``ArduinoNanoRP2040Device.upload``, which
adds an explicit 1200-bps touch and ``--retries`` extra attempts.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.microcontrollers.arduino_nano_rp2040 import (  # noqa: E402
    ArduinoNanoRP2040Device,
    run_upload_once,
    wait_for_serial_port,
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
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--mode", choices=("plain", "robust"), default="plain")
    parser.add_argument("--retries", type=int, default=2, help="Extra attempts per upload in robust mode.")
    parser.add_argument("--timeout-s", type=float, default=90.0, help="Per-attempt upload timeout.")
    parser.add_argument("--port-wait-s", type=float, default=10.0)
    parser.add_argument("--arena-kb", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=201)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Compile once, upload ``--count`` times, write a TSV log.

    Returns
    -------
    int
        ``0`` when every upload succeeded, ``1`` otherwise.
    """
    args = parse_args()
    sketch_path = args.candidate_dir.resolve()
    device = ArduinoNanoRP2040Device(
        serial_port=args.port,
        device_options={
            "upload_timeout_s": args.timeout_s,
            "upload_retries": args.retries if args.mode == "robust" else 0,
            "port_wait_s": args.port_wait_s,
        },
    )
    print("compiling once ...", flush=True)
    compile_result = device.compile(
        sketch_path=sketch_path,
        arena_kb=args.arena_kb,
        window_size=args.window_size,
        num_channels=args.num_channels,
        build_defines=device.runtime_mode_build_defines(),
    )
    if not compile_result.success or compile_result.build_dir is None:
        print(compile_result.log)
        print("compile failed; nothing uploaded.")
        return 1

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_RESULTS_DIR / f"upload_stress_{args.mode}_{stamp}.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = ["iteration\tmode\tsuccess\tattempts\tduration_s\tport_back\tport_present_before\tlog_tail"]
    failures = 0
    for iteration in range(1, args.count + 1):
        port_present_before = Path(args.port).exists()
        started = time.monotonic()
        if args.mode == "plain":
            result = run_upload_once(
                sketch_path=sketch_path,
                build_dir=compile_result.build_dir,
                serial_port=args.port,
                timeout_s=args.timeout_s,
            )
            port_back = wait_for_serial_port(args.port, args.port_wait_s) if result.success else Path(args.port).exists()
            attempts = 1
        else:
            result = device.upload(sketch_path=sketch_path, build_dir=compile_result.build_dir, serial_port=args.port)
            port_back = Path(args.port).exists()
            attempts = result.log.count("--- upload attempt ")
        duration = time.monotonic() - started
        if not result.success:
            failures += 1
        tail = " | ".join(line.strip() for line in result.log.strip().splitlines()[-4:] if line.strip())
        rows.append(
            f"{iteration}\t{args.mode}\t{int(result.success)}\t{attempts}\t{duration:.1f}\t"
            f"{int(port_back)}\t{int(port_present_before)}\t{tail.replace(chr(9), ' ')}"
        )
        print(
            f"[{iteration}/{args.count}] success={result.success} attempts={attempts} "
            f"duration={duration:.1f}s port_back={port_back}",
            flush=True,
        )
        if not port_back:
            # Give a late re-enumeration one more chance before the next try.
            wait_for_serial_port(args.port, args.port_wait_s)
        output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"{args.count - failures}/{args.count} uploads succeeded; log: {output}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
