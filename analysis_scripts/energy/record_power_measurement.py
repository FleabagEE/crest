# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Store ina228_reader.py summaries in board_power_measured.yaml.

Pass the reader's ``INA228_WINDOW ...`` lines (``--line``, repeatable, or
``--from-file`` with the reader's saved output). Each window becomes one
sample of ``--board`` / ``--state``, with the meter details and a timestamp.
The file's comment header is preserved.

Example (after copying the reader output from the Pi):
    python analysis_scripts/energy/record_power_measurement.py \\
        --board rp2040 --state inference_nina_reset --from-file rp2040_inference_nina_reset.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MEASURED = SCRIPT_DIR / "board_power_measured.yaml"
KEEP_FIELDS = ("power_w_mean", "power_w_std", "volts_mean", "amps_mean", "amps_std", "amps_min", "amps_max",
               "n", "seconds", "rate_hz", "power_resolution_w", "current_lsb_a", "die_temp_c", "shunt_ohms",
               "adc_range", "window", "t_start", "label")


def parse_window_line(line: str) -> Dict[str, Any]:
    """Parse one ``INA228_WINDOW key=value ...`` line into numbers/strings."""
    if not line.startswith("INA228_WINDOW "):
        raise ValueError(f"not an INA228_WINDOW line: {line[:60]!r}")
    fields: Dict[str, Any] = {}
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        try:
            fields[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            fields[key] = value
    return {k: fields[k] for k in KEEP_FIELDS if k in fields}


def split_header(text: str) -> tuple:
    """Return ``(comment_header, yaml_body)``; the header is everything before the first key."""
    lines = text.splitlines(keepends=True)
    header: List[str] = []
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            break
        header.append(line)
    return "".join(header), "".join(lines[len(header):])


def record(path: Path, board: str, state: str, windows: List[Dict[str, Any]], note: str = "") -> Dict[str, Any]:
    """Append windows to ``boards.<board>.states.<state>``; return the updated document."""
    header, body = split_header(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(body)
    states = data["boards"][board]["states"]
    if state not in states:
        raise KeyError(f"unknown state {state!r} for {board}; expected one of {sorted(states)}")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for window in windows:
        states[state] = (states[state] or []) + [dict(window, recorded_utc=stamp, note=note)]
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data


def main() -> int:
    """Record the given windows."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--board", required=True, choices=("rp2040", "esp32s3", "rpi4"))
    parser.add_argument("--state", required=True)
    parser.add_argument("--line", action="append", default=[], help="An INA228_WINDOW line (repeatable).")
    parser.add_argument("--from-file", type=Path, default=None, help="Reader output; INA228_WINDOW lines are used.")
    parser.add_argument("--note", default="", help="Free text, e.g. meter/breakout notes or ambient temperature.")
    parser.add_argument("--measured", type=Path, default=DEFAULT_MEASURED)
    args = parser.parse_args()
    lines = list(args.line)
    if args.from_file:
        lines += [l for l in args.from_file.read_text(encoding="utf-8").splitlines() if l.startswith("INA228_WINDOW ")]
    if not lines:
        parser.error("no INA228_WINDOW lines given")
    windows = [parse_window_line(l.strip()) for l in lines]
    record(args.measured, args.board, args.state, windows, args.note)
    watts = ", ".join(f"{w['power_w_mean'] * 1e3:.2f} mW" for w in windows)
    print(f"recorded {len(windows)} window(s) for {args.board}/{args.state}: {watts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
