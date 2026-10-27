# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Energy per inference and joint model-board selection from latency results.

Reads ``rp2040_summary.csv`` and ``rpi4_summary.csv`` (median ``invoke()``
latency per model, board and thread count) and ``board_power.yaml`` (power
ranges per board), and applies the energy model for a sensing spec
(window ``L`` seconds, stride ``S`` seconds, inference time ``t``):

    S <= L: E = P_active * t + P_listen * (S - t)
    S >  L: E = P_active * t + P_listen * L + P_sleep * (S - L - t)

The average power is ``E / S``. Every number is computed twice, with the low
and the high end of each power range.

A (model, board, threads) pair is evaluated for a spec only if the model's
input window equals ``L`` (1 s exports for L = 1 s, 2 s exports and the
baseline for L = 2 s). It is infeasible if its benchmark status is not
``ok``, if ``t > S``, or, when ``S > L``, if ``t > S - L`` (no time left to
listen for ``L`` and then compute; this is the case where the sleep term
would be negative).

Selection: for a minimum accuracy, the feasible pair with the lowest energy
wins and the next one is the runner-up. Accuracy is the int8 TFLite
**validation-fold** accuracy from each subnet's run.json (the held-out test
fold has not been evaluated). Pairs without an accuracy (the untrained
baseline) are never selected.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_POWER = SCRIPT_DIR / "board_power.yaml"
DEFAULT_RP2040_CSV = REPO_ROOT / "analysis_scripts" / "rp2040_latency_bench" / "results" / "rp2040_summary.csv"
DEFAULT_RPI4_CSV = REPO_ROOT / "analysis_scripts" / "rpi4_latency_bench" / "results" / "rpi4_summary.csv"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SPECS = ((1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 4.0))
DEFAULT_MIN_ACCURACIES = (0.64, 0.67, 0.69)
BOUNDS = ("low", "high")
BOARD_KEYS = {"nano_rp2040_connect": "rp2040", "rpi4": "rpi4"}


@dataclass(frozen=True)
class BoardPower:
    """Power ranges for one board, in watts, as ``(low, high)`` tuples."""

    active: Dict[int, Tuple[float, float]]
    listen: Tuple[float, float]
    sleep: Tuple[float, float]

    def active_for(self, threads: int) -> Tuple[float, float]:
        """Return ``P_active`` for a thread count (single-entry boards ignore it)."""
        if threads in self.active:
            return self.active[threads]
        if len(self.active) == 1:
            return next(iter(self.active.values()))
        raise KeyError(f"no P_active for threads={threads}")


@dataclass(frozen=True)
class Pair:
    """One measured (model, board, threads) result."""

    model: str
    board: str
    threads: int
    window_s: float
    latency_s: Optional[float]
    status: str
    val_acc: Optional[float]
    val_macro_f1: Optional[float]
    xnnpack: str = ""


@dataclass
class Estimate:
    """Energy result for one pair under one spec."""

    pair: Pair
    L: float
    S: float
    applicable: bool
    feasible: bool
    reason: str
    energy_j: Dict[str, float] = field(default_factory=dict)
    avg_power_w: Dict[str, float] = field(default_factory=dict)


def _range(value: Sequence[float]) -> Tuple[float, float]:
    """Validate a ``[low, high]`` watt range."""
    low, high = float(value[0]), float(value[1])
    if not 0.0 <= low <= high:
        raise ValueError(f"power range must satisfy 0 <= low <= high, got {value}")
    return low, high


def load_power(path: Path = DEFAULT_POWER) -> Dict[str, BoardPower]:
    """Load ``board_power.yaml`` into ``BoardPower`` objects.

    Parameters
    ----------
    path : Path
        YAML file.

    Returns
    -------
    dict[str, BoardPower]
        Keyed by board (``rp2040``, ``rpi4``).
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    boards: Dict[str, BoardPower] = {}
    for name, spec in data["boards"].items():
        active_spec = spec["P_active"]
        if "watts" in active_spec:
            active = {1: _range(active_spec["watts"])}
        else:
            active = {int(key.split("_")[1]): _range(v["watts"]) for key, v in active_spec.items()}
        listen = _range(spec["P_listen"]["watts"])
        sleep_spec = spec["P_sleep"]
        sleep = listen if sleep_spec.get("same_as") == "P_listen" else _range(sleep_spec["watts"])
        boards[name] = BoardPower(active=active, listen=listen, sleep=sleep)
    return boards


def _float_or_none(value: str) -> Optional[float]:
    """Parse a CSV cell, mapping empty to ``None``."""
    return float(value) if value not in ("", None) else None


def load_pairs(paths: Iterable[Path]) -> List[Pair]:
    """Read pairs from the RP2040 and Pi summary CSVs.

    Parameters
    ----------
    paths : iterable of Path
        Summary CSVs (``rp2040_summary.csv`` has no ``threads`` column; its
        rows count as 1 thread).

    Returns
    -------
    list[Pair]
        One per CSV row.
    """
    pairs: List[Pair] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                median_us = _float_or_none(row.get("median_us", ""))
                pairs.append(
                    Pair(
                        model=row["model"],
                        board=BOARD_KEYS.get(row["board"], row["board"]),
                        threads=int(row.get("threads") or 1),
                        window_s=float(row["stride_s"]),
                        latency_s=None if median_us is None else median_us / 1e6,
                        status=row.get("status", ""),
                        val_acc=_float_or_none(row.get("val_acc", "")),
                        val_macro_f1=_float_or_none(row.get("val_macro_f1", "")),
                        xnnpack=row.get("xnnpack", ""),
                    )
                )
    return pairs


def energy_per_inference(t: float, L: float, S: float, p_active: float, p_listen: float, p_sleep: float) -> float:
    """Energy in joules for one inference period.

    Parameters
    ----------
    t : float
        Inference time in seconds (must satisfy ``t <= S``, and ``t <= S - L``
        when ``S > L``).
    L, S : float
        Window length and stride in seconds.
    p_active, p_listen, p_sleep : float
        Power in watts.

    Returns
    -------
    float
        Joules per inference.

    Raises
    ------
    ValueError
        If the timing is infeasible.
    """
    if t < 0 or L <= 0 or S <= 0:
        raise ValueError("t must be >= 0 and L, S > 0")
    if t > S:
        raise ValueError(f"t={t} s exceeds stride S={S} s")
    if S <= L:
        return p_active * t + p_listen * (S - t)
    if t > S - L:
        raise ValueError(f"t={t} s exceeds S - L = {S - L} s; no time to listen and compute")
    return p_active * t + p_listen * L + p_sleep * (S - L - t)


def estimate(pair: Pair, L: float, S: float, power: Dict[str, BoardPower]) -> Estimate:
    """Evaluate one pair under one spec with the low and high power values."""
    result = Estimate(pair=pair, L=L, S=S, applicable=math.isclose(pair.window_s, L), feasible=False, reason="")
    if not result.applicable:
        result.reason = f"window {pair.window_s:g} s != L"
        return result
    if pair.status != "ok" or pair.latency_s is None:
        result.reason = f"status {pair.status or 'missing'}"
        return result
    t = pair.latency_s
    if t > S:
        result.reason = "too slow (t > S)"
        return result
    if S > L and t > S - L:
        result.reason = "too slow (t > S - L)"
        return result
    board = power[pair.board]
    active = board.active_for(pair.threads)
    for i, bound in enumerate(BOUNDS):
        joules = energy_per_inference(t, L, S, active[i], board.listen[i], board.sleep[i])
        result.energy_j[bound] = joules
        result.avg_power_w[bound] = joules / S
    result.feasible = True
    result.reason = "ok"
    return result


def select(estimates: Sequence[Estimate], min_accuracy: float, bound: str) -> List[Estimate]:
    """Return feasible, accurate-enough estimates sorted by energy for ``bound``.

    Ties on energy are broken by higher accuracy, then by name, so the order
    is deterministic. The first entry is the winner and the second the
    runner-up.
    """
    eligible = [
        e for e in estimates
        if e.feasible and e.pair.val_acc is not None and e.pair.val_acc >= min_accuracy
    ]
    return sorted(
        eligible,
        key=lambda e: (e.energy_j[bound], -(e.pair.val_acc or 0.0), e.pair.model, e.pair.board, e.pair.threads),
    )


def pair_label(pair: Pair) -> str:
    """Short human-readable pair name."""
    board = "RP2040" if pair.board == "rp2040" else f"Pi {pair.threads}T"
    return f"{pair.model} @ {board}"


def write_estimates_csv(path: Path, estimates: Sequence[Estimate]) -> None:
    """Write every applicable estimate (all specs) to one CSV."""
    columns = [
        "L_s", "S_s", "model", "board", "threads", "xnnpack", "latency_s", "feasible", "reason",
        "energy_mj_low", "energy_mj_high", "avg_power_mw_low", "avg_power_mw_high", "val_acc", "val_macro_f1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for e in estimates:
            if not e.applicable:
                continue
            writer.writerow({
                "L_s": e.L, "S_s": e.S, "model": e.pair.model, "board": e.pair.board,
                "threads": e.pair.threads, "xnnpack": e.pair.xnnpack, "latency_s": e.pair.latency_s,
                "feasible": e.feasible, "reason": e.reason,
                "energy_mj_low": round(e.energy_j["low"] * 1e3, 4) if e.feasible else "",
                "energy_mj_high": round(e.energy_j["high"] * 1e3, 4) if e.feasible else "",
                "avg_power_mw_low": round(e.avg_power_w["low"] * 1e3, 4) if e.feasible else "",
                "avg_power_mw_high": round(e.avg_power_w["high"] * 1e3, 4) if e.feasible else "",
                "val_acc": e.pair.val_acc if e.pair.val_acc is not None else "",
                "val_macro_f1": e.pair.val_macro_f1 if e.pair.val_macro_f1 is not None else "",
            })


def _fmt_choice(ranked: Sequence[Estimate], index: int, bound: str) -> str:
    """Format the winner (index 0) or runner-up (index 1) cell."""
    if len(ranked) <= index:
        return "none feasible" if index == 0 else "-"
    e = ranked[index]
    return f"{pair_label(e.pair)} ({e.energy_j[bound] * 1e3:,.1f} mJ, acc {e.pair.val_acc:.3f})"


def spec_report(estimates: Sequence[Estimate], L: float, S: float, min_accuracies: Sequence[float]) -> str:
    """Markdown for one spec: all applicable pairs, then the selections."""
    rows = [e for e in estimates if e.L == L and e.S == S and e.applicable]
    lines = [
        f"### L = {L:g} s, S = {S:g} s",
        "",
        "| Pair | t (ms) | Val acc | E low (mJ) | E high (mJ) | P avg low (mW) | P avg high (mW) | Feasible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(rows, key=lambda e: (not e.feasible, e.energy_j.get("low", math.inf), pair_label(e.pair))):
        t_ms = f"{e.pair.latency_s * 1e3:,.2f}" if e.pair.latency_s is not None else "-"
        acc = f"{e.pair.val_acc:.3f}" if e.pair.val_acc is not None else "n/a"
        if e.feasible:
            lines.append(
                f"| {pair_label(e.pair)} | {t_ms} | {acc} | {e.energy_j['low'] * 1e3:,.1f} | "
                f"{e.energy_j['high'] * 1e3:,.1f} | {e.avg_power_w['low'] * 1e3:,.1f} | "
                f"{e.avg_power_w['high'] * 1e3:,.1f} | yes |"
            )
        else:
            lines.append(f"| {pair_label(e.pair)} | {t_ms} | {acc} | - | - | - | - | no: {e.reason} |")
    lines += [
        "",
        "| Min val acc | Winner (low power) | Runner-up (low) | Winner (high power) | Runner-up (high) | Winner changes? |",
        "|---|---|---|---|---|---|",
    ]
    for min_acc in min_accuracies:
        low = select(rows, min_acc, "low")
        high = select(rows, min_acc, "high")
        changes = "-" if not low or not high else ("yes" if low[0].pair != high[0].pair else "no")
        lines.append(
            f"| {min_acc:.2f} | {_fmt_choice(low, 0, 'low')} | {_fmt_choice(low, 1, 'low')} | "
            f"{_fmt_choice(high, 0, 'high')} | {_fmt_choice(high, 1, 'high')} | {changes} |"
        )
    return "\n".join(lines)


def _parse_spec(text: str) -> Tuple[float, float]:
    """Parse ``L,S`` (seconds)."""
    left, right = text.split(",")
    return float(left), float(right)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--power", type=Path, default=DEFAULT_POWER)
    parser.add_argument("--csv", type=Path, nargs="+", default=[DEFAULT_RP2040_CSV, DEFAULT_RPI4_CSV])
    parser.add_argument("--spec", type=_parse_spec, nargs="+", default=list(DEFAULT_SPECS),
                        help="Specs as L,S in seconds (default: 1,1 1,2 2,2 2,4).")
    parser.add_argument("--min-acc", type=float, nargs="+", default=list(DEFAULT_MIN_ACCURACIES),
                        help="Minimum validation-fold accuracies (default: 0.64 0.67 0.69).")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    """Estimate every spec, write the CSV and the markdown report."""
    args = parse_args()
    power = load_power(args.power)
    pairs = load_pairs(args.csv)
    estimates = [estimate(pair, L, S, power) for L, S in args.spec for pair in pairs]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_estimates_csv(args.results_dir / "energy_estimates.csv", estimates)
    report = "\n\n".join(spec_report(estimates, L, S, args.min_acc) for L, S in args.spec)
    header = (
        "Accuracy is int8 validation-fold accuracy (fold 9); the held-out test fold is not evaluated.\n"
        "Low/high use the low/high ends of every power range in board_power.yaml.\n"
    )
    (args.results_dir / "energy_selection.md").write_text(header + "\n" + report + "\n", encoding="utf-8")
    print(header)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
