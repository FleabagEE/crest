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
baseline for L = 2 s). It is infeasible if:

- its benchmark status is not ``ok``;
- ``t > S`` (it cannot keep up with the stride);
- ``S > L`` and ``t > S - L``. This is a **conservative assumption**: compute
  is not overlapped with listening, so each period is listen ``L``, compute
  ``t``, sleep ``S - L - t``, and the sleep term must not go negative. If a
  pair ever fails only this rule, revisit whether overlap is possible;
- the spec has a maximum response latency ``R`` and ``t > R``. Response
  latency is the time from the end of a window to the result. Today it is
  the measured ``invoke()`` time only; on-device log-mel preprocessing will
  add to it once implemented.

Designs (``--design``), all with ``t_inf`` = measured median ``invoke()``:

- ``inference_only``: ``t = t_inf`` (no preprocessing; the earlier results).
- ``naive``: the whole window's log-mel is computed after the window ends,
  ``t = window_s[L] + t_inf``; response latency ``= t``.
- ``streaming``: each frame is computed as its hop of audio arrives, so only
  the last frame remains after the window: ``t = frame_s + t_inf``, response
  latency ``= t``, and listening power becomes
  ``P_listen_eff = P_listen + duty * (P_active - P_listen)`` with
  ``duty = frame_s / hop_s``. A board whose frame time exceeds the hop cannot
  stream (infeasible).

Preprocessing times come from ``preprocessing.yaml``; a (board, threads)
without a measured entry is excluded from ``naive``/``streaming`` with an
explicit reason, never treated as zero.

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
DEFAULT_PREPROCESSING = SCRIPT_DIR / "preprocessing.yaml"
DESIGNS = ("inference_only", "streaming", "naive")
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


@dataclass(frozen=True)
class Preprocessing:
    """Measured log-mel time for one (board, threads)."""

    frame_s: float
    window_s: Dict[float, float]


def load_preprocessing(path: Path = DEFAULT_PREPROCESSING) -> Tuple[float, Dict[Tuple[str, int], Preprocessing]]:
    """Load ``preprocessing.yaml``; returns ``(hop_s, {(board, threads): Preprocessing})``.

    Entries marked ``pending`` (not measured yet) are left out.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    table: Dict[Tuple[str, int], Preprocessing] = {}
    for board, modes in data["boards"].items():
        for key, entry in modes.items():
            if entry.get("pending"):
                continue
            table[(board, int(key.split("_")[1]))] = Preprocessing(
                frame_s=float(entry["frame_s"]),
                window_s={float(k): float(v) for k, v in entry["window_s"].items()},
            )
    return float(data["hop_s"]), table


@dataclass
class Estimate:
    """Energy result for one pair under one spec."""

    pair: Pair
    L: float
    S: float
    applicable: bool
    feasible: bool
    reason: str
    max_response_s: Optional[float] = None
    design: str = "inference_only"
    active_s: Optional[float] = None
    duty: float = 0.0
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


def estimate(
    pair: Pair,
    L: float,
    S: float,
    power: Dict[str, BoardPower],
    max_response_s: Optional[float] = None,
    design: str = "inference_only",
    preprocessing: Optional[Tuple[float, Dict[Tuple[str, int], Preprocessing]]] = None,
) -> Estimate:
    """Evaluate one pair under one spec with the low and high power values.

    ``max_response_s`` (``R``) is an optional limit on the response latency
    ``t``, separate from the stride. ``design`` selects how preprocessing is
    accounted (see the module docstring); ``preprocessing`` is the output of
    :func:`load_preprocessing` and is required for ``naive``/``streaming``.
    """
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}")
    result = Estimate(pair=pair, L=L, S=S, applicable=math.isclose(pair.window_s, L), feasible=False, reason="",
                      max_response_s=max_response_s, design=design)
    if not result.applicable:
        result.reason = f"window {pair.window_s:g} s != L"
        return result
    if pair.status != "ok" or pair.latency_s is None:
        result.reason = f"status {pair.status or 'missing'}"
        return result
    t = pair.latency_s
    duty = 0.0
    if design != "inference_only":
        hop_s, table = preprocessing if preprocessing is not None else (0.0, {})
        prep = table.get((pair.board, pair.threads))
        if prep is None:
            result.reason = "no preprocessing timing"
            return result
        if design == "naive":
            window = prep.window_s.get(float(L))
            if window is None:
                result.reason = f"no {L:g} s window preprocessing timing"
                return result
            t = window + t
        else:
            if prep.frame_s > hop_s:
                result.reason = "cannot stream (frame time > hop)"
                return result
            duty = prep.frame_s / hop_s
            t = prep.frame_s + t
    result.active_s = t
    result.duty = duty
    if t > S:
        result.reason = "too slow (t > S)"
        return result
    if S > L and t > S - L:
        result.reason = "too slow (t > S - L, no compute/listen overlap)"
        return result
    if max_response_s is not None and t > max_response_s:
        result.reason = "too slow (t > R)"
        return result
    board = power[pair.board]
    active = board.active_for(pair.threads)
    for i, bound in enumerate(BOUNDS):
        listen = board.listen[i] + duty * (active[i] - board.listen[i])
        joules = energy_per_inference(t, L, S, active[i], listen, board.sleep[i])
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
        "design", "L_s", "S_s", "max_response_s", "model", "board", "threads", "xnnpack", "latency_s",
        "active_s", "listen_duty", "feasible", "reason",
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
                "design": e.design, "L_s": e.L, "S_s": e.S,
                "max_response_s": "" if e.max_response_s is None else e.max_response_s,
                "active_s": "" if e.active_s is None else round(e.active_s, 6), "listen_duty": round(e.duty, 4),
                "model": e.pair.model, "board": e.pair.board,
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


def _fmt_response(max_response_s: Optional[float]) -> str:
    """Format a response limit for headings."""
    return "no limit" if max_response_s is None else f"{max_response_s * 1e3:g} ms"


def spec_report(
    estimates: Sequence[Estimate],
    L: float,
    S: float,
    min_accuracies: Sequence[float],
    max_response_s: Optional[float] = None,
    design: str = "inference_only",
) -> str:
    """Markdown for one spec: all applicable pairs, then the selections."""
    rows = [
        e for e in estimates
        if e.L == L and e.S == S and e.applicable and e.max_response_s == max_response_s and e.design == design
    ]
    lines = [
        f"### {design}: L = {L:g} s, S = {S:g} s, R = {_fmt_response(max_response_s)}",
        "",
        "| Pair | t active (ms) | Val acc | E low (mJ) | E high (mJ) | P avg low (mW) | P avg high (mW) | Feasible |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(rows, key=lambda e: (not e.feasible, e.energy_j.get("low", math.inf), pair_label(e.pair))):
        t_ms = f"{e.active_s * 1e3:,.2f}" if e.active_s is not None else "-"
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


def _winner_cell(ranked_low: Sequence[Estimate], ranked_high: Sequence[Estimate]) -> str:
    """One compact cell: the winner, noting when low and high power disagree."""
    if not ranked_low and not ranked_high:
        return "none feasible"
    low = pair_label(ranked_low[0].pair) if ranked_low else "none"
    high = pair_label(ranked_high[0].pair) if ranked_high else "none"
    return low if low == high else f"low: {low} / high: {high}"


def winners(
    estimates: Sequence[Estimate],
    specs: Sequence[Tuple[float, float]],
    responses: Sequence[Optional[float]],
    min_accuracies: Sequence[float],
    design: str,
) -> Dict[Tuple[float, float, Optional[float], float], str]:
    """Winner cell per (L, S, R, min accuracy) for one design."""
    cells = {}
    for L, S in specs:
        for response in responses:
            rows = [e for e in estimates if e.L == L and e.S == S and e.applicable
                    and e.max_response_s == response and e.design == design]
            for a in min_accuracies:
                cells[(L, S, response, a)] = _winner_cell(select(rows, a, "low"), select(rows, a, "high"))
    return cells


def changes_vs(
    base: Dict[Tuple[float, float, Optional[float], float], str],
    other: Dict[Tuple[float, float, Optional[float], float], str],
) -> List[str]:
    """Markdown rows for cells whose winner differs from ``base``."""
    rows = []
    for key, value in other.items():
        if base.get(key) != value:
            L, S, response, a = key
            rows.append(f"| {L:g} s | {S:g} s | {_fmt_response(response)} | {a:.2f} | {base.get(key)} | {value} |")
    return rows


def response_summary(
    estimates: Sequence[Estimate],
    specs: Sequence[Tuple[float, float]],
    responses: Sequence[Optional[float]],
    min_accuracies: Sequence[float],
    design: str = "inference_only",
) -> str:
    """Markdown table: winner per (spec, response limit) and accuracy floor."""
    lines = [
        "| L | S | R | " + " | ".join(f"min acc {a:.2f}" for a in min_accuracies) + " |",
        "|---|---|---|" + "---|" * len(min_accuracies),
    ]
    for L, S in specs:
        for response in responses:
            rows = [e for e in estimates if e.L == L and e.S == S and e.applicable
                    and e.max_response_s == response and e.design == design]
            cells = [_winner_cell(select(rows, a, "low"), select(rows, a, "high")) for a in min_accuracies]
            lines.append(f"| {L:g} s | {S:g} s | {_fmt_response(response)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parse_response(text: str) -> Optional[float]:
    """Parse a response limit in seconds, or ``none``."""
    return None if text.strip().lower() == "none" else float(text)


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
    parser.add_argument("--max-response", type=_parse_response, nargs="+", default=[None],
                        help="Response-latency limits in seconds, or 'none' (default: none).")
    parser.add_argument("--design", nargs="+", choices=DESIGNS, default=["inference_only"],
                        help="Preprocessing accounting (default: inference_only).")
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    """Estimate every spec, write the CSV and the markdown report."""
    args = parse_args()
    power = load_power(args.power)
    pairs = load_pairs(args.csv)
    preprocessing = load_preprocessing(args.preprocessing)
    estimates = [
        estimate(pair, L, S, power, response, design, preprocessing)
        for design in args.design
        for L, S in args.spec
        for response in args.max_response
        for pair in pairs
    ]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_estimates_csv(args.results_dir / "energy_estimates.csv", estimates)
    sections = []
    cells = {d: winners(estimates, args.spec, args.max_response, args.min_acc, d) for d in args.design}
    for design in args.design:
        sections.append(f"## Winners: {design}\n\n"
                        + response_summary(estimates, args.spec, args.max_response, args.min_acc, design))
    if "inference_only" in cells:
        for design in args.design:
            if design == "inference_only":
                continue
            rows = changes_vs(cells["inference_only"], cells[design])
            sections.append(
                f"## Changes: {design} vs inference_only\n\n"
                + ("| L | S | R | min acc | inference_only | " + design + " |\n|---|---|---|---|---|---|\n"
                   + "\n".join(rows) if rows else "No winner changes.")
            )
    details = "\n\n".join(
        spec_report(estimates, L, S, args.min_acc, response, design)
        for design in args.design
        for L, S in args.spec
        for response in args.max_response
    )
    header = (
        "Accuracy is int8 validation-fold accuracy (fold 9); the held-out test fold is not evaluated.\n"
        "Low/high use the low/high ends of every power range in board_power.yaml.\n"
    )
    summary = "\n\n".join(sections)
    (args.results_dir / "energy_selection.md").write_text(
        header + "\n" + summary + "\n\n## Details\n\n" + details + "\n", encoding="utf-8")
    print(header)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
