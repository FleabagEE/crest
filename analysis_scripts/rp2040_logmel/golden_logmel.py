# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Golden test for the integer log-mel front end (``sketches/common/crest_logmel.h``).

Clips (int16 PCM at 16 kHz) are built on the host:

- synthetic: silence, 1 kHz sine at -6 dBFS, 50 Hz-7.95 kHz chirp at
  -12 dBFS, white noise at -20 dBFS, and a quiet clip (440 Hz sine at
  -50 dBFS plus noise at -60 dBFS);
- optional real clips from ``--wav`` (read and resampled exactly like the
  training repo's ``read_waveform``, centre-cropped or padded to the clip
  length like ``crop_or_pad``).

Every clip is quantized to int16 first, and the reference sees exactly those
samples (``pcm / 32768``), so the comparison isolates the front end.

The reference is the training repo's own ``training.features.log_mel``
(imported from ``--training-repo``, not copied) with ``configs/baseline.yaml``.
The int8 reference is ``np.clip(np.rint(x / scale) + zero_point, -128, 127)``
as in ``quantization/validate_int8.py``.

Candidates:

- ``host-q15`` / ``host-q31``: ``crest_logmel.h`` compiled with g++ here;
- ``board``: a dump from ``sketches/logmel_golden`` (``--board-dump``).

Reported per clip and overall: max and mean absolute error of the float
log-mel, and the fraction of int8 values that differ (plus the largest int8
difference).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COMMON_DIR = REPO_ROOT / "sketches" / "common"
TFLM_SRC = REPO_ROOT / "tools" / "arduino-user" / "libraries" / "Arduino_TensorFlowLite" / "src"
DEFAULT_TRAINING_REPO = REPO_ROOT.parent / "embedded-systme-project"
DEFAULT_BUILD_DIR = SCRIPT_DIR / "build"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
SAMPLE_RATE = 16000
# Input quantization shared by all DSCNN_64_SC int8 exports (1s and 2s).
INPUT_SCALE = 0.07683266699314117
INPUT_ZERO_POINT = 52


def inv_scale_q24(scale: float = INPUT_SCALE) -> int:
    """``round(2^24 / scale)`` as used by ``crest_logmel::quantize``."""
    return int(round((1 << 24) / scale))


def to_int16(waveform: np.ndarray) -> np.ndarray:
    """Quantize a float waveform in [-1, 1) to int16 PCM (round, clip)."""
    return np.clip(np.round(np.asarray(waveform, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def synthetic_clips(seconds: float, seed: int = 0) -> Dict[str, np.ndarray]:
    """Deterministic synthetic test clips as int16 PCM."""
    n = int(round(SAMPLE_RATE * seconds))
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    db = lambda level: 10 ** (level / 20)
    chirp = np.sin(2 * np.pi * (50 * t + (7950 - 50) / (2 * seconds) * t ** 2))
    return {
        "silence": np.zeros(n, dtype=np.int16),
        "sine_1k_-6dB": to_int16(db(-6) * np.sin(2 * np.pi * 1000 * t)),
        "chirp_-12dB": to_int16(db(-12) * chirp),
        "noise_-20dB": to_int16(np.clip(db(-20) * rng.standard_normal(n), -1, 1)),
        "quiet_-50dB": to_int16(db(-50) * np.sin(2 * np.pi * 440 * t) + db(-60) * rng.standard_normal(n)),
    }


def wav_clip(path: Path, seconds: float, training_repo: Path) -> np.ndarray:
    """Load a real clip like the training repo, centre crop/pad, quantize to int16."""
    sys.path.insert(0, str(training_repo))
    from training.augment import crop_or_pad
    from training.dataset import read_waveform

    waveform = read_waveform(str(path), SAMPLE_RATE)
    return to_int16(crop_or_pad(waveform, int(round(SAMPLE_RATE * seconds))))


def reference(pcm: np.ndarray, training_repo: Path) -> np.ndarray:
    """Training-code log-mel ``(frames, 64)`` for int16 PCM (as ``pcm / 32768``)."""
    sys.path.insert(0, str(training_repo))
    from training.features import log_mel

    audio = yaml.safe_load((training_repo / "configs" / "baseline.yaml").read_text())["audio"]
    waveform = (pcm.astype(np.float32) / np.float32(32768.0)).astype(np.float32)
    return log_mel(waveform, audio)[..., 0]


def reference_int8(log_mel: np.ndarray) -> np.ndarray:
    """int8 input exactly as ``quantization/validate_int8.py`` computes it."""
    value = log_mel.astype(np.float32)
    return np.clip(np.rint(value / INPUT_SCALE) + INPUT_ZERO_POINT, -128, 127).astype(np.int8)


def build_host(bits: int, build_dir: Path) -> Path:
    """Compile the host harness for a FFT width; return the binary path."""
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / f"host_logmel_q{bits - 1}"
    sources = [str(SCRIPT_DIR / "host_logmel.cpp")]
    if bits == 16:
        sources.append(str(TFLM_SRC / "tensorflow" / "lite" / "experimental" / "microfrontend" / "lib" / "kiss_fft_int16.cpp"))
    cmd = ["g++", "-O2", "-std=c++17", f"-DCREST_LOGMEL_FFT_BITS={bits}", f"-I{COMMON_DIR}", f"-I{TFLM_SRC}",
           *sources, "-o", str(binary)]
    subprocess.run(cmd, check=True)
    return binary


def run_host(binary: Path, pcm: np.ndarray) -> Dict[str, np.ndarray]:
    """Run the host harness on one clip."""
    with tempfile.TemporaryDirectory() as tmp:
        pcm_path, out_path = Path(tmp) / "in.pcm", Path(tmp) / "out.bin"
        pcm.astype("<i2").tofile(pcm_path)
        subprocess.run([str(binary), str(pcm_path), str(out_path), str(inv_scale_q24()), str(INPUT_ZERO_POINT)],
                       check=True)
        raw = out_path.read_bytes()
    frames = int(np.frombuffer(raw[:4], "<i4")[0])
    count = frames * 64
    ln = np.frombuffer(raw[4:4 + 4 * count], "<i4").reshape(frames, 64)
    q = np.frombuffer(raw[4 + 4 * count:4 + 5 * count], np.int8).reshape(frames, 64)
    return {"ln_q16": ln.copy(), "int8": q.copy()}


def parse_board_dump(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse ``LM``/``Q8`` lines from the golden sketch's serial output."""
    ln: Dict[str, Dict[int, List[int]]] = {}
    q8: Dict[str, Dict[int, List[int]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] == "LM":
            words = bytes.fromhex(parts[3])
            ln.setdefault(parts[1], {})[int(parts[2])] = np.frombuffer(words, "<i4").tolist()
        elif len(parts) == 4 and parts[0] == "Q8":
            q8.setdefault(parts[1], {})[int(parts[2])] = np.frombuffer(bytes.fromhex(parts[3]), np.int8).tolist()
    out = {}
    for clip, frames in ln.items():
        order = sorted(frames)
        out[clip] = {
            "ln_q16": np.array([frames[f] for f in order], dtype=np.int32),
            "int8": np.array([q8[clip][f] for f in order], dtype=np.int8),
        }
    return out


def compare(ref: np.ndarray, ref_q: np.ndarray, cand: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Error statistics of a candidate against the reference."""
    if cand["ln_q16"].shape != ref.shape:
        return {"error": f"shape {cand['ln_q16'].shape} != {ref.shape}"}
    ln = cand["ln_q16"].astype(np.float64) / 65536.0
    abs_err = np.abs(ln - ref.astype(np.float64))
    q_diff = np.abs(cand["int8"].astype(np.int32) - ref_q.astype(np.int32))
    return {
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "int8_diff_fraction": float((q_diff > 0).mean()),
        "int8_max_diff": int(q_diff.max()),
        "values": int(ref.size),
    }


def write_clip_header(path: Path, clips: Dict[str, np.ndarray]) -> None:
    """Write the clips as a C header for the golden sketch."""
    lines = [
        "// GENERATED by analysis_scripts/rp2040_logmel/golden_logmel.py; do not edit.",
        "#pragma once",
        "#include <stdint.h>",
        f"#define GOLDEN_NUM_CLIPS {len(clips)}",
    ]
    names = list(clips)
    for i, name in enumerate(names):
        values = ",".join(str(int(v)) for v in clips[name])
        lines.append(f"static const int16_t kGoldenClip{i}[{len(clips[name])}] = {{{values}}};")
    lines.append("static const int16_t* const kGoldenClips[] = {" + ",".join(f"kGoldenClip{i}" for i in range(len(names))) + "};")
    lines.append("static const int32_t kGoldenClipLength[] = {" + ",".join(str(len(clips[n])) for n in names) + "};")
    lines.append("static const char* const kGoldenClipName[] = {" + ",".join(f'"{n}"' for n in names) + "};")
    lines.append(f"#define GOLDEN_INV_SCALE_Q24 {inv_scale_q24()}")
    lines.append(f"#define GOLDEN_ZERO_POINT {INPUT_ZERO_POINT}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(per_clip: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Combine per-clip statistics (weighted by value count)."""
    values = sum(s["values"] for s in per_clip.values())
    return {
        "max_abs_err": max(s["max_abs_err"] for s in per_clip.values()),
        "mean_abs_err": sum(s["mean_abs_err"] * s["values"] for s in per_clip.values()) / values,
        "int8_diff_fraction": sum(s["int8_diff_fraction"] * s["values"] for s in per_clip.values()) / values,
        "int8_max_diff": max(s["int8_max_diff"] for s in per_clip.values()),
        "values": values,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-repo", type=Path, default=DEFAULT_TRAINING_REPO)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--wav", type=Path, nargs="*", default=[], help="Real clips (e.g. UrbanSound8K .wav).")
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 32], choices=[16, 32])
    parser.add_argument("--board-dump", type=Path, default=None, help="Serial output of sketches/logmel_golden.")
    parser.add_argument("--write-header", type=Path, default=None, help="Write the clips as a C header and exit.")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Run the golden comparison."""
    args = parse_args()
    clips = synthetic_clips(args.seconds)
    for wav in args.wav:
        clips[f"wav_{wav.stem}"] = wav_clip(wav, args.seconds, args.training_repo)
    if args.write_header:
        write_clip_header(args.write_header, clips)
        print(f"wrote {args.write_header} ({len(clips)} clips)")
        return 0

    refs = {name: reference(pcm, args.training_repo) for name, pcm in clips.items()}
    candidates: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for bits in args.bits:
        binary = build_host(bits, args.build_dir)
        candidates[f"host-q{bits - 1}"] = {name: run_host(binary, pcm) for name, pcm in clips.items()}
    if args.board_dump is not None:
        candidates["board"] = parse_board_dump(args.board_dump)

    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cand_name, outputs in candidates.items():
        per_clip = {}
        for clip, ref in refs.items():
            if clip not in outputs:
                continue
            per_clip[clip] = compare(ref, reference_int8(ref), outputs[clip])
        report[cand_name] = {"per_clip": per_clip, "overall": summarize(per_clip)}
        if "board" in candidates and cand_name == "board":
            for host_name in [c for c in candidates if c.startswith("host-")]:
                same = all(np.array_equal(outputs[c]["ln_q16"], candidates[host_name][c]["ln_q16"])
                           and np.array_equal(outputs[c]["int8"], candidates[host_name][c]["int8"])
                           for c in outputs if c in candidates[host_name])
                report[cand_name][f"bit_exact_vs_{host_name}"] = same

    for cand_name, data in report.items():
        print(f"== {cand_name}")
        for clip, s in data["per_clip"].items():
            print(f"  {clip:18s} max_abs_err={s['max_abs_err']:.5f} mean_abs_err={s['mean_abs_err']:.6f} "
                  f"int8_diff={s['int8_diff_fraction'] * 100:.3f}% int8_max_diff={s['int8_max_diff']}")
        o = data["overall"]
        print(f"  OVERALL            max_abs_err={o['max_abs_err']:.5f} mean_abs_err={o['mean_abs_err']:.6f} "
              f"int8_diff={o['int8_diff_fraction'] * 100:.3f}% int8_max_diff={o['int8_max_diff']} (n={o['values']})")
        for key, value in data.items():
            if key.startswith("bit_exact"):
                print(f"  {key}: {value}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
