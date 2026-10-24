# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Export the smallest UrbanSound8K DS-CNN as a staged RP2040 candidate.

The candidate directory gets ``g_model.tflite``, ``model.cc``/``model.h`` and
the staged ``crest_inference_latency.ino`` sketch, i.e. the same layout that
``ArduinoNanoRP2040Device.prepare_candidate`` produces inside a NAS run.

The model is untrained. Latency and memory depend on the graph and tensor
shapes, not on the weights, so random calibration data is enough unless
``--calibration-npy`` points at real log-mel windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_MODEL_ID = "dscnn_us8k_min_int8"

# Smallest choice for every audio_dscnn hyperparameter: 2 blocks, 3 pointwise
# filters (round(4 * 0.75)), stride 2 in both blocks, no dense head.
SMALLEST_DSCNN_HPARAMS = {
    "base_channels": 4,
    "num_blocks": 2,
    "kernel_time": 3,
    "kernel_freq": 3,
    "stride_schedule": "aggressive",
    "channel_growth": 1.0,
    "max_channels": 64,
    "depth_multiplier": 1,
    "pointwise_scale": 0.75,
    "dropout_rate": 0.0,
    "norm_flag": True,
    "dense_units": 0,
    "global_pool_type": "avg",
    "activation": "relu",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "rp2040_bench" / DEFAULT_MODEL_ID,
        help="Candidate directory to create (default: models/rp2040_bench/<model-id>).",
    )
    parser.add_argument(
        "--quantization",
        choices=("int8_ptq", "float"),
        default="int8_ptq",
        help="Export mode (default: int8_ptq).",
    )
    parser.add_argument(
        "--calibration-npy",
        type=Path,
        default=None,
        help="Optional .npy of shape (N, 201, 64) log-mel windows for int8 calibration.",
    )
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """Build, quantize and stage the smallest DS-CNN candidate.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args()

    import numpy as np
    import tensorflow as tf

    from crest.datasets.urbansound8k_common import EXPECTED_FRAMES, MEL_BINS
    from crest.devices import _sync_arduino_sketch_variant_for_config
    from crest.hardware import convert_to_cpp_model, convert_to_tflite_model
    from crest.microcontrollers.arduino_nano_rp2040 import BOARD_NAME
    from crest.model_families.audio_dscnn import AudioDSCNNFamily
    from crest.pipeline_types import ModelBuildContext, TargetSpec

    tf.keras.utils.set_random_seed(args.seed)
    ctx = ModelBuildContext(
        input_shape=(EXPECTED_FRAMES, MEL_BINS),
        input_dtype="float32",
        target_spec=TargetSpec(
            task_type="classification",
            output_names=["class_logits"],
            output_shapes=[(10,)],
            metadata={"num_classes": 10, "label_encoding": "class_index", "from_logits": True},
        ),
    )
    model_config = {"params": {"export_variant": "untrained"}, "search": {}}
    model = AudioDSCNNFamily().build_model(dict(SMALLEST_DSCNN_HPARAMS), ctx, model_config)

    calibration = None
    if args.quantization == "int8_ptq":
        if args.calibration_npy is not None:
            calibration = np.load(args.calibration_npy).astype(np.float32)
        else:
            rng = np.random.default_rng(args.seed)
            calibration = rng.normal(
                size=(args.calibration_samples, EXPECTED_FRAMES, MEL_BINS)
            ).astype(np.float32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = output_dir / "g_model.tflite"
    convert_to_tflite_model(
        model=model,
        training_data=calibration,
        quantization_mode=args.quantization,
        output_name=str(tflite_path),
    )
    convert_to_cpp_model(tflite_path=tflite_path, output_dir=output_dir)
    sketch = _sync_arduino_sketch_variant_for_config(
        {
            "device": {"name": BOARD_NAME},
            "training": {"energy_aware": False, "input_mode": "uniform"},
        },
        output_dir,
    )
    metadata = {
        "model_id": output_dir.name,
        "family": "audio_dscnn",
        "hparams": SMALLEST_DSCNN_HPARAMS,
        "quantization": args.quantization,
        "input_shape": [EXPECTED_FRAMES, MEL_BINS],
        "tflite_bytes": tflite_path.stat().st_size,
        "params": int(model.count_params()),
        "calibration": str(args.calibration_npy) if args.calibration_npy else "random_normal",
    }
    (output_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"candidate dir : {output_dir}")
    print(f"sketch        : {sketch}")
    print(f"tflite bytes  : {metadata['tflite_bytes']}")
    print(f"params        : {metadata['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
