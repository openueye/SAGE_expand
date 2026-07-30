# SAGE

SAGE (Structure-Anchored Gaussian Enhancement) is the complete multi-sensor
Gaussian-mapping method used in this thesis. From either a calibrated finite
Odin recording or a compatible Prepared Scene, one command builds the Gaussian
structure, refines its appearance while preserving geometry and topology,
publishes the final checkpoint, and evaluates every accepted input frame.

Sparse LiDAR remains the metric authority. SPNet depth is locally aligned to
LiDAR and contributes guarded dense-normal supervision instead of dense metric
depth targets. Per-frame exposure variables are bounded training nuisances and
are not applied to official evaluation images.

## Installation

```bash
git clone https://github.com/openueye/SAGE.git
cd SAGE
conda-lock install --name sage conda-lock.yml
conda run -n sage python tools/install_locked_pip.py --environment sage
conda run -n sage python tools/build_renderer.py
conda activate sage
```

The included renderer build targets CUDA compute capability 8.9. Set
`RENDERER_CUDA_ARCH_LIST` and rebuild for a different architecture. The
renderer manifest records and validates the selected build identity.

## Models

SPNet source and weights are not redistributed. Install the pinned source and
import locally obtained model files:

```bash
git clone https://github.com/Wang-xjtu/SPNet.git third_party/SPNet
git -C third_party/SPNet checkout b836bd044517b33d3737094acd6a1f09c2362f04
export SAGE_MODEL_ROOT=/path/to/sage-models
python tools/download_models.py spnet-large-300 --source /path/to/Large_300.pth
python tools/download_models.py alexnet-imagenet --source /path/to/alexnet-owt-7be5be79.pth
```

Set `SAGE_SPNET_ROOT` only when the pinned SPNet checkout is stored elsewhere.

## Inputs

SAGE supports two equivalent complete input paths:

- a finite Odin ROSBAG plus its `cam_in_ex.txt` camera/LiDAR calibration;
- a compatible Prepared Scene, including public datasets converted to this
  contract.

Both paths run structure optimization, appearance refinement, and final
evaluation. A ROSBAG can be converted once for reuse:

```bash
python tools/prepare_dataset.py \
  --rosbag /path/to/odin1-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output /path/to/prepared
```

The producer processes the bag to EOF and publishes only a complete scene with
source, transform, frame-order, and content identities. Passing
`--write-through PATH` to ROSBAG training records the same representation while
the end-to-end run proceeds.

## Configuration

`configs/sage.yaml` is the only method configuration. Its `mapping`,
`refinement`, and `evaluation` sections describe successive parts of one
workflow; refinement is not a separate mode.

Unknown, missing, or incompatible fields are rejected. The method has one
deterministic seeded schedule and no alternative experimental branches.

## Training

Run the complete method from one command:

```bash
sage train \
  --config configs/sage.yaml \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/sage \
  --device cuda
```

For a Prepared Scene, replace the two ROSBAG arguments with:

```bash
sage train \
  --config configs/sage.yaml \
  --prepared-scene /path/to/prepared-scene \
  --output outputs/sage \
  --device cuda
```

Add `--preflight` to validate the complete configuration, selected input,
CUDA runtime, models, renderer, and output path without training.

The output is atomic at each recoverable boundary:

```text
outputs/sage/
├── structure/
│   ├── checkpoint.pt
│   ├── map.ply
│   └── run_manifest.json
├── final/
│   ├── appearance_checkpoint.pt
│   ├── appearance_map.ply
│   ├── comparison.json
│   └── run_manifest.json
├── evaluation/
│   ├── evaluation.json
│   └── run_manifest.json
└── run_manifest.json
```

`final/appearance_checkpoint.pt` is the final SAGE checkpoint. The structure
checkpoint is retained only so an interrupted run can resume without repeating
completed structure optimization. Existing output paths are never silently
overwritten.

## Evaluation

Training performs final evaluation automatically. To evaluate an existing
final checkpoint against any compatible input, use the same input choice:

```bash
sage evaluate \
  --config configs/sage.yaml \
  --checkpoint outputs/sage/final/appearance_checkpoint.pt \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/sage_evaluation \
  --device cuda
```

Use `--prepared-scene /path/to/prepared-scene` instead for the second input
path. Evaluation discovers and processes all accepted frames emitted by the
input; it contains no dataset-specific frame list or assumed frame count.

`python -m sage` exposes the same `train` and `evaluate` commands.

## Reproducibility contract

Published checkpoints record input and configuration hashes, selected mapping
frames, aligned dense-prior provenance, bounded exposure nuisance state,
dependency identities, and producer-code identity. Publication verifies that
positions, scales, rotations, stable IDs, creation metadata, and source
provenance remain bitwise unchanged during appearance optimization.

Official image and geometry metrics use the native final map without learned
exposure correction. Exposure-corrected mapping-frame metrics are reported
separately as diagnostics.

This release targets finite, calibrated scenes and produces one scene-specific
checkpoint per run. It is not validated for live ROS2 control, real-time
deployment, resource-constrained hardware, or safety-critical navigation.
