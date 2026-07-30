# SAGE Expand

`SAGE_expand` is the thesis engineering variant of SAGE (Structure-Anchored
Gaussian Enhancement). One command turns a calibrated finite Odin ROSBAG, or a
compatible Prepared Scene, into a structure map, an appearance-refined final
map, and an evaluation over every accepted frame. LiDAR remains the metric
authority; SPNet supplies guarded dense-normal supervision only.

## Install

```bash
git clone https://github.com/openueye/SAGE.git
cd SAGE
conda-lock install --name sage conda-lock.yml
conda run -n sage python tools/install_locked_pip.py --environment sage
conda run -n sage python tools/build_renderer.py
```

The bundled renderer targets CUDA capability 8.9. Set
`RENDERER_CUDA_ARCH_LIST` and rebuild it for another GPU architecture.

## Models

Install the pinned SPNet source and locally obtained weights:

```bash
git clone https://github.com/Wang-xjtu/SPNet.git third_party/SPNet
git -C third_party/SPNet checkout b836bd044517b33d3737094acd6a1f09c2362f04
export SAGE_MODEL_ROOT=/path/to/sage-models  # required only while importing
python tools/download_models.py spnet-large-300 --source /path/to/Large_300.pth
python tools/download_models.py alexnet-imagenet --source /path/to/alexnet-owt-7be5be79.pth
```

Training reads its model directory from the YAML configuration, for example:

```yaml
runtime:
  model_root: ../../SAGE-models
  require_clean_worktree: false
```

The path is relative to the YAML file and overrides `SAGE_MODEL_ROOT`.
`third_party/SPNet` is used by default; set `SAGE_SPNET_ROOT` only for a
different verified checkout.

## Run

```bash
sage train \
  --config configs/sage.yaml \
  --rosbag /path/to/odin-bag \
  --calibration /path/to/cam_in_ex.txt \
  --output outputs/sage \
  --device cuda
```

Use `--prepared-scene /path/to/prepared-scene` instead of the ROSBAG arguments
when replaying a prepared input. Add `--preflight` to validate the input, CUDA,
renderer, models, and output path without training.

Training runs mapping, appearance refinement, and final evaluation in order:

```text
outputs/sage/
├── structure/{checkpoint.pt, map.ply, run_manifest.json}
├── structure.execution.json
├── final/{appearance_checkpoint.pt, appearance_map.ply, run_manifest.json}
├── evaluation/{evaluation.json, run_manifest.json}
└── run_manifest.json
```

Use `sage evaluate --checkpoint outputs/sage/final/appearance_checkpoint.pt`
with the same input choice to publish a separate evaluation output. Existing
outputs are never silently overwritten.

## Experiment identity

YAML fields are strict. The configuration SHA-256, input identities, model
hashes, renderer identity, source-state hash, and `worktree_dirty` flag are
recorded with each run. A dirty checkout is allowed by default; set
`runtime.require_clean_worktree: true` to reject it during preflight and
mapping. Changing the configuration while a run is active is always rejected.

This project targets finite, calibrated scenes. It is not a real-time ROS2 or
safety-critical navigation system.
