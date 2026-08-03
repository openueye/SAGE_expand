# SAGE Expand

`SAGE_expand` is the thesis engineering variant of SAGE (Structure-Anchored
Gaussian Enhancement). One command turns a calibrated finite ROS 2 bag, or a
Prepared Scene, into a structure map, an appearance-refined final map, and an
evaluation over every accepted frame. Both inputs are equal peers: a bag trains
end to end without being converted first, and a Prepared Scene needs no ROS. LiDAR remains the metric
authority; SPNet supplies guarded dense-normal supervision only, and both
SPNet/AlexNet artifacts are required runtime dependencies for mapping and
evaluation.

## Install

```bash
git clone https://github.com/openueye/SAGE.git
cd SAGE
conda-lock install --name sage conda-lock.yml
conda run -n sage python tools/install_locked_pip.py --environment sage
```

SAGE uses the pinned `gsplat` package. Its CUDA extension is compiled and cached
on the first renderer invocation, so the first preflight or training run takes
longer and requires the locked CUDA compiler toolchain. By default PyTorch
targets the CUDA architectures of the GPUs visible during that first invocation;
set `TORCH_CUDA_ARCH_LIST` before it if a fixed target architecture is required.
The run receipt records both the active GPU compute capability and the raw
`TORCH_CUDA_ARCH_LIST` value (`null` when it is unset).

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
`third_party/SPNet` is the required local checkout by default; set
`SAGE_SPNET_ROOT` only for a different verified clone.

## Run

```bash
sage train --config configs/sage.yaml --output outputs/sage --device cuda
```

The input is defined entirely by the configuration's `input:` section, so the
command line carries run control only. Add `--preflight` to validate the input,
CUDA, renderer, models, and output path without training.

A ROS 2 bag needs its three topics named explicitly and a canonical
`calibration.yaml` (convert a `cam_in_ex.txt` with
`tools/convert_calibration.py`):

```yaml
input:
  type: rosbag2
  rosbag: /path/to/bag
  calibration: /path/to/calibration.yaml
  topics:
    lidar: /points
    image: /camera/image/compressed
    odometry: /odom
  lidar:
    # lidar_frame for raw scans; reference_frame when the producer already
    # registered each scan into the odometry frame.
    points_frame: lidar_frame
    enable_fused: true
  synchronization: {...}
  fusion: {...}
  depth: {...}
```

A Prepared Scene only needs its directory:

```yaml
input:
  type: prepared_scene
  scene: /path/to/prepared-scene
```

`sage prepare --config <config> --output <scene>` exports the configured input
as a Prepared Scene. It is the only path that persists an input; training never
writes one as a side effect. An exported scene replays the bag's canonical
frames element for element and shares its canonical sequence identity, so a
checkpoint trained on one can be evaluated against the other.

Training runs mapping, appearance refinement, and final evaluation in order:

```text
outputs/sage/
├── structure/{checkpoint.pt, map.ply, spnet_dense.pt, run_manifest.json}
├── structure.execution.json
├── final/{appearance_checkpoint.pt, appearance_map.ply, run_manifest.json}
├── evaluation/{evaluation.json, run_manifest.json}
└── run_manifest.json
```

`structure/spnet_dense.pt` is an identity- and manifest-hash-bound cache of
dense SPNet predictions already computed during mapping. Appearance refinement
reuses those predictions and runs SPNet online only for mapping frames absent
from the cache (normally the bootstrap frame). Older valid structure outputs
without this optional cache remain resumable and fall back to online inference.

Use `sage evaluate --checkpoint outputs/sage/final/appearance_checkpoint.pt`
with a config naming an equivalent input to publish a separate evaluation
output. Existing outputs are never silently overwritten.

## Experiment identity

YAML fields are strict. Each run records three input identities plus a training
identity:

- `canonical_sequence_identity` — the ordered frames SAGE actually consumed
  (timestamps, image, K, pose, every observation array).
- `canonical_contract_identity` — the frame semantics they were consumed under.
- `adapter_provenance_identity` and `source_identity` — how and from what bytes
  they were produced; recorded for audit, never a compatibility gate.
- `training_config_identity` — the configuration minus its input section.

Checkpoint reuse is bound to the canonical and training identities, which is
what lets a ROSBAG run and its exported Prepared Scene exchange checkpoints
while a different frame selection is rejected. The resolved contract and the
preflight report are written next to each run's artifacts. The configuration
SHA-256, model hashes, renderer identity, source-state hash, and
`worktree_dirty` flag are recorded as well. A dirty checkout is allowed by default; set
`runtime.require_clean_worktree: true` to reject it during preflight and
mapping. Changing the configuration while a run is active is always rejected.

This project targets finite, calibrated scenes. It is not a real-time ROS2 or
safety-critical navigation system.
