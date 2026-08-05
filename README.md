# SAGE Expand

SAGE Expand is the thesis engineering variant of SAGE (Structure-Anchored
Gaussian Enhancement). The full training command reads a finite, calibrated
ROS 2 bag, compiles it into canonical frames, builds a structure map, refines
appearance, and evaluates the final map over every accepted frame.

The normal thesis path is the `rosbag2` adapter. A Prepared Scene is an
optional persisted replay of the same canonical input; it is not required for
training.

## Runtime and installation

Use the dedicated `sage` conda environment for preflight, mapping,
refinement, and evaluation — never the unrelated `3dgs_train` environment.

```bash
cd 00_Baselines/SAGE
conda-lock install --name sage conda-lock.yml
conda run -n sage python tools/install_locked_pip.py --environment sage
```

The first renderer invocation compiles the pinned `gsplat` CUDA extension, so
it needs the locked CUDA toolchain and is slower than later runs. Set
`TORCH_CUDA_ARCH_LIST` before preflight for a fixed architecture; the run
receipt records the value used.

## Models

SAGE needs the local SPNet source and user-provided model weights.

```bash
cd 00_Baselines/SAGE
git clone https://github.com/Wang-xjtu/SPNet.git third_party/SPNet
git -C third_party/SPNet checkout b836bd044517b33d3737094acd6a1f09c2362f04

export SAGE_MODEL_ROOT=/path/to/sage-models
python tools/download_models.py spnet-large-300 --source /path/to/Large_300.pth
python tools/download_models.py alexnet-imagenet --source /path/to/alexnet-owt-7be5be79.pth
```

This workspace's model directory:

```text
00_Baselines/SAGE-models/
├── Large_300.pth
└── alexnet-owt-7be5be79.pth
```

`runtime.model_root` is authoritative when set in the config, and resolves
relative to the config file's own directory (not the shell's cwd) — configs
stored outside `00_Baselines/SAGE/configs` should use an absolute path.

SPNet can be turned off for the mapping stage with `spnet.enabled: false` (no
`model_id` needed): no weights are loaded and no source tree is verified;
growth and pruning simply never see `SPNET_COMPLETED` evidence. Appearance
refinement has no SPNet-free path — its dense geometry priors are built from
SPNet predictions — so a `spnet.enabled: false` config fails fast at
refinement preflight with a clear error rather than training end-to-end.

## Dataset and calibration layout

Calibration belongs to the capture rig, not to an individual Scene. Recordings
from the same rig share one canonical calibration at the dataset-group root:

```text
03_Datasets/001_Odin/
├── calibration.yaml       # one canonical runtime calibration for this rig
├── cam_in_ex.txt           # source calibration, kept for provenance
├── Ferrari1/
│   ├── metadata.yaml
│   └── *.db3
├── Graffiti1/
└── ...
```

Don't duplicate a runtime calibration per Scene when the rig calibration is
identical; a different device/camera/lens/output-grid gets its own
dataset-group calibration. Convert Odin's `cam_in_ex.txt` once per rig:

```bash
cd 00_Baselines/SAGE
conda run --no-capture-output -n sage python tools/convert_calibration.py \
  --source .../001_Odin/cam_in_ex.txt \
  --output .../001_Odin/calibration.yaml \
  --reference-frame odom --pose-frame odin1_base_link \
  --lidar-frame odin1_base_link --camera-frame camera \
  --output-size 800 648 \
  --pose-from-lidar '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'
```

Odin's `/odin1/cloud_slam` is already registered into `odom`, so its configs
declare `input.lidar.points_frame: reference_frame` to avoid applying the
pose chain twice. A raw scan in the sensor's own frame must use `lidar_frame`
(SAGE's default) with the matching calibration contract instead. Verify
shared-calibration assumptions before reusing bytes across Scenes:
`sha256sum 03_Datasets/001_Odin/*/cam_in_ex.txt`.

## Configuration

`configs/sage.yaml` is the single default reference config: every field is
explicit, and it is the one config edited in place as the project's baseline
evolves. Its `input:` section is a concrete Ferrari1 example, not a universal
default — don't treat it as one.

`configs/odin_sage.yaml` is the terser, Odin-rig-scoped counterpart: it
resolves to the exact same values, but any field that already equals the
code's own default (in `GrowthConfig`, `PruningConfig`, `MappingLossConfig`,
`AppearanceRefinementConfig`, `SynchronizationSpec`/`FusionSpec`/`DepthSpec`,
...) is simply omitted rather than repeated. Only genuinely dataset-specific
values (topics, `points_frame`, calibrated thresholds that differ from
default) stay explicit. Use it as the template for a new Scene: copy it and
change `input.rosbag`.

For a per-Scene run config derived from either file, change only the input
identity (and any machine-local runtime paths). **Never edit a config after
preflight or after training starts** — it's part of the run identity, and its
SHA-256 is recorded in the artifacts.

```yaml
# 04_Outputs/SAGE/Odin/Graffiti1_sage.yaml
input:
  type: rosbag2
  rosbag: .../001_Odin/Graffiti1
  calibration: .../001_Odin/calibration.yaml
  topics: {lidar: /odin1/cloud_slam, image: /odin1/image/compressed, odometry: /odin1/odometry}
  lidar: {points_frame: reference_frame}
```

The YAML is strict about what it accepts: unknown fields are always rejected,
required fields must be present, and *optional* fields (documented per
section in the source dataclasses) fall back to the code default when
omitted — never to a silently different value. The CLI carries run control
only; it never overrides the bag, calibration, topics, or synchronization
policy.

### Online mapping and evaluation policy

New runs may opt into one-pass ROSBAG input:

```yaml
input:
  type: rosbag2
  execution: online-window-v2   # frozen baseline stays batch-v1; don't retro-change a reproduction config
```

`online-window-v2` indexes header/row metadata first, sorts by effective
header time, then keeps image/cloud payload reads inside one canonical-frame
pass — so physical write order in the bag doesn't need to match sensor header
order. For bags with an empty image `Header.frame_id`:

```yaml
input:
  execution: online-window-v2
  allow_empty_image_frame_id: true   # only an empty frame ID is accepted; a wrong one still errors
```

Stage 1 writes a transient `.stage2-input-cache` beside `structure/` (mapping
frames only); Stage 2 reads it directly and never replays the bag. It
survives a failed Stage 2/3 attempt for retry and is removed only once the
full pipeline publishes successfully.

To get separate reports for both checkpoints in one Stage 3 pass:

```yaml
evaluation:
  checkpoint_stages: [stage1_mapping, stage2_refinement]  # default: [stage2_refinement]
```

This writes `evaluation/stage1_mapping/` and `evaluation/stage2_refinement/`
separately; the refinement checkpoint remains the default final-quality report.

## ROSBAG adapter contract

The `rosbag2` adapter owns bag decoding, timestamp association, pose
interpolation, image rectification, LiDAR projection, and centered-window
fusion. Baseline policy:

- image header stamps as canonical frame anchors;
- nearest LiDAR association within 20 ms;
- linear-slerp pose interpolation, 100 ms max pose gap, no extrapolation;
- centered five-scan fusion window, nearest-depth z-buffering, conflict rejection;
- depth limits 0.1 m–200 m; evaluation over every accepted canonical frame.

`batch-v1` runs a full preflight before mapping: bag schema, declared topics,
message types, frame IDs, synchronization plan, calibration, projection
coverage, CUDA, renderer, SPNet, and metric dependencies.

## Preflight and end-to-end training

Run from the SAGE repository root — required by the fresh-process execution
boundary and its receipt.

```bash
export THESIS_ROOT=/home/haibo/Documents/Thesis
export SAGE_ROOT="${THESIS_ROOT}/00_Baselines/SAGE"
export SCENE_NAME=Graffiti1
export DEVICE=cuda
export CONFIG="${THESIS_ROOT}/04_Outputs/SAGE/Odin/${SCENE_NAME}_sage.yaml"
export OUTPUT="${THESIS_ROOT}/04_Outputs/SAGE/Odin/${SCENE_NAME}"
cd "${SAGE_ROOT}"
```

Preflight against an unused output identity (does not start training):

```bash
conda run --no-capture-output -n sage python -m sage train \
  --config "${CONFIG}" --output "${OUTPUT}" --device "${DEVICE}" --preflight
```

Full run, once preflight succeeds — mapping, then refinement, then final
evaluation, in one command:

```bash
conda run --no-capture-output -n sage python -m sage train \
  --config "${CONFIG}" --output "${OUTPUT}" --device "${DEVICE}"
```

To republish evaluation separately later:

```bash
conda run --no-capture-output -n sage python -m sage evaluate \
  --checkpoint "${OUTPUT}/final/appearance_checkpoint.pt" \
  --config "${CONFIG}" --output "${OUTPUT}/evaluation_republished" --device "${DEVICE}"
```

## Output layout

```text
04_Outputs/SAGE/Odin/Graffiti1/
├── structure/{checkpoint.pt, map.ply, spnet_dense.pt, run_manifest.json}
├── structure.execution.json
├── final/{appearance_checkpoint.pt, appearance_map.ply, run_manifest.json}
├── evaluation/{evaluation.json, run_manifest.json}
└── run_manifest.json
```

`structure/spnet_dense.pt` is an identity- and manifest-bound cache of dense
SPNet predictions from mapping, reused by refinement (absent when SPNet is
disabled). Manifests record the fresh child process, exit code, wall time,
peak RSS, formal config identity, artifact hashes, canonical sequence and
input contract identities, adapter/source provenance, training identity,
model identities, renderer identity, and worktree state.

Existing complete outputs are never silently overwritten — use a new
Scene/run directory per experiment so recorded identities stay auditable.

## Prepared Scene (optional)

Training doesn't export one as a side effect; export explicitly if a stable,
replayable input artifact is needed:

```bash
conda run --no-capture-output -n sage python -m sage prepare \
  --config "${CONFIG}" --output /path/to/prepared-scene
```

A separate config can then use `input.type: prepared_scene`. The ROSBAG run
and its Prepared Scene share the canonical sequence identity; adapter
provenance stays distinguishable.

## Common failure points

- `calibration.yaml` missing: create it once with `tools/convert_calibration.py`
  and point every Scene config at it.
- Frame-ID mismatch: check the bag's odometry/point-cloud header frame IDs,
  fix the calibration contract or `points_frame` — don't disable the check.
- No projected LiDAR coverage: verify `Tcl_0`/`camera_from_lidar`, camera
  output grid, point-frame declaration, and calibration units.
- Missing models or renderer errors: run preflight in `sage` and check
  `runtime.model_root`, `third_party/SPNet`, and the locked CUDA toolchain.
- Existing output refusal: pick a new Scene/run output identity — don't
  delete or overwrite an auditable run.

SAGE targets finite, calibrated research scenes. It is not a real-time ROS 2
or safety-critical navigation system.
