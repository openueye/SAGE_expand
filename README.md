# SAGE Expand

SAGE Expand is the thesis engineering variant of SAGE (Structure-Anchored
Gaussian Enhancement). The complete training command reads a finite,
calibrated ROS 2 bag, compiles it into canonical frames, builds a structure
map, refines appearance, and evaluates the final map over every accepted
frame.

The normal thesis path is the `rosbag2` adapter. A Prepared Scene is an
optional persisted replay of the same canonical input; it is not required for
training.

## Runtime and installation

Use the dedicated `sage` environment for preflight, mapping, refinement, and
evaluation. Do not use the unrelated `3dgs_train` environment for SAGE.

From a clean checkout:

```bash
cd 00_Baselines/SAGE
conda-lock install --name sage conda-lock.yml
conda run -n sage python tools/install_locked_pip.py --environment sage
```

The first renderer invocation can compile the pinned `gsplat` CUDA extension.
It therefore requires the locked CUDA compiler toolchain and can make the
first preflight slower than subsequent runs. If a fixed architecture is
needed, set `TORCH_CUDA_ARCH_LIST` before preflight; the run receipt records
the value used.

## Models

SAGE requires the local SPNet source and the user-provided model artifacts.
The repository's model manifest records the expected model identities.

```bash
cd 00_Baselines/SAGE
git clone https://github.com/Wang-xjtu/SPNet.git third_party/SPNet
git -C third_party/SPNet checkout b836bd044517b33d3737094acd6a1f09c2362f04

export SAGE_MODEL_ROOT=/path/to/sage-models
python tools/download_models.py spnet-large-300 --source /path/to/Large_300.pth
python tools/download_models.py alexnet-imagenet --source /path/to/alexnet-owt-7be5be79.pth
```

For this thesis workspace the model directory is:

```text
/home/haibo/Documents/Thesis/00_Baselines/SAGE-models/
├── Large_300.pth
└── alexnet-owt-7be5be79.pth
```

`runtime.model_root` in the YAML configuration is authoritative when it is
present. A relative model path is resolved relative to the configuration
file, not relative to the current shell directory. Scene-specific configs
stored outside `00_Baselines/SAGE/configs` should therefore use an absolute
model path, or an explicitly correct relative path.

## Dataset and calibration layout

Calibration belongs to the capture device or camera rig, not to an individual
Scene. When several Scenes were recorded by the same device, keep one
canonical calibration at the dataset-group root:

```text
03_Datasets/001_Odin/
├── calibration.yaml       # one canonical runtime calibration for this rig
├── cam_in_ex.txt           # source calibration, if retained for provenance
├── Downtown1/
│   ├── metadata.yaml
│   └── *.db3
├── Ferrari1/
├── Graffiti1/
└── ...
```

Do not create a separate runtime calibration under every Scene when the
device calibration is identical. A different device, camera, lens, output
grid, or static transform gets its own dataset-group calibration.

The canonical runtime adapter accepts SAGE's strict `calibration.yaml`
schema. Odin supplies `cam_in_ex.txt`, so convert it once per device group:

```bash
cd /home/haibo/Documents/Thesis/00_Baselines/SAGE

conda run --no-capture-output -n sage \
  python tools/convert_calibration.py \
  --source /home/haibo/Documents/Thesis/03_Datasets/001_Odin/cam_in_ex.txt \
  --output /home/haibo/Documents/Thesis/03_Datasets/001_Odin/calibration.yaml \
  --reference-frame odom \
  --pose-frame odin1_base_link \
  --lidar-frame odin1_base_link \
  --camera-frame camera \
  --output-size 800 648 \
  --pose-from-lidar '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'
```

For the Odin bags in this workspace, `/odin1/cloud_slam` is already
registered in `odom`. Therefore its SAGE config declares
`input.lidar.points_frame: reference_frame`, which prevents applying the pose
chain to that point cloud a second time. A raw scan recorded in the LiDAR
frame must use `lidar_frame` and the corresponding calibration contract
instead.

Before sharing a calibration between Scenes, verify that the source
calibration bytes and device-level assumptions are the same. For example:

```bash
THESIS_ROOT=/home/haibo/Documents/Thesis
sha256sum "${THESIS_ROOT}"/03_Datasets/001_Odin/*/cam_in_ex.txt
```

## Configuration identity

`configs/sage.yaml` is the SAGE method baseline and currently points to the
local Ferrari1 capture. It contains the mapping, refinement, evaluation, and
ROSBAG synchronization settings. Its `input:` section is a concrete Scene
example, so it should not be treated as a universal dataset default.

For each Scene, create a separate run config derived from `configs/sage.yaml`
and change only the input identity and any machine-local runtime paths. Do not
edit a config after preflight or after training starts. The config is part of
the run identity and its SHA-256 is recorded in the artifacts.

For example, the Graffiti1 run config used in this workspace is:

```text
04_Outputs/SAGE/Odin/Graffiti1_sage.yaml
```

Its relevant fields are:

```yaml
runtime:
  model_root: /home/haibo/Documents/Thesis/00_Baselines/SAGE-models

input:
  type: rosbag2
  rosbag: /home/haibo/Documents/Thesis/03_Datasets/001_Odin/Graffiti1
  calibration: /home/haibo/Documents/Thesis/03_Datasets/001_Odin/calibration.yaml
  topics:
    lidar: /odin1/cloud_slam
    image: /odin1/image/compressed
    odometry: /odin1/odometry
  lidar:
    points_frame: reference_frame
    enable_fused: true
```

The full YAML remains strict: unknown fields and missing fields are rejected.
The command line carries run control only; it does not override the bag,
calibration, topics, or synchronization policy.

### Online mapping and evaluation policy

New runs can opt into a one-pass ROSBAG input by adding the explicit input
execution revision below. The frozen baseline remains `batch-v1`; do not
change an existing reproduction config in place.

```yaml
input:
  type: rosbag2
  execution: online-window-v2
  # rosbag, calibration, topics, lidar, synchronization, fusion, depth as above

evaluation:
  # Default when omitted: [stage2_refinement]
  checkpoint_stages: [stage2_refinement]
```

`online-window-v2` performs only calibration and topic-table checks before
mapping. Its one reader pass performs effective-time synchronization, frame
and PointCloud2-layout checks, projection checks, and the canonical sequence
digest while Stage 1 maps. It requires non-decreasing effective header times
in bag write order; use `batch-v1` for a bag that needs global reordering.

Stage 1 writes `.stage2-input-cache` beside `structure/`. This is a transient,
identity-bound handoff containing only mapping frames; Stage 2 reads it with
bounded CPU/GPU LRUs and never replays an online ROSBAG. The cache remains
after a failed Stage 2 or Stage 3 attempt for retry, and is removed only after
the complete three-stage pipeline publishes successfully.

To compare both checkpoints in one Stage 3 input pass, make the policy
explicit:

```yaml
evaluation:
  min_alpha: 0.01
  epsilon: 0.000001
  alpha_support: 0.01
  hit_target_center: 0.5
  hit_target_fused: 0.35
  checkpoint_stages: [stage1_mapping, stage2_refinement]
```

This writes separate reports below `evaluation/stage1_mapping/` and
`evaluation/stage2_refinement/`. Stage 1 reports are marked as mapping-only;
the default final-quality report remains the refinement checkpoint.

## ROSBAG adapter contract

The `rosbag2` adapter owns bag decoding, timestamp association, pose
interpolation, image rectification, LiDAR projection, and centered-window
fusion. The current baseline uses:

- image header stamps as the canonical frame anchors;
- nearest LiDAR association within 20 ms;
- linear-slerp pose interpolation with a maximum 100 ms pose gap and no
  extrapolation;
- a centered five-scan fusion window;
- nearest-depth z-buffering and conflict rejection;
- depth limits of 0.1 m to 200 m;
- evaluation over all accepted canonical frames.

The `batch-v1` adapter performs a full preflight before mapping. It checks the bag schema,
declared topics, message types, frame IDs, synchronization plan, calibration,
projection coverage, CUDA, renderer, SPNet, and metric dependencies.

## Preflight and end-to-end training

Run commands from the SAGE repository root. This is required by the formal
fresh-process execution boundary and its execution receipt.

Set the Scene-specific paths once:

```bash
export THESIS_ROOT=/home/haibo/Documents/Thesis
export SAGE_ROOT="${THESIS_ROOT}/00_Baselines/SAGE"
export SCENE_NAME=Graffiti1
export DEVICE=cuda
export CONFIG="${THESIS_ROOT}/04_Outputs/SAGE/Odin/${SCENE_NAME}_sage.yaml"
export OUTPUT="${THESIS_ROOT}/04_Outputs/SAGE/Odin/${SCENE_NAME}"
```

Preflight does not start training and should be run against an unused output
identity:

```bash
cd "${SAGE_ROOT}"

conda run --no-capture-output -n sage \
  python -m sage train \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --preflight
```

After preflight succeeds, launch the complete three-stage run explicitly:

```bash
cd "${SAGE_ROOT}"

conda run --no-capture-output -n sage \
  python -m sage train \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}"
```

The command runs, in order:

1. structure mapping with the configured mapping iterations and pruning;
2. appearance refinement with the configured refinement iterations;
3. final evaluation over all accepted frames.

There is no need to run a separate evaluation command for a normal complete
training run. To publish a separate evaluation output later:

```bash
conda run --no-capture-output -n sage \
  python -m sage evaluate \
  --checkpoint "${OUTPUT}/final/appearance_checkpoint.pt" \
  --config "${CONFIG}" \
  --output "${OUTPUT}/evaluation_republished" \
  --device "${DEVICE}"
```

## Output layout

For the Graffiti1 Scene, the output is explicitly namespaced below
`04_Outputs/SAGE/Odin/Graffiti1/`:

```text
04_Outputs/SAGE/Odin/Graffiti1/
├── structure/
│   ├── checkpoint.pt
│   ├── map.ply
│   ├── spnet_dense.pt
│   └── run_manifest.json
├── structure.execution.json
├── final/
│   ├── appearance_checkpoint.pt
│   ├── appearance_map.ply
│   └── run_manifest.json
├── evaluation/
│   ├── evaluation.json
│   └── run_manifest.json
└── run_manifest.json
```

`structure/spnet_dense.pt` is an identity- and manifest-bound cache of dense
SPNet predictions computed during mapping. Appearance refinement reuses it.

The structure execution receipt records the fresh child process, exit code,
wall-clock time, peak RSS when available, formal config identity, and artifact
hashes. Manifests also record the canonical sequence identity, canonical input
contract, adapter/source provenance, training identity, model identities,
renderer identity, and worktree state.

Existing complete outputs are never silently overwritten. Use a new Scene/run
directory for a new experiment; keep the config and output directory paired so
the recorded identities remain auditable.

## Prepared Scene (optional)

Training does not export a Prepared Scene as a side effect. If a stable,
replayable input artifact is needed, export it explicitly:

```bash
conda run --no-capture-output -n sage \
  python -m sage prepare \
  --config "${CONFIG}" \
  --output /path/to/prepared-scene
```

The exported Scene replays the same canonical frames and can be used by a
separate config with `input.type: prepared_scene`. A ROSBAG run and its
equivalent Prepared Scene share the canonical sequence identity, while their
adapter provenance remains distinguishable.

## Common failure points

- `calibration.yaml` not found: create the device-level file once with
  `tools/convert_calibration.py` and point every Scene config to it.
- Frame-ID mismatch: inspect the bag's odometry and point-cloud header frame
  IDs, then correct the calibration contract or `points_frame`; do not disable
  the check.
- No projected LiDAR coverage: verify the direct `Tcl_0` /`camera_from_lidar`
  convention, camera output grid, point-frame declaration, and calibration
  units.
- Missing models or renderer errors: run preflight in `sage` and check
  `runtime.model_root`, `third_party/SPNet`, and the locked CUDA toolchain.
- Existing output refusal: choose a new Scene/run output identity instead of
  deleting or overwriting an auditable run.

SAGE targets finite, calibrated research scenes. It is not a real-time ROS 2
or safety-critical navigation system.
