# ADR-001: Online ROSBAG input uses an EOF-settled contract and transient Stage 2 cache

## Status

Accepted

## Date

2026-08-04

## Context

The original three-stage workflow compiled a global ROSBAG plan before Stage
1 and independently replayed the bag for Stage 2 and Stage 3. Reading an
identity before consuming frames could itself trigger a further full pass.
That made long sequences IO-bound before optimization started and coupled
checkpoint verification to the global replay implementation.

Stage 2 needs random access over mapping frames for its seeded optimization,
so it cannot preserve its existing deterministic semantics as a pure one-pass
refinement. Stage 3 can evaluate multiple checkpoints per frame, but should
not read the same ROSBAG once per checkpoint.

## Decision

Add an explicit `input.execution: online-window-v2` mode while preserving the
legacy `batch-v1` configuration and artifacts.

- Startup checks only calibration and declared topic metadata.
- The one online reader pass verifies effective-time monotonicity, frame IDs,
  PointCloud2 layout, synchronization, projection coverage and canonical frame
  content while Stage 1 maps.
- An online contract has a deferred frame count and may be consumed once.
  Its canonical sequence identity settles at EOF; identity access before EOF
  is an error rather than an implicit replay. The v2 canonical-contract
  identity deliberately normalizes the deferred frame count, while the final
  count remains bound by the sequence digest.
- Stage 1 atomically writes only its mapping cohort to
  `.stage2-input-cache`. Stage 2 verifies the cache against the checkpoint
  identity, uses bounded CPU and GPU LRUs, and persists dense priors on disk
  rather than retaining all frames or targets in memory.
- The cache is not a published artifact. It stays after an unsuccessful Stage
  2/3 run for retry and is removed after complete pipeline success.
- `evaluation.checkpoint_stages` defaults to `stage2_refinement`. An explicit
  policy can request both `stage1_mapping` and `stage2_refinement`; Stage 3
  consumes one input stream and renders models serially for every frame,
  writing one output directory per checkpoint stage.

## Alternatives considered

### Global preflight with a separate identity replay

It provides complete ordering information before Stage 1 but repeats the
expensive decode/association work and makes logging or validation cause hidden
replays.

### Pure streaming Stage 2

It would change the current seeded random frame-selection semantics and make
the 4000-step refinement depend on non-repeatable historical state. A bounded
disk handoff preserves semantics without another ROSBAG read.

### Re-evaluate each checkpoint independently

It is simple but doubles ROSBAG decoding and assembly for comparison runs.
Serial per-frame rendering keeps outputs separate while sharing the input
stream.

## Consequences

- `online-window-v2` rejects an effective-time out-of-order bag; users must
  select `batch-v1` if they require global reordering.
- Existing v1 configs retain their behavior and serialized contract revision.
- Online contracts use revision 2 and do not silently exchange checkpoints
  with old revision-1 prepared scenes.
- A failed or interrupted complete pipeline may intentionally leave the
  transient cache on disk until the user retries or removes the incomplete run.
