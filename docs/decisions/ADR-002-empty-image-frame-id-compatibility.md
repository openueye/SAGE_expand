# ADR-002: Explicit compatibility for empty image frame IDs in online ROSBAG input

## Status

Accepted

## Date

2026-08-04

## Context

The Ferrari1 ROS 2 bag contains compressed-image messages whose
`Header.frame_id` is empty. SAGE does not use that header value to transform
the image: the configured calibration and canonical camera grid own the image
geometry. The online adapter nevertheless recorded one error per image and
raised it at EOF, after Stage 1 had already processed most of the sequence.

The frozen `batch-v1` contract should remain strict, and a non-empty frame ID
that contradicts calibration must not be silently accepted.

## Decision

Add `input.allow_empty_image_frame_id`, defaulting to `false`. It is valid only
with `execution: online-window-v2`. When enabled, the online adapter accepts an
empty image `Header.frame_id` while continuing to reject any non-empty value
that differs from the calibration camera frame. The setting is included in
adapter provenance so the exception is part of run identity and audit data.

## Alternatives considered

### Rewrite the original ROSBAG in place

Rejected because it changes the source artifact and weakens reproducibility.

### Treat every image frame ID as optional

Rejected because a non-empty contradictory frame ID can indicate a wrong topic
or calibration and should remain a hard input-contract error.

### Make the frozen batch adapter permissive

Rejected because it would change the established reproduction contract. The
compatibility is limited to the explicit online execution revision.

## Consequences

- Ferrari1 can use its original bag without a multi-gigabyte data rewrite.
- Runs using the compatibility option remain distinguishable in their input
  configuration and adapter provenance.
- The original bag still has incomplete image frame metadata; this option
  documents and scopes the accepted limitation rather than repairing the bag.
