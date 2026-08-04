# ADR-003: Use a Median Z-Buffer for Centered Fused5 Depth

## Status

Accepted

## Date

2026-08-04

## Context

The centered Fused5 adapter projects the two preceding, center, and two
following scans into the center camera. The center observation remains
authoritative wherever it is valid. For center holes, the previous
implementation selected the minimum valid neighbor depth per pixel.

That minimum is not a neutral temporal fusion rule: when multiple projected
neighbors are valid, it is an order statistic that is biased toward a nearer
candidate. On the smallest strict paired TUM RGB-D experiment
(`freiburg1_desk`, five accepted center frames), the current fused pixels had
an average `nearest - median` residual of approximately `-2.05 mm`; the `+2`
temporal offset supplied 57.26% of the selected fused pixels.

## Decision

Use the per-pixel median of valid projected candidates as the default Fused5
z-buffer. Keep `nearest` as an explicit compatibility option for reproducing
older runs.

Do not make `min_support_count=2` the formal default. It reduced the fused
observation set by 43.6% in the paired experiment and changed the image/depth
trade-off in a way that requires broader dataset validation.

## Evidence

The following runs used the same five center images, poses, center scans, and
center masks. Reported depth MAE is render-fit residual against the input
observation, not ground-truth sensor error.

| Strategy | Fused pixels | Fused MAE | Total MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| nearest, support 1 | 126,910 | 54.78 mm | 14.65 mm | 18.686 |
| median, support 1 | 126,910 | 52.90 mm | 14.44 mm | 18.676 |
| nearest, support 2 | 71,541 | 40.72 mm | 12.61 mm | 18.066 |
| median, support 2 | 71,541 | 40.63 mm | 12.67 mm | 18.056 |

The median improvement remains small but consistent when support is 1. When
support is restricted to 2, nearest and median become nearly indistinguishable;
this indicates that sparse single-neighbor evidence is a larger remaining
factor than the z-buffer order statistic.

## Alternatives Considered

### Keep nearest as the default

Rejected for new formal runs because it introduces a measurable nearer-depth
selection bias. It remains available explicitly for historical reproduction.

### Require at least two supporting neighbors

Deferred. It improves the render-fit depth residual in this five-frame test,
but discards many observations and lowers image quality. It needs validation on
larger and more varied sequences before becoming a formal policy.

### Average all valid candidates

Not selected. The median is robust to a single occlusion/outlier candidate and
does not make the mean sensitive to the full depth spread.

## Consequences

- New configurations that omit `fusion.z_buffer` use `median` and record it in
  the canonical input provenance.
- Existing runs remain reproducible by setting `fusion.z_buffer: nearest`.
- The conflict rejection rule is unchanged.
- Future validation should prioritize support-count distributions and evaluate
  both image quality and depth fit; the latter is not a ground-truth metric.
