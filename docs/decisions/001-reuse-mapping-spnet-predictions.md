# ADR-001: Reuse Mapping SPNet Predictions in Refinement

## Status

Accepted

## Date

2026-08-01

## Context

Mapping and appearance refinement consume different post-processed views of
the same deterministic SPNet prediction. Running the network again for every
mapping frame during refinement duplicates GPU work and scene decoding. The
mapping checkpoint format is strict and should remain focused on Gaussian
state, while cached predictions are potentially hundreds of megabytes.

## Decision

Mapping publishes dense float32 predictions as the separate
`structure/spnet_dense.pt` artifact. The structure manifest binds the cache by
SHA-256 and records its frame count and SPNet identity. Refinement accepts the
cache only when the same manifest also binds its source checkpoint, validates
the cache identity against the configured online provider, and records cache
provenance in the refined checkpoint. Missing cache frames use the verified
online provider, preserving compatibility with older structure outputs.

## Alternatives Considered

### Store predictions inside `checkpoint.pt`

Rejected because it couples large, immutable evidence to trainable Gaussian
state and would require changing the strict checkpoint version.

### Use an unrecorded local cache

Rejected because results would depend on mutable external state that is absent
from experiment provenance.

### Run mapping and refinement in one process

Rejected because it removes the fresh-process evidence boundary and weakens
the existing resumability contract.

## Consequences

- Typical refinement performs one live SPNet inference instead of one per
  mapping frame.
- Structure outputs gain an optional large sidecar artifact.
- Cache corruption or identity mismatch fails closed rather than silently
  recomputing a mixture of untrusted and live predictions.
- Older structure outputs remain valid and use the original online path.
