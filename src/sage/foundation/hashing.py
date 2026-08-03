from __future__ import annotations

import hashlib
from pathlib import Path

# ponytail: in-process memoization keyed by (path, mtime_ns, size). A single
# Prepared Scene's input files get hashed 5-7 times within one stage
# (start_identity / _validate_contract / prepare_close). Cross-stage reuse
# would need an on-disk cache; not needed here.
# Cost: a file rewritten with unchanged mtime_ns and size returns a stale digest.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    resolved = Path(path)
    stat = resolved.stat()
    key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    cached = _DIGESTS.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _DIGESTS[key] = result
    return result
