import hashlib
import os

import numpy as np

from sage.data.scene import _resize_rgb
from sage.foundation.hashing import sha256_file


def test_resize_rgb_uint8_fast_path_matches_float_path() -> None:
    rng = np.random.default_rng(0)
    image_uint8 = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    image_float = image_uint8.astype(np.float32) / 255.0

    via_uint8 = _resize_rgb(image_uint8, (48, 32))
    via_float = _resize_rgb(image_float, (48, 32))

    np.testing.assert_array_equal(via_uint8, via_float)


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"sage-hashing-test-payload")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected


def test_sha256_file_recomputes_after_content_and_mtime_change(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"first")
    first = sha256_file(path)

    path.write_bytes(b"second")
    future = os.stat(path).st_mtime_ns + 1_000_000
    os.utime(path, ns=(future, future))
    second = sha256_file(path)

    assert first != second
    assert second == hashlib.sha256(b"second").hexdigest()
