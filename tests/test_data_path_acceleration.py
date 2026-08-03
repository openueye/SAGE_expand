import hashlib
import os

from sage.foundation.hashing import sha256_file


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
