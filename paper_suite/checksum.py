"""Lightweight, non-cryptographic checks for accidental changes to files."""
from __future__ import annotations

import json
from pathlib import Path
import zlib


def checksum_bytes(data: bytes) -> dict:
    """CRC32 and byte count; no authenticity or collision-resistance guarantee."""
    return {"size_bytes": len(data), "crc32": f"{zlib.crc32(data) & 0xffffffff:08x}"}


def checksum_file(path: str | Path) -> dict:
    size, crc = 0, 0
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(data)
            crc = zlib.crc32(data, crc)
    return {"size_bytes": size, "crc32": f"{crc & 0xffffffff:08x}"}


def checksum_json(obj, *, default=None) -> str:
    """Stable short identifier for cache keys and protocol objects."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=default).encode("utf-8")
    check = checksum_bytes(data)
    return f"{check['crc32']}-{check['size_bytes']}"
