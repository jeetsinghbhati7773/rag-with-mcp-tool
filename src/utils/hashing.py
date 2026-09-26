from __future__ import annotations

import hashlib
from typing import Union


def calculate_sha256(content: Union[bytes, str]) -> str:
    """Calculates SHA-256 hash string for given bytes or string content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Formats bytes into human readable KB / MB string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
