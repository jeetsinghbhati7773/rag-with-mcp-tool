from __future__ import annotations

import os
from typing import Tuple
from src.security.limits import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES


def validate_file_upload(file_name: str, file_size_bytes: int) -> Tuple[bool, str]:
    """
    Validates uploaded document file type and size limit.
    Returns (is_valid, error_message).
    """
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return False, f"File size ({file_size_bytes / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."

    return True, ""


def sanitize_input_query(query: str) -> str:
    """Sanitizes user query string."""
    return query.strip()
