from __future__ import annotations

from src.config import settings

MAX_FILE_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_RETRIEVAL_K = settings.retrieval_k
MAX_DOCUMENTS = settings.max_documents
MAX_CHUNKS_PER_DOCUMENT = settings.max_chunks_per_document
MCP_TIMEOUT_SECONDS = settings.mcp_timeout_seconds
