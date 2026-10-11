from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_env_or_secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key)
        except Exception:
            pass
    return str(val) if val is not None else default


@dataclass
class Settings:
    """Centralized configuration settings for AI Knowledge Assistant."""

    # Models
    groq_model: str = field(default_factory=lambda: _get_env_or_secret("GROQ_MODEL", "llama-3.3-70b-versatile"))
    openai_model: str = field(default_factory=lambda: _get_env_or_secret("OPENAI_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(default_factory=lambda: _get_env_or_secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # PostgreSQL Database URL
    database_url: str = field(
        default_factory=lambda: _get_env_or_secret(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/ai_knowledge_assistant",
        )
    )

    # RAG Parameters
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "5")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "3")))

    # Vector DB Storage
    chroma_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CHROMA_DIR", str(Path(__file__).resolve().parent.parent / "chroma_db")))
    )
    collection_name: str = "knowledge_base"

    # Security & Limits
    max_file_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "20")))
    max_documents: int = field(default_factory=lambda: int(os.getenv("MAX_DOCUMENTS", "50")))
    max_chunks_per_document: int = field(default_factory=lambda: int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "5000")))
    max_chat_history: int = field(default_factory=lambda: int(os.getenv("MAX_CHAT_HISTORY", "15")))
    mcp_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("MCP_TIMEOUT_SECONDS", "15")))

    # Confidence Thresholds
    confidence_high_threshold: float = 0.65
    confidence_moderate_threshold: float = 0.40


settings = Settings()
