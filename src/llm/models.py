from __future__ import annotations

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm():
    """
    Returns an LLM instance (ChatGroq with OpenAI fallback).
    Supports GROQ_MODEL environment variable override.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key:
        try:
            import streamlit as st
            groq_key = st.secrets.get("GROQ_API_KEY")
            openai_key = openai_key or st.secrets.get("OPENAI_API_KEY")
        except Exception as e:
            logger.debug("Could not read keys from Streamlit secrets: %s", e)

    if groq_key:
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_key,
            max_retries=5,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            max_retries=5,
        )
    else:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (or OPENAI_API_KEY) in your "
            ".env file before invoking the assistant."
        )


class LazyLLM:
    """Lazy proxy wrapper to avoid instantiating LLM at module import time."""
    _instance = None

    def __getattr__(self, name: str):
        if LazyLLM._instance is None:
            LazyLLM._instance = get_llm()
        return getattr(LazyLLM._instance, name)

    def invoke(self, *args, **kwargs):
        if LazyLLM._instance is None:
            LazyLLM._instance = get_llm()
        return LazyLLM._instance.invoke(*args, **kwargs)


llm = LazyLLM()
