from __future__ import annotations

import streamlit as st


def apply_theme():
    """Applies sleek dark theme styling and custom CSS tokens for AI Knowledge Assistant."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #1E293B;
            border-right: 1px solid #334155;
        }
        .citation-badge {
            background-color: #334155;
            color: #38BDF8;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .tool-badge {
            background-color: #1E1B4B;
            color: #818CF8;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            border: 1px solid #4338CA;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
