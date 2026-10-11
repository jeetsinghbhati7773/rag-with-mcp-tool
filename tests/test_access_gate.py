from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import pytest
from src.config import Settings
from src.ui.components import render_access_gate


def test_settings_loads_app_access_code():
    with patch.dict(os.environ, {"APP_ACCESS_CODE": "secret_pin_987"}):
        custom_settings = Settings()
        assert custom_settings.app_access_code == "secret_pin_987"


def test_render_access_gate_already_authenticated():
    with patch("streamlit.session_state", {"authenticated": True}):
        assert render_access_gate() is True


def test_render_access_gate_when_no_code_configured():
    with patch("src.config.settings.app_access_code", ""):
        with patch("streamlit.session_state", {"authenticated": False}):
            assert render_access_gate() is True


def test_render_access_gate_unauthenticated_blocks():
    mock_form = MagicMock()
    mock_form.__enter__.return_value = None
    mock_form.__exit__.return_value = None

    with patch("src.config.settings.app_access_code", "1234"), \
         patch("streamlit.session_state", {"authenticated": False}), \
         patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock()]), \
         patch("streamlit.markdown"), \
         patch("streamlit.form", return_value=mock_form), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.form_submit_button", return_value=False):
        
        assert render_access_gate() is False
