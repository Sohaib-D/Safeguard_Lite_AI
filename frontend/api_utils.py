from __future__ import annotations
import os
import sys
import streamlit as st
from pathlib import Path
from frontend.api_client import APIClientError, SafeguardAPIClient
from frontend.logging_config import configure_logger

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = configure_logger("safeguard.frontend.api", "logs/frontend.log")

def get_client() -> SafeguardAPIClient:
    return SafeguardAPIClient(
        base_url=st.session_state.get("api_base_url", os.environ.get("SAFEGUARD_API_BASE_URL", "http://127.0.0.1:8000")),
        token=st.session_state.get("auth_token"),
    )

def run_api_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except APIClientError as exc:
        logger.warning(
            "Frontend API call failed.",
            extra={
                "event_type": "frontend_api_error",
                "details": {
                    "message": exc.message,
                    "errors": exc.errors,
                    "status_code": exc.status_code,
                },
            },
        )
        return None, exc
    except Exception as exc:
        logger.exception(
            "Unexpected frontend exception.", extra={"event_type": "frontend_exception"}
        )
        return None, APIClientError(str(exc))

def fetch_model_info(force: bool = False) -> dict | None:
    if st.session_state.get("model_info_cache") is not None and not force:
        return st.session_state["model_info_cache"]
    result, err = run_api_call(get_client().model_info)
    if err is None:
        st.session_state["model_info_cache"] = result
        return result
    return None
