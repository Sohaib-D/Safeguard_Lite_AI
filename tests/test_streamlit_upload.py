from __future__ import annotations

import pytest

streamlit = pytest.importorskip("streamlit")
stpytest = pytest.importorskip("stpytest")


@pytest.mark.skip(
    reason="stpytest/streamlit UI harness is not installed in this environment."
)
def test_streamlit_upload_placeholder():
    assert streamlit is not None
    assert stpytest is not None
