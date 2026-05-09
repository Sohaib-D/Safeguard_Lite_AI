"""
Packet Capture Control Page for Streamlit
"""

import sys
import streamlit as st
import requests
import time
from pathlib import Path
from typing import Dict, Any

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from frontend.api_utils import get_client, run_api_call, fetch_model_info
from frontend.ui_components import apply_custom_css, render_topbar, render_api_error

API_BASE = "http://127.0.0.1:8000"

def get_capture_stats() -> Dict[str, Any]:
    """Get capture statistics from API"""
    try:
        response = requests.get(f"{API_BASE}/api/v1/capture/stats", timeout=5)
        return response.json()
    except:
        return {"error": "Unable to connect to backend"}

def start_capture(interface: str = None):
    """Start packet capture"""
    try:
        data = {"interface": interface} if interface else {}
        response = requests.post(f"{API_BASE}/api/v1/capture/start", json=data, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def stop_capture():
    """Stop packet capture"""
    try:
        response = requests.post(f"{API_BASE}/api/v1/capture/stop", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    apply_custom_css()
    render_topbar()
    st.title("🛡️ Packet Capture Control")

    st.markdown("""
    Control real-time network packet capture and monitoring.
    This feature requires administrative privileges and network access.
    """)

    # Status section
    st.header("Capture Status")
    stats = get_capture_stats()

    if "error" in stats:
        st.error(f"Backend connection error: {stats['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "Running" if stats.get('running') else "Stopped")
        with col2:
            st.metric("Interface", stats.get('interface', 'N/A'))
        with col3:
            st.metric("Queue Size", stats.get('queue_size', 0))

        if stats.get('running'):
            st.success("Packet capture is active")
        else:
            st.info("Packet capture is stopped")

    # Control section
    st.header("Controls")

    col1, col2 = st.columns(2)

    with col1:
        interface = st.text_input("Network Interface (optional)", placeholder="eth0")
        if st.button("Start Capture", type="primary"):
            with st.spinner("Starting capture..."):
                result = start_capture(interface if interface else None)
            if "error" in result:
                st.error(f"Failed to start: {result['error']}")
            else:
                st.success(f"Capture started on {result.get('interface', 'default')}")
                time.sleep(1)
                st.rerun()

    with col2:
        if st.button("Stop Capture", type="secondary"):
            with st.spinner("Stopping capture..."):
                result = stop_capture()
            if "error" in result:
                st.error(f"Failed to stop: {result['error']}")
            else:
                st.success("Capture stopped")
                time.sleep(1)
                st.rerun()

    # Threat feed (placeholder for real-time updates)
    st.header("Recent Detections")
    st.info("Real-time threat detections will appear here when capture is active.")

    # Performance metrics
    st.header("Performance Metrics")
    if "error" not in stats:
        st.json({
            "flows_tracked": stats.get('flows_count', 0),
            "ips_monitored": stats.get('ip_count', 0),
            "queue_size": stats.get('queue_size', 0)
        })

if __name__ == "__main__":
    main()
