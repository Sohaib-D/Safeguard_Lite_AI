import streamlit as st
import pandas as pd
import time
from frontend.app import get_client, run_api_call, render_api_error, apply_custom_css
from frontend.pages.Capture_Control import start_capture, stop_capture, get_capture_stats

def render_live_monitor():
    apply_custom_css()
    st.title("📡 Live Network Monitor")
    st.markdown("""
    This page shows you what is happening on your network **right now**. 
    It acts like a security camera for your internet connection, watching for suspicious activity.
    """)

    with st.expander("📖 How does this work?"):
        st.write("""
        When you click **Start Live Capture**, our AI starts listening to the traffic flowing through your computer's network card.
        - If someone tries to scan your computer (looking for open doors), we will catch it.
        - If someone tries to guess your passwords (brute force), we will catch it.
        - If a program sends way too much data (potential data theft), we will catch it.
        
        *Don't worry, we only look at the 'envelopes' of the data, not the contents inside!*
        """)

    col1, col2 = st.columns(2)

    stats = get_capture_stats()
    is_running = stats.get('running', False) if isinstance(stats, dict) else False

    with col1:
        if not is_running:
            st.info("The network camera is currently OFF.")
            if st.button("▶️ Start Live Capture", type="primary"):
                with st.spinner("Starting..."):
                    res = start_capture()
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.success("Started successfully!")
                        time.sleep(1)
                        st.rerun()
        else:
            st.success("🟢 The network camera is ON and watching for threats.")
            if st.button("⏹️ Stop Live Capture", type="secondary"):
                with st.spinner("Stopping..."):
                    res = stop_capture()
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.success("Stopped successfully!")
                        time.sleep(1)
                        st.rerun()

    with col2:
        flows_count = stats.get('flows_count', 0) if isinstance(stats, dict) else 0
        ip_count = stats.get('ip_count', 0) if isinstance(stats, dict) else 0
        
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="section-label">Total Connections Monitored</div>
            <h2 style="margin:0; color:#fff;">{flows_count}</h2>
        </div>
        <div class="metric-card">
            <div class="section-label">Devices Tracked</div>
            <h2 style="margin:0; color:#fff;">{ip_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚨 Real-Time Threat Alerts")
    
    # In a real app, this would use websockets. For Streamlit, we will use an auto-refresh pattern or just pull alerts from the API.
    # We will fetch recent alerts from the backend.
    
    if st.button("🔄 Refresh Alerts"):
        pass # Streamlit reruns the script on button click, which will fetch fresh data below.

    client = get_client()
    alerts_data, err = run_api_call(client._handle_response, __import__("requests").get(f"{client.base_url}/api/v1/alerts", headers=client._headers()))

    if err:
        render_api_error(err)
        return

    if not alerts_data:
        st.success("No threats detected recently! Your network looks safe.")
    else:
        st.warning(f"Found {len(alerts_data)} recent alerts!")
        df = pd.DataFrame(alerts_data)
        if not df.empty:
            for _, row in df.iterrows():
                severity = row.get('severity', 'Medium')
                color = "🔴" if severity == "High" else "🟡"
                with st.expander(f"{color} {severity} Risk: {row.get('threat_type', 'Suspicious Activity')}", expanded=(severity=="High")):
                    st.write(f"**Description:** {row.get('description')}")
                    if row.get('src_ip'):
                        st.write(f"**Source IP:** `{row.get('src_ip')}`")
                    st.write(f"**Time:** {row.get('created_at')}")
                    
                    st.markdown("### 🤖 AI Explanation")
                    st.info("Click 'Analyze with AI' in the Security Center to get a plain-English explanation of this threat.")

if __name__ == "__main__":
    if not st.session_state.get("auth_token"):
        st.warning("Please sign in from the main dashboard to use the Live Monitor.")
    else:
        render_live_monitor()
