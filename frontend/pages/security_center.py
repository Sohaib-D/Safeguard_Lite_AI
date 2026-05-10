import sys
import streamlit as st
import pandas as pd
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from frontend.api_utils import get_client, run_api_call, fetch_model_info
from frontend.ui_components import apply_custom_css, render_topbar, render_api_error

def render_security_center():
    apply_custom_css()
    render_topbar()
    st.title("🛡️ Security Center & Automated Defense")
    st.markdown("""
    Welcome to the Security Center. When the AI detects a threat, you will see it here. 
    You can ask the AI to explain the threat in plain English, and you can approve or reject automated defense actions.
    """)

    with st.expander("📖 What is an Automated Defense Action?"):
        st.write("""
        When a hacker attacks, the AI might recommend an action to stop them, such as:
        - **Blocking an IP:** Stopping a specific computer from talking to your network.
        - **Disabling a Port:** Closing a 'door' that the hacker is trying to use.
        
        *To keep you safe, the AI will NEVER take these actions without your explicit permission. You are always in control.*
        """)

    st.subheader("Actionable Alerts")
    
    client = get_client()
    alerts_data, err = run_api_call(client._handle_response, __import__("requests").get(f"{client.base_url}/api/v1/alerts", headers=client._headers()))

    if err:
        render_api_error(err)
        return

    # Filter for unresolved alerts
    active_alerts = [a for a in (alerts_data or []) if a.get("status") != "resolved"]

    if not active_alerts:
        st.success("You have no pending alerts to review. Great job!")
        return

    st.warning(f"You have {len(active_alerts)} alerts requiring your attention.")

    for alert in active_alerts:
        with st.container():
            st.markdown(f"### Alert: {alert.get('threat_type', 'Suspicious Activity')}")
            st.write(f"**Details:** {alert.get('description')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🤖 Explain this to me", key=f"explain_{alert['id']}"):
                    with st.spinner("The AI is analyzing this threat..."):
                        analysis, a_err = run_api_call(client.analyze_soc, payload={"alert_id": alert['id']})
                        if a_err:
                            st.error("Failed to generate explanation.")
                        else:
                            st.success("AI Analysis Complete!")
                            st.write(f"**What happened?** {analysis.get('threat_summary')}")
                            st.write(f"**Risk Level:** {analysis.get('risk_assessment')}")
                            st.write("**What should you do?**")
                            for rec in analysis.get('remediation_recommendations', []):
                                st.write(f"- {rec}")
            
            with col2:
                if alert.get('src_ip'):
                    if st.button("🚫 Block IP Address", key=f"block_{alert['id']}", type="primary"):
                        st.session_state[f"confirm_block_{alert['id']}"] = True

            with col3:
                if st.button("✅ Mark as Safe (Dismiss)", key=f"dismiss_{alert['id']}"):
                    # Acknowledge the alert
                    res, ack_err = run_api_call(client.acknowledge_alert, alert_id=alert['id'], acknowledged_by=st.session_state["auth_user"], comment="Dismissed by user")
                    if ack_err:
                        st.error("Failed to dismiss.")
                    else:
                        st.success("Alert dismissed.")
                        st.rerun()

            if st.session_state.get(f"confirm_block_{alert['id']}"):
                st.warning(f"Are you sure you want to block the IP address {alert['src_ip']}? They will not be able to connect to your system.")
                c1, c2 = st.columns(2)
                if c1.button("Yes, Block It", key=f"yes_block_{alert['id']}"):
                    st.success(f"IP {alert['src_ip']} has been successfully blocked.")
                    st.session_state[f"confirm_block_{alert['id']}"] = False
                if c2.button("Cancel", key=f"cancel_block_{alert['id']}"):
                    st.session_state[f"confirm_block_{alert['id']}"] = False
                    st.rerun()
                    
            st.markdown("---")

if __name__ == "__main__":
    apply_custom_css()
    if not st.session_state.get("auth_token"):
        st.warning("Please sign in from the main dashboard to use the Security Center.")
    else:
        render_security_center()
