import streamlit as st
import pandas as pd
from frontend.app import get_client, run_api_call, render_api_error

st.set_page_config(
    page_title="Active Target Scanner",
    page_icon="🎯",
    layout="wide",
)

def render_active_scanner():
    st.title("🎯 Active Target Scanner")
    st.markdown("""
    **Welcome to the Active Scanner!** 
    This tool allows you to safely check a computer or website (like `192.168.1.5` or `example.com`) to see what "doors" (ports) are open and what security risks might exist. 
    
    *Note: We only perform safe reconnaissance (looking around). We do not launch actual attacks like DDoS or Bruteforcing.*
    """)

    with st.expander("📖 What is Port Scanning?"):
        st.write("""
        Imagine a computer is a house, and **ports** are the doors and windows. 
        - Port 80 (HTTP) is like the front door for web traffic.
        - Port 22 (SSH) is like a secure back door for administrators.
        
        If a door is open and the lock (password/security) is weak, hackers can break in. This scanner checks which doors are open so you can secure them!
        """)

    target = st.text_input("Enter IP Address or Domain Name (e.g., 127.0.0.1 or google.com)", value="127.0.0.1")

    if st.button("Start Safe Reconnaissance Scan", type="primary"):
        if not target:
            st.error("Please enter a target.")
            return

        with st.spinner(f"Scanning {target}... This may take a minute as we check common ports."):
            client = get_client()
            result, err = run_api_call(client.active_scan, target=target)

        if err:
            render_api_error(err)
            return

        if result.get("error"):
            st.error(f"Scan failed: {result['error']}")
            return

        st.success("Scan Complete!")

        # 1. DNS & Latency
        st.subheader("🌐 Basic Info (Reconnaissance)")
        col1, col2 = st.columns(2)
        dns_info = result.get("dns", {})
        col1.metric("Resolved IP Address", dns_info.get("ip_address", "N/A"))
        col2.metric("Response Time (Latency)", f"{result.get('latency_ms', 'N/A')} ms")

        # 2. Open Ports
        st.subheader("🚪 Open Ports & Vulnerability Context")
        ports = result.get("ports", [])
        if not ports:
            st.success("No common open ports found. This is usually good for security!")
        else:
            st.warning(f"Found {len(ports)} open port(s). Review the context below.")
            for p in ports:
                with st.expander(f"Port {p['port']} ({p['service']}) - OPEN", expanded=True):
                    st.write(f"**Description:** {p['description']}")
                    st.write(f"**Security Context:** {p['vulnerability_context']}")

                    if p['port'] == 22:
                        st.info("💡 **Simulated Bruteforce Check:** If a hacker sees this port, they will try thousands of password combinations. Ensure you use strong passwords or SSH keys!")
                    elif p['port'] == 80:
                        st.error("💡 **Security Warning:** Traffic on this port is unencrypted. Anyone listening on the network can see the data.")

        # 3. SSL Info
        if result.get("ssl"):
            st.subheader("🔒 SSL/TLS Certificate (Encryption)")
            st.write("We found a secure web port (443). Here is the encryption certificate data:")
            st.json(result["ssl"])
            
        # 4. HTTP Headers
        if result.get("http_headers"):
            st.subheader("📄 Web Server Information")
            st.write("The web server told us the following details about itself:")
            st.json(result["http_headers"])

        # 5. WHOIS
        if result.get("whois"):
            st.subheader("🏢 Domain Registration (WHOIS)")
            st.write("Publicly available information about who owns this domain:")
            st.json(result["whois"])

if __name__ == "__main__":
    # Ensure they are logged in if navigating directly
    if not st.session_state.get("auth_token"):
        st.warning("Please sign in from the main dashboard to use the Active Scanner.")
    else:
        render_active_scanner()
