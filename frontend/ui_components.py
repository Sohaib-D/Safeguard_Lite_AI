import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from frontend.api_client import APIClientError
from frontend.api_utils import get_client, run_api_call, fetch_model_info
from frontend.logging_config import configure_logger

logger = configure_logger("safeguard.frontend.ui", "logs/frontend.log")

def apply_custom_css() -> None:
    # Inject logo — always visible in both collapsed and expanded sidebar
    if os.path.exists("frontend/assets/logo.png"):
        st.logo("frontend/assets/logo.png", size="large")
    st.html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
        }

        .stApp {
            background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
            background-attachment: fixed;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        .stButton > button {
            background: rgba(59, 130, 246, 0.05) !important;
            color: #60a5fa !important;
            border: 1px solid rgba(59, 130, 246, 0.2) !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            text-transform: none !important;
            letter-spacing: 0.025em !important;
        }
        .stButton > button:hover {
            background: rgba(59, 130, 246, 0.12) !important;
            border-color: rgba(56, 189, 248, 0.6) !important;
            color: #38bdf8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }
        .stButton > button:active {
            transform: translateY(0px) !important;
        }
        
        .stButton > button[kind="secondary"] {
            background: rgba(248, 250, 252, 0.03) !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            color: #94a3b8 !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background: rgba(248, 250, 252, 0.08) !important;
            border-color: rgba(148, 163, 184, 0.3) !important;
            color: #f1f5f9 !important;
        }

        .hero-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 2rem 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .hero-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(56,189,248,0.1) 0%, transparent 70%);
            z-index: 0;
            pointer-events: none;
        }

        .metric-card {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(56, 189, 248, 0.5);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.15);
            transform: translateY(-2px);
        }

        .section-label {
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: #38bdf8;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        
        .hero-card h2, .hero-card p {
            position: relative;
            z-index: 1;
        }

        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            background: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
        }

        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.5) !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        .streamlit-expanderContent {
            background: rgba(15, 23, 42, 0.4) !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
            border-top: none !important;
            border-bottom-left-radius: 12px !important;
            border-bottom-right-radius: 12px !important;
        }

        .stAlert > div {
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(8px) !important;
        }
        
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        </style>
        """
    )

def render_api_error(err: APIClientError) -> None:
    logger.warning(
        "Rendering API error in UI.",
        extra={
            "event_type": "ui_error",
            "details": {"message": err.message, "errors": err.errors},
        },
    )
    st.error(err.message)
    for item in err.errors:
        st.caption(f"- {item}")

def render_sidebar() -> None:
    """Minimal sidebar — logo only. Auth lives in the top-right popover."""
    with st.sidebar:
        st.caption("Safeguard-AI Lite")
        st.caption("Analyst console for lightweight intrusion detection")
        st.markdown("---")
        client = get_client()
        health, err = run_api_call(client.health)
        if err is None and health:
            badge = "✅ Healthy" if health["status"] == "ok" else "⚠️ Degraded"
            st.caption(f"Backend: {badge}")
        else:
            st.caption("Backend: ❌ Unreachable")

def render_topbar() -> None:
    """Render a right-aligned Sign In / Admin popover in the Streamlit header area."""
    # Build the label shown on the popover button
    if st.session_state.get("auth_token"):
        btn_label = f"👤 {st.session_state['auth_user']}"
    else:
        btn_label = "🔐 Sign In"

    # Place the popover in the far-right column of a header row
    spacer, col_btn = st.columns([10, 1.2])
    with col_btn:
        with st.popover(btn_label, use_container_width=True):
            st.markdown("#### ⚙️ Connection")
            st.session_state["api_base_url"] = st.text_input(
                "API Base URL",
                value=st.session_state.get("api_base_url", "http://127.0.0.1:8000"),
                label_visibility="collapsed",
                placeholder="http://127.0.0.1:8000",
            )

            st.markdown("---")
            if st.session_state.get("auth_token"):
                st.success(f"Signed in as **{st.session_state['auth_user']}**")
                if st.button("Sign Out", use_container_width=True):
                    st.session_state["auth_token"] = None
                    st.session_state["auth_user"] = None
                    st.session_state["model_info_cache"] = None
                    st.rerun()
            else:
                st.markdown("#### 🔐 Authentication")
                username = st.text_input("Username", key="login_username", placeholder="admin")
                password = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")

                client = get_client()
                c1, c2 = st.columns(2)
                if c1.button("Login", use_container_width=True, type="primary"):
                    result, login_err = run_api_call(
                        client.login, username=username, password=password
                    )
                    if login_err is None and result:
                        st.session_state["auth_token"] = result["access_token"]
                        st.session_state["auth_user"] = result["username"]
                        st.rerun()
                    elif login_err:
                        render_api_error(login_err)

                if c2.button("Create Admin", use_container_width=True):
                    st.session_state["show_create_admin"] = not st.session_state.get("show_create_admin", False)

                if st.session_state.get("show_create_admin"):
                    st.markdown("##### Create Admin Account")
                    admin_user = st.text_input("New Username", key="admin_create_user")
                    admin_pass = st.text_input("New Password", type="password", key="admin_create_pass")
                    if st.button("Confirm Creation", use_container_width=True):
                        result, create_err = run_api_call(
                            client.create_admin, username=admin_user, password=admin_pass
                        )
                        if create_err is None and result:
                            st.success(f"Admin **{result['username']}** created.")
                            st.session_state["show_create_admin"] = False
                        elif create_err:
                            render_api_error(create_err)

                st.caption("Protected tabs require a valid bearer token.")

def build_prediction_results_frame(result: dict) -> pd.DataFrame:
    """Flatten prediction payloads into a table that exports cleanly as CSV."""
    rows: list[dict] = []
    for item in result.get("predictions", []):
        row = {
            "row_index": item.get("row_index"),
            "predicted_label": item.get("predicted_label"),
            "predicted_index": item.get("predicted_index"),
            "confidence": item.get("confidence"),
            "recommendation_severity": item.get("recommendation_severity"),
            "recommendations": " | ".join(item.get("recommendations", [])),
        }
        for class_name, probability in (item.get("class_probabilities") or {}).items():
            row[f"prob_{class_name}"] = probability
        rows.append(row)
    return pd.DataFrame(rows)

def render_first_row_explanation(result: dict) -> None:
    """Display the first returned prediction and its local SHAP-style explanation."""
    predictions = result.get("predictions", [])
    if not predictions:
        return

    first = predictions[0]
    st.markdown("**First Row Explanation**")
    st.markdown(f"Predicted label: `{first.get('predicted_label', 'N/A')}`")
    st.markdown(f"Confidence: `{first.get('confidence', 'N/A')}`")
    if first.get("recommendations"):
        st.markdown("**Recommended Actions**")
        for suggestion in first.get("recommendations", []):
            st.write(f"- {suggestion}")

    probabilities = first.get("class_probabilities") or {}
    if probabilities:
        prob_df = pd.DataFrame(
            {
                "class_name": list(probabilities.keys()),
                "probability": list(probabilities.values()),
            }
        ).sort_values(by="probability", ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(prob_df))))
        ax.barh(prob_df["class_name"], prob_df["probability"], color="#f59e0b")
        ax.set_title("First Row Class Probabilities")
        ax.set_xlim(0, 1)
        st.pyplot(fig, use_container_width=True)

    contributions = pd.DataFrame(first.get("top_contributions", []))
    if not contributions.empty:
        chart_df = contributions.sort_values(by="abs_shap_value", ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(chart_df))))
        ax.barh(chart_df["feature"], chart_df["abs_shap_value"], color="#38bdf8")
        ax.set_title("Top Feature Contributions")
        ax.set_xlabel("|SHAP value|")
        st.pyplot(fig, use_container_width=True)
        st.dataframe(contributions, use_container_width=True, hide_index=True)
