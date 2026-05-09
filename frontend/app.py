from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Add project root to path so imports work when running with streamlit
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from frontend.api_client import APIClientError, SafeguardAPIClient
from frontend.logging_config import configure_logger
from frontend.sample_data import ATTACK_PROFILES, generate_live_records

st.set_page_config(
    page_title="Safeguard-AI Lite",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)
logger = configure_logger("safeguard.frontend", "logs/frontend.log")


def init_state() -> None:
    st.session_state.setdefault(
        "api_base_url",
        os.environ.get("SAFEGUARD_API_BASE_URL", "http://127.0.0.1:8000"),
    )
    st.session_state.setdefault("auth_token", None)
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("model_info_cache", None)
    st.session_state.setdefault("latest_prediction_result", None)
    st.session_state.setdefault("latest_upload_result", None)
    st.session_state.setdefault("live_history", [])
    st.session_state.setdefault("live_recent_events", [])
    st.session_state.setdefault("live_alerts", [])
    st.session_state.setdefault("show_create_admin", False)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(
                135deg,
                rgba(17,24,39,0.95),
                rgba(30,41,59,0.95)
            );
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 20px 45px rgba(2, 6, 23, 0.35);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }
        .section-label {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #38bdf8;
            font-size: 0.8rem;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_client() -> SafeguardAPIClient:
    return SafeguardAPIClient(
        base_url=st.session_state["api_base_url"],
        token=st.session_state["auth_token"],
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
    except Exception as exc:  # pragma: no cover - UI safety
        logger.exception(
            "Unexpected frontend exception.", extra={"event_type": "frontend_exception"}
        )
        return None, APIClientError(str(exc))


def fetch_model_info(force: bool = False) -> dict | None:
    if st.session_state["model_info_cache"] is not None and not force:
        return st.session_state["model_info_cache"]
    result, err = run_api_call(get_client().model_info)
    if err is None:
        st.session_state["model_info_cache"] = result
        return result
    return None


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


def serialize_live_history(history: list[dict]) -> list[dict]:
    """Convert session-state live events into a cache-safe payload."""
    serialized: list[dict] = []
    for item in history:
        serialized.append(
            {
                "timestamp": (
                    item["timestamp"].isoformat()
                    if isinstance(item.get("timestamp"), datetime)
                    else str(item.get("timestamp"))
                ),
                "predicted_label": item.get("predicted_label"),
                "confidence": item.get("confidence"),
                "severity": item.get("severity"),
            }
        )
    return serialized


@st.cache_data(show_spinner=False)
def compute_analytics_payload(
    live_history_payload: list[dict], stats_payload: dict | None
) -> dict:
    """Build cached analytics summaries and chart frames."""
    events_df = pd.DataFrame(live_history_payload)
    if not events_df.empty:
        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], errors="coerce")
        events_df = events_df.dropna(subset=["timestamp"])

    total_scans = (
        int(stats_payload["total_predictions"]) if stats_payload else len(events_df)
    )
    if not events_df.empty:
        attack_mask = events_df["predicted_label"].astype(str).str.lower() != "normal"
        attack_count = int(attack_mask.sum())
        percent_attacks = (
            round((attack_count / len(events_df)) * 100, 2) if len(events_df) else 0.0
        )
        attack_counts = (
            events_df.loc[attack_mask, "predicted_label"]
            .value_counts()
            .rename_axis("attack_type")
            .reset_index(name="count")
        )
        daily_trend = (
            events_df.assign(day=lambda df: df["timestamp"].dt.floor("D"))
            .groupby("day")
            .size()
            .reset_index(name="events")
            .sort_values("day")
        )
    else:
        attack_counts = pd.DataFrame(columns=["attack_type", "count"])
        daily_trend = pd.DataFrame(columns=["day", "events"])
        label_counts = (stats_payload or {}).get("predictions_by_label", {})
        normal_count = int(label_counts.get("Normal", 0))
        attack_count = max(total_scans - normal_count, 0)
        percent_attacks = (
            round((attack_count / total_scans) * 100, 2) if total_scans else 0.0
        )
        attack_counts = (
            pd.DataFrame(
                [
                    {"attack_type": key, "count": value}
                    for key, value in label_counts.items()
                    if str(key).lower() != "normal"
                ]
            )
            .sort_values(by="count", ascending=False)
            .reset_index(drop=True)
        )

    top_attack_types = attack_counts.head(3)
    top_attack_summary = (
        ", ".join(
            f"{row.attack_type} ({int(row.count)})"
            for row in top_attack_types.itertuples()
        )
        or "No attack labels yet"
    )

    return {
        "events_df": events_df,
        "attack_counts": attack_counts,
        "daily_trend": daily_trend,
        "total_scans": total_scans,
        "percent_attacks": percent_attacks,
        "top_attack_summary": top_attack_summary,
    }


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


def append_live_events(result: dict) -> None:
    """Persist live events and alerts in session state."""
    for item in result.get("predictions", []):
        event = {
            "timestamp": datetime.utcnow(),
            "predicted_label": item["predicted_label"],
            "confidence": item.get("confidence") or 0.0,
            "severity": item.get(
                "recommendation_severity",
                "Alert" if item["predicted_label"] != "Normal" else "Normal",
            ),
            "recommendations": item.get("recommendations", []),
        }
        st.session_state["live_history"].append(event)
        st.session_state["live_recent_events"].append(event)
        if event["severity"] == "Alert":
            st.session_state["live_alerts"].append(event)

    st.session_state["live_history"] = st.session_state["live_history"][-200:]
    st.session_state["live_recent_events"] = st.session_state["live_recent_events"][
        -30:
    ]
    st.session_state["live_alerts"] = st.session_state["live_alerts"][-20:]


def render_live_dashboard(
    table_placeholder, alerts_placeholder, charts_placeholder
) -> None:
    """Render recent live events, alerts, and compact charts into placeholders."""
    events_df = pd.DataFrame(st.session_state.get("live_recent_events", []))
    alerts_df = pd.DataFrame(st.session_state.get("live_alerts", []))

    with table_placeholder.container():
        st.markdown("**Recent Events**")
        if events_df.empty:
            st.info("No streamed events yet.")
        else:
            display_df = events_df.copy()
            display_df["timestamp"] = display_df["timestamp"].astype(str)
            st.dataframe(
                display_df.sort_values(by="timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with alerts_placeholder.container():
        st.markdown("**Recent Alerts**")
        if alerts_df.empty:
            st.success("No active alerts in the recent event window.")
        else:
            display_df = alerts_df.copy()
            display_df["timestamp"] = display_df["timestamp"].astype(str)
            display_df["recommendations"] = display_df["recommendations"].map(
                lambda items: " | ".join(items) if isinstance(items, list) else items
            )
            st.dataframe(
                display_df.sort_values(by="timestamp", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    with charts_placeholder.container():
        if events_df.empty:
            return

        col1, col2 = st.columns(2)
        with col1:
            label_counts = events_df["predicted_label"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                label_counts.values,
                labels=label_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title("Recent Attack Mix")
            st.pyplot(fig, use_container_width=True)

        with col2:
            timeline_df = events_df.copy()
            timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"])
            timeline = (
                timeline_df.assign(second=lambda df: df["timestamp"].dt.floor("2s"))
                .groupby(["second", "predicted_label"])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )
            st.markdown("**2-Second Event Timeline**")
            st.line_chart(timeline)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Safeguard-AI Lite")
        st.caption("Analyst console for lightweight intrusion detection")

        st.session_state["api_base_url"] = st.text_input(
            "API Base URL", value=st.session_state["api_base_url"]
        )
        client = get_client()
        health, err = run_api_call(client.health)
        if err is None and health:
            badge = "Healthy" if health["status"] == "ok" else "Degraded"
            st.success(f"Backend: {badge}")
        else:
            st.warning("Backend unreachable")

        st.markdown("---")
        st.subheader("Authentication")
        if st.session_state["auth_token"]:
            st.success(f"Signed in as {st.session_state['auth_user']}")
            if st.button("Sign Out", use_container_width=True):
                st.session_state["auth_token"] = None
                st.session_state["auth_user"] = None
                st.session_state["model_info_cache"] = None
                st.rerun()
        else:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            col1, col2 = st.columns(2)
            if col1.button("Login", use_container_width=True):
                logger.info(
                    "Frontend login submitted.",
                    extra={
                        "event_type": "frontend_login_attempt",
                        "username": username,
                    },
                )
                result, login_err = run_api_call(
                    client.login, username=username, password=password
                )
                if login_err is None and result:
                    st.session_state["auth_token"] = result["access_token"]
                    st.session_state["auth_user"] = result["username"]
                    st.success("Login successful.")
                    logger.info(
                        "Frontend login succeeded.",
                        extra={
                            "event_type": "frontend_login_success",
                            "username": username,
                        },
                    )
                    st.rerun()
                elif login_err:
                    render_api_error(login_err)

            if col2.button("Create Admin", use_container_width=True):
                st.session_state["show_create_admin"] = not st.session_state[
                    "show_create_admin"
                ]

            if st.session_state["show_create_admin"]:
                admin_user = st.text_input(
                    "New Admin Username", key="admin_create_user"
                )
                admin_pass = st.text_input(
                    "New Admin Password", type="password", key="admin_create_pass"
                )
                if st.button("Confirm Admin Creation", use_container_width=True):
                    logger.info(
                        "Frontend admin creation submitted.",
                        extra={
                            "event_type": "frontend_create_admin_attempt",
                            "username": admin_user,
                        },
                    )
                    result, create_err = run_api_call(
                        client.create_admin, username=admin_user, password=admin_pass
                    )
                    if create_err is None and result:
                        st.success(f"Admin {result['username']} created.")
                        st.session_state["show_create_admin"] = False
                    elif create_err:
                        render_api_error(create_err)

        st.markdown("---")
        st.caption(
            "Protected tabs require a valid bearer token from the FastAPI backend."
        )


def render_home() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">👋 Welcome to Safeguard-AI Lite!</div>
            <h2 style="margin:0;">Your Personal AI Security Assistant</h2>
            <p style="margin-top:0.6rem;color:#cbd5e1;">
                Think of this software as a very smart security guard for your network. It watches the traffic (data) going in and out of your computer and looks for anything suspicious—like a hacker trying to break in or a virus trying to steal information.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with st.expander("📖 Getting Started Guide (Start Here!)", expanded=True):
        st.markdown("""
        **Don't know anything about cybersecurity? No problem!**
        
        Here is what you can do with this app:
        1. **🎯 Active Scanner (Sidebar):** Type in an IP address or a website, and we will safely check it to see if it has any "open doors" (ports) that hackers could use.
        2. **📡 Live Prediction (Sidebar):** Simulate what happens when a hacker tries to attack your network, and watch the AI catch it in real-time.
        3. **📊 Statistics (Sidebar):** See a summary of all the attacks we've caught.
        
        *Whenever you see a 📖 icon, click it to learn what the information means in plain English!*
        """)

    model_info = fetch_model_info()
    stats_result, _ = (
        run_api_call(get_client().stats)
        if st.session_state["auth_token"]
        else (None, None)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", model_info["model_name"] if model_info else "Unavailable")
    c2.metric("Attack Classes", len(model_info["label_classes"]) if model_info else 0)
    c3.metric("Tracked Features", model_info["feature_count"] if model_info else 0)
    c4.metric(
        "Predictions Logged", stats_result["total_predictions"] if stats_result else 0
    )

    if model_info:
        st.subheader("Operational Snapshot")
        left, right = st.columns([1.2, 1])
        with left:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Label Classes": model_info["label_classes"],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            schema = model_info.get("raw_input_schema", {})
            st.markdown("**Expected Input Shape**")
            st.write(f"Numeric columns: {len(schema.get('numeric_columns', []))}")
            st.write(
                f"Categorical columns: {len(schema.get('categorical_columns', []))}"
            )


def render_upload() -> None:
    st.subheader("Upload Dataset")
    st.caption(
        "Upload CSV logs, preview them, send them to `/predict`, "
        "and export the returned predictions."
    )

    uploaded_file = st.file_uploader("Choose CSV", type=["csv"], key="upload_csv")
    predict_file_name = "uploaded_dataset.csv"

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        preview = pd.read_csv(io.BytesIO(file_bytes))
        st.markdown("**Preview**")
        st.dataframe(preview.head(10), use_container_width=True)
        st.caption(f"Rows: {len(preview)} | Columns: {len(preview.columns)}")

        if st.button("Submit to Prediction API", use_container_width=True):
            logger.info(
                "Frontend upload prediction submitted.",
                extra={
                    "event_type": "frontend_upload_predict",
                    "details": {
                        "file_name": uploaded_file.name,
                        "row_count": len(preview),
                    },
                },
            )
            pred_result, pred_err = run_api_call(
                get_client().predict_csv,
                file_name=uploaded_file.name or predict_file_name,
                file_bytes=file_bytes,
            )
            if pred_err:
                render_api_error(pred_err)
            else:
                st.session_state["latest_prediction_result"] = pred_result
                logger.info(
                    "Frontend prediction result received.",
                    extra={
                        "event_type": "frontend_predict_success",
                        "details": {
                            "labels": pred_result.get("summary", {}).get("labels", {})
                        },
                    },
                )
                st.success("Prediction complete.")

        result = st.session_state.get("latest_prediction_result")
        if result:
            st.markdown("**Prediction Results**")
            results_df = build_prediction_results_frame(result)
            if not results_df.empty:
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_bytes,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if result.get("summary", {}).get("recommended_actions"):
                st.markdown("**Batch Recommendations**")
                for suggestion in result["summary"]["recommended_actions"]:
                    st.write(f"- {suggestion}")
            render_first_row_explanation(result)


def render_live_predictions() -> None:
    st.subheader("Live Prediction Simulation")
    model_info = fetch_model_info()
    if not model_info:
        st.info("Sign in and connect to the backend to load the model schema.")
        return

    schema = model_info["raw_input_schema"]
    attack_label = st.selectbox(
        "Simulation profile", options=list(ATTACK_PROFILES.keys()), index=1
    )
    batch_size = st.slider("Batch size", min_value=1, max_value=25, value=1)
    explanation_top_k = st.slider(
        "Explanation depth", min_value=1, max_value=10, value=5
    )
    stream_cycles = st.slider("Stream iterations", min_value=3, max_value=30, value=8)

    metrics_row = st.columns(4)
    metrics_row[0].metric(
        "Recent Events", len(st.session_state.get("live_recent_events", []))
    )
    metrics_row[1].metric("Recent Alerts", len(st.session_state.get("live_alerts", [])))
    metrics_row[2].metric("Profile", attack_label)
    metrics_row[3].metric("Cadence", "2 sec")

    status_placeholder = st.empty()
    table_placeholder = st.empty()
    alerts_placeholder = st.empty()
    charts_placeholder = st.empty()

    if st.button("Run Simulation", use_container_width=True):
        logger.info(
            "Frontend live simulation requested.",
            extra={
                "event_type": "frontend_live_simulation",
                "details": {"profile": attack_label, "batch_size": batch_size},
            },
        )
        records = generate_live_records(
            schema=schema, attack_label=attack_label, count=batch_size
        )
        result, err = run_api_call(
            get_client().predict_records,
            records=records,
            include_explanations=True,
            explanation_top_k=explanation_top_k,
        )
        if err:
            render_api_error(err)
        else:
            st.session_state["latest_prediction_result"] = result
            append_live_events(result)
            logger.info(
                "Frontend simulation completed.",
                extra={
                    "event_type": "frontend_live_simulation_success",
                    "details": {"labels": result["summary"]["labels"]},
                },
            )
            st.success(
                f"Generated {result['summary']['prediction_count']} live predictions."
            )

    if st.button("Start Fake Live Feed", use_container_width=True):
        logger.info(
            "Frontend fake live feed started.",
            extra={
                "event_type": "frontend_live_feed_start",
                "details": {"profile": attack_label, "cycles": stream_cycles},
            },
        )
        status_placeholder.info("Streaming fake traffic to the classifier...")
        for step in range(stream_cycles):
            records = generate_live_records(
                schema=schema, attack_label=attack_label, count=batch_size
            )
            result, err = run_api_call(
                get_client().predict_records,
                records=records,
                include_explanations=True,
                explanation_top_k=explanation_top_k,
            )
            if err:
                render_api_error(err)
                status_placeholder.error(
                    "Live feed stopped because the API returned an error."
                )
                break

            st.session_state["latest_prediction_result"] = result
            append_live_events(result)
            if any(
                item.get("predicted_label") != "Normal"
                for item in result.get("predictions", [])
            ):
                logger.warning(
                    "Frontend live feed detected an attack.",
                    extra={
                        "event_type": "frontend_attack_detected",
                        "details": {
                            "labels": result["summary"]["labels"],
                            "tick": step + 1,
                        },
                    },
                )
            status_placeholder.success(
                f"Stream tick {step + 1}/{stream_cycles}: classified "
                f"{result['summary']['prediction_count']} event(s)."
            )
            render_live_dashboard(
                table_placeholder, alerts_placeholder, charts_placeholder
            )
            time.sleep(2)
        else:
            logger.info(
                "Frontend fake live feed completed.",
                extra={"event_type": "frontend_live_feed_complete"},
            )
            status_placeholder.success("Fake live feed completed.")

    render_live_dashboard(table_placeholder, alerts_placeholder, charts_placeholder)

    if st.session_state["latest_prediction_result"]:
        pred_df = pd.DataFrame(
            st.session_state["latest_prediction_result"]["predictions"]
        )
        if not pred_df.empty:
            st.markdown("**Latest API Response**")
            columns = [
                col
                for col in [
                    "row_index",
                    "predicted_label",
                    "confidence",
                    "recommendation_severity",
                    "recommendations",
                ]
                if col in pred_df.columns
            ]
            display_df = pred_df[columns].copy()
            if "recommendations" in display_df.columns:
                display_df["recommendations"] = display_df["recommendations"].map(
                    lambda items: (
                        " | ".join(items) if isinstance(items, list) else items
                    )
                )
            st.dataframe(display_df, use_container_width=True)


def render_statistics() -> None:
    st.subheader("Statistics")
    result, err = run_api_call(get_client().stats)
    if err:
        render_api_error(err)
        return

    top = st.columns(4)
    top[0].metric("Total Predictions", result["total_predictions"])
    top[1].metric("Total Uploads", result["total_uploads"])
    top[2].metric("Average Confidence", result["avg_confidence"])
    top[3].metric("Latest Prediction", str(result["latest_prediction_at"] or "N/A"))

    labels = result.get("predictions_by_label", {})
    if labels:
        pie_df = pd.DataFrame(
            {"label": list(labels.keys()), "count": list(labels.values())}
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            pie_df["count"], labels=pie_df["label"], autopct="%1.1f%%", startangle=90
        )
        ax.set_title("Attack Type Distribution")
        st.pyplot(fig, use_container_width=True)

    history = st.session_state.get("live_history", [])
    if history:
        history_df = pd.DataFrame(history)
        timeline = (
            history_df.assign(minute=lambda df: df["timestamp"].dt.floor("min"))
            .groupby(["minute", "predicted_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        st.markdown("**Live Prediction Timeline**")
        st.line_chart(timeline)


def render_analytics() -> None:
    st.subheader("Analytics")
    st.caption("Cached operational rollups for attack mix and event trends.")

    stats_result, err = run_api_call(get_client().stats)
    if err:
        render_api_error(err)
        return

    analytics = compute_analytics_payload(
        serialize_live_history(st.session_state.get("live_history", [])),
        stats_result,
    )

    top = st.columns(4)
    top[0].metric("Total Scans", analytics["total_scans"])
    top[1].metric("% Attacks", f"{analytics['percent_attacks']}%")
    top[2].metric("Top Attack Types", analytics["top_attack_summary"])
    latest_day = (
        analytics["daily_trend"]["day"].max().strftime("%Y-%m-%d")
        if not analytics["daily_trend"].empty
        else "N/A"
    )
    top[3].metric("Daily Trend", latest_day)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Attack Type Breakdown**")
        attack_counts = analytics["attack_counts"]
        if attack_counts.empty:
            st.info("Attack breakdown will appear once events are classified.")
        else:
            plot_df = attack_counts.head(8).sort_values(by="count", ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(plot_df))))
            ax.barh(plot_df["attack_type"], plot_df["count"], color="#38bdf8")
            ax.set_title("Top Attack Types")
            ax.set_xlabel("Count")
            st.pyplot(fig, use_container_width=True)

    with chart_right:
        st.markdown("**Daily Event Trend**")
        daily_trend = analytics["daily_trend"]
        if daily_trend.empty:
            st.info(
                "Daily trend will appear once live or uploaded predictions accumulate."
            )
        else:
            trend_df = daily_trend.set_index("day")
            st.line_chart(trend_df)

    if not analytics["events_df"].empty:
        st.markdown("**Recent Event Analytics Feed**")
        recent_df = (
            analytics["events_df"]
            .sort_values(by="timestamp", ascending=False)
            .head(25)
            .copy()
        )
        recent_df["timestamp"] = recent_df["timestamp"].astype(str)
        st.dataframe(recent_df, use_container_width=True, hide_index=True)


def render_explanations() -> None:
    st.subheader("Explainability")
    result = st.session_state.get("latest_prediction_result")
    if not result:
        st.info(
            "Run an upload prediction or live simulation first "
            "to populate explanations."
        )
        return

    summary = result.get("summary", {})
    importance = summary.get("global_feature_importance", [])
    if importance:
        imp_df = pd.DataFrame(importance).sort_values(
            by="mean_abs_shap", ascending=True
        )
        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(imp_df))))
        ax.barh(imp_df["feature"], imp_df["mean_abs_shap"], color="#38bdf8")
        ax.set_title("Global Feature Importance")
        ax.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig, use_container_width=True)
        st.dataframe(
            pd.DataFrame(importance), use_container_width=True, hide_index=True
        )

    predictions = result.get("predictions", [])
    if predictions:
        row_options = [item["row_index"] for item in predictions]
        selected_row = st.selectbox("Inspect row", options=row_options, index=0)
        selected = next(
            item for item in predictions if item["row_index"] == selected_row
        )
        st.markdown(f"**Predicted label:** `{selected['predicted_label']}`")
        st.markdown(f"**Confidence:** `{selected.get('confidence')}`")
        if selected.get("recommendations"):
            st.markdown("**Recommendations**")
            for suggestion in selected.get("recommendations", []):
                st.write(f"- {suggestion}")

        if selected.get("class_probabilities"):
            prob_df = pd.DataFrame(
                {
                    "class_name": list(selected["class_probabilities"].keys()),
                    "probability": list(selected["class_probabilities"].values()),
                }
            ).sort_values(by="probability", ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(prob_df))))
            ax.barh(prob_df["class_name"], prob_df["probability"], color="#f59e0b")
            ax.set_title("Class Probabilities")
            ax.set_xlim(0, 1)
            st.pyplot(fig, use_container_width=True)

        contributions = pd.DataFrame(selected.get("top_contributions", []))
        if not contributions.empty:
            st.markdown("**Top Local Contributions**")
            st.dataframe(contributions, use_container_width=True, hide_index=True)


def render_soc_dashboard() -> None:
    st.subheader("SOC Operations Dashboard")
    st.caption(
        "Real-time alert stream, network event feed, analyst notifications, and acknowledgement workflow."
    )

    api_base_url = st.session_state["api_base_url"]
    token = st.session_state.get("auth_token") or ""
    if not token:
        st.warning("Sign in to enable acknowledgement actions and receive secure analyst notifications.")

    html = """
    <div style="font-family: 'Segoe UI', sans-serif; color: #e2e8f0; background:#0f172a; padding:18px; border-radius:18px;">
      <style>
        .soc-box { background:#111827; border:1px solid rgba(59,130,246,0.24); border-radius:14px; padding:14px; margin-bottom:12px; }
        .soc-title { font-size:1.3rem; color:#38bdf8; margin-bottom:6px; }
        .soc-count { font-size:2.2rem; color:#f8fafc; margin:4px 0; }
        .soc-label { color:#94a3b8; margin-bottom:8px; }
        .soc-list { list-style:none; padding:0; margin:0; }
        .soc-list li { border-bottom:1px solid rgba(148,163,184,0.12); padding:10px 0; }
        .soc-button { background:#2563eb; color:#fff; border:none; padding:8px 12px; border-radius:8px; cursor:pointer; }
        .soc-button.disabled { background:#475569; cursor:not-allowed; }
        .soc-notice { background: rgba(14, 165, 233, 0.08); border:1px solid rgba(14,165,233,0.24); padding:12px; border-radius:12px; margin-bottom:10px; }
      </style>
      <div class="soc-box">
        <div class="soc-title">Connection</div>
        <div class="soc-label">WebSocket channel:</div>
        <div id="ws-status" class="soc-count">Connecting…</div>
      </div>
      <div class="soc-box" id="metrics-panel">
        <div class="soc-title">Operational Metrics</div>
        <div class="soc-label">Alerts, attack cadence, and event velocity updated live.</div>
        <div style="display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px;">
          <div style="background:#0f172a; border-radius:12px; padding:10px;"><div class="soc-label">Alerts</div><div id="metric-alerts" class="soc-count">0</div></div>
          <div style="background:#0f172a; border-radius:12px; padding:10px;"><div class="soc-label">Detections</div><div id="metric-detections" class="soc-count">0</div></div>
          <div style="background:#0f172a; border-radius:12px; padding:10px;"><div class="soc-label">Notifications</div><div id="metric-notifications" class="soc-count">0</div></div>
          <div style="background:#0f172a; border-radius:12px; padding:10px;"><div class="soc-label">Logs</div><div id="metric-logs" class="soc-count">0</div></div>
        </div>
      </div>
      <div class="soc-box" id="timeline-panel">
        <div class="soc-title">Alert Timeline</div>
        <ul id="timeline-list" class="soc-list"></ul>
      </div>
      <div class="soc-box" id="alert-panel">
        <div class="soc-title">Live Alert Feed</div>
        <ul id="alert-list" class="soc-list"></ul>
      </div>
      <div class="soc-box" id="notification-panel">
        <div class="soc-title">Notification Center</div>
        <ul id="notification-list" class="soc-list"></ul>
      </div>
      <div class="soc-box" id="log-panel">
        <div class="soc-title">Streaming Logs</div>
        <ul id="log-list" class="soc-list"></ul>
      </div>
      <audio id="alert-sound" src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YSoAAAAA"></audio>
    </div>
    <script>
      const baseUrl = "{{API_BASE_URL}}";
      const token = "{{TOKEN}}";
      const wsUrl = baseUrl.replace(/^http/, "ws") + "/ws/realtime?channels=alerts,traffic,notifications,logs";
      const socket = new WebSocket(wsUrl);
      const alerts = [];
      const timeline = [];
      const notifications = [];
      const logs = [];
      let alertCount = 0;
      let detectionCount = 0;
      let notificationCount = 0;
      let logCount = 0;

      function renderState() {
        document.getElementById("metric-alerts").textContent = alertCount;
        document.getElementById("metric-detections").textContent = detectionCount;
        document.getElementById("metric-notifications").textContent = notificationCount;
        document.getElementById("metric-logs").textContent = logCount;
        const alertList = document.getElementById("alert-list");
        alertList.innerHTML = alerts.slice(0, 8).map(item => `
          <li>
            <strong>[${item.severity.toUpperCase()}]</strong> ${item.description} <br/>
            <small>${item.timestamp} • ${item.threat_type} • ${item.src_ip}</small><br/>
            <button class="soc-button" onclick="ackAlert(${item.id})">Acknowledge</button>
          </li>
        `).join("");
        const timelineList = document.getElementById("timeline-list");
        timelineList.innerHTML = timeline.slice(0, 8).map(item => `
          <li><strong>${item.type}</strong> ${item.summary}<br/><small>${item.timestamp}</small></li>
        `).join("");
        const notificationList = document.getElementById("notification-list");
        notificationList.innerHTML = notifications.slice(0, 6).map(item => `
          <li><strong>${item.level}</strong> ${item.message}<br/><small>${item.timestamp}</small></li>
        `).join("");
        const logList = document.getElementById("log-list");
        logList.innerHTML = logs.slice(0, 8).map(item => `
          <li>${item.timestamp} • ${item.level.toUpperCase()} • ${item.message}</li>
        `).join("");
      }

      function showDesktopNotification(payload) {
        if (!window.Notification) return;
        if (Notification.permission === "default") {
          Notification.requestPermission();
        }
        if (Notification.permission === "granted") {
          new Notification(payload.title || "SOC Notification", {
            body: payload.message,
            icon: "https://api.iconify.design/mdi/shield-alert.svg?color=%2338bdf8",
          });
        }
        document.getElementById("alert-sound").play().catch(() => {});
      }

      async function ackAlert(alertId) {
        if (!token) {
          alert("Sign in to acknowledge alerts.");
          return;
        }
        try {
          await fetch(`${baseUrl}/api/v1/alerts/${alertId}/acknowledge`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,
            },
            body: JSON.stringify({
              acknowledged_by: "analyst",
              comment: "Acknowledged via SOC dashboard",
            }),
          });
          const ackTimeline = {
            type: "acknowledgement",
            summary: `Alert ${alertId} acknowledged`,
            timestamp: new Date().toISOString(),
          };
          timeline.unshift(ackTimeline);
          renderState();
        } catch (err) {
          console.error(err);
        }
      }

      socket.addEventListener("open", () => {
        document.getElementById("ws-status").textContent = "Connected";
      });
      socket.addEventListener("close", () => {
        document.getElementById("ws-status").textContent = "Disconnected";
      });
      socket.addEventListener("message", event => {
        try {
          const message = JSON.parse(event.data);
          const payload = message.payload || {};
          const timestamp = payload.timestamp || new Date().toISOString();
          if (message.type === "alert") {
            alertCount += 1;
            timeline.unshift({ type: "alert", summary: payload.description, timestamp });
            alerts.unshift(payload);
            showDesktopNotification({ title: "New SOC Alert", message: payload.description });
          }
          if (message.type === "traffic") {
            detectionCount += 1;
            timeline.unshift({ type: "traffic", summary: payload.description, timestamp });
          }
          if (message.type === "notification") {
            notificationCount += 1;
            notifications.unshift(payload);
            showDesktopNotification(payload);
          }
          if (message.type === "log") {
            logCount += 1;
            logs.unshift(payload);
          }
          if (message.type === "alert_ack") {
            timeline.unshift({ type: "ack", summary: `Alert ${payload.id} acknowledged`, timestamp });
          }
          renderState();
        } catch (err) {
          console.error(err);
        }
      });

      renderState();
    </script>
    """
    html = html.replace("{{API_BASE_URL}}", api_base_url).replace("{{TOKEN}}", token)
    components.html(html, height=980, scrolling=True)


def render_soc_assistant() -> None:
    st.subheader("SOC Analyst Assistant")
    st.caption(
        "Generate analyst-readable threat summaries, incident timelines, remediation guidance, and SHAP explanations via Groq AI."
    )

    client = get_client()
    latest_result = st.session_state.get("latest_prediction_result")
    manual_source = "{}"
    if latest_result and latest_result.get("predictions"):
        first_prediction = latest_result["predictions"][0]
        manual_source = json.dumps(
            {
                "alert_id": None,
                "packet_metadata": {"source": "live_prediction"},
                "detection_result": first_prediction,
                "threat_intelligence": [],
                "shap_explanations": first_prediction.get("shap_values", {}),
                "historical_events": [],
                "system_metrics": {},
                "analyst_notes": "Use the latest prediction context to generate a concise incident briefing.",
            },
            indent=2,
        )

    input_json = st.text_area(
        "SOC context payload (JSON)",
        value=manual_source,
        height=280,
        help="Provide structured packet, detection, threat intelligence, SHAP and system context for the analyst assistant.",
    )

    if st.button("Run SOC Analysis", use_container_width=True):
        try:
            payload = json.loads(input_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON payload: {exc}")
            return

        result, err = run_api_call(client.analyze_soc, payload)
        if err:
            render_api_error(err)
            return

        st.markdown("### Threat Summary")
        st.write(result.get("threat_summary", "No summary returned."))

        st.markdown("### Risk Assessment")
        st.write(result.get("risk_assessment", "No risk assessment returned."))

        st.markdown("### Remediation Recommendations")
        for suggestion in result.get("remediation_recommendations", []):
            st.write(f"- {suggestion}")

        st.markdown("### Incident Timeline")
        for event in result.get("incident_timeline", []):
            st.write(f"- {event.get('timestamp', 'unknown')}: {event.get('event', event)}")
            if event.get("detail"):
                st.caption(event["detail"])

        st.markdown("### False Positive Analysis")
        st.write(result.get("false_positive_analysis", "Not available."))

        st.markdown("### Correlated Events")
        for event in result.get("correlated_events", []):
            st.write(
                f"- {event.get('event_id', 'unnamed')}: {event.get('correlation_reason', event)}"
            )

        st.markdown("### SHAP Explanation")
        st.write(result.get("shap_explanation", "Not available."))

        st.markdown("### Incident Report")
        st.write(result.get("incident_report", "No incident report returned."))

        if result.get("raw_response"):
            with st.expander("Raw AI response"):
                st.json(result.get("raw_response"))


def render_about() -> None:
    st.subheader("About")
    st.write("""
        Safeguard-AI Lite is a lightweight intrusion-detection workspace built around:

        - `FastAPI` for authenticated inference and logging
        - `scikit-learn` / optional gradient-boosting models for classification
        - `SHAP` for explainability
        - `SQLite` for lightweight operational telemetry
        - `Streamlit` for rapid analyst-facing workflows
        """)

    st.markdown("**Recommended run flow**")
    st.code(
        "uvicorn backend.api.main:app --reload\nstreamlit run frontend/app.py",
        language="bash",
    )


def main() -> None:
    init_state()
    apply_custom_css()
    render_sidebar()

    tabs = st.tabs(
        [
            "Home",
            "Upload",
            "Live Predictions",
            "Statistics",
            "Analytics",
            "Explanations",
            "SOC Dashboard",
            "SOC Assistant",
            "About",
        ]
    )
    with tabs[0]:
        render_home()
    with tabs[1]:
        render_upload()
    with tabs[2]:
        render_live_predictions()
    with tabs[3]:
        render_statistics()
    with tabs[4]:
        render_analytics()
    with tabs[5]:
        render_explanations()
    with tabs[6]:
        render_soc_dashboard()
    with tabs[7]:
        render_soc_assistant()
    with tabs[8]:
        render_about()


if __name__ == "__main__":
    main()
