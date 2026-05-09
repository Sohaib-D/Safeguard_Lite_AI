from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.logging_config import configure_logger
from backend.dependencies.auth import get_current_user, get_optional_current_user
from backend.core.security import configure_security
from backend.schemas.alert import AlertAcknowledgementRequest, DetectionAlert
from backend.schemas.auth import (
    CreateAdminRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
)
from backend.schemas.common import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    StatsResponse,
    UploadResponse,
)
from backend.schemas.predict import PredictionRequest, PredictionResponse, UploadRequest
from backend.schemas.response import (
    ResponseActionApproval,
    ResponseActionItem,
    ResponseActionRequest,
    RollbackRequest,
)
from backend.schemas.groq import GroqAssistantRequest, GroqAssistantResponse
from backend.services.alert_service import AlertService
from backend.services.auth_service import AuthService, AuthenticationError
from backend.services.detection_engine import DetectionEngine
from backend.services.groq_service import GroqAssistant
from backend.services.log_service import LogService
from backend.services.model_service import ModelService
from backend.services.response_engine import ResponseEngine
from backend.services.rule_parser import RuleParser
from backend.services.validation_service import InputValidationError
from backend.services.websocket_manager import WebSocketManager
from backend.services.packet_capture import PacketCaptureService
from backend.utils.sanitization import sanitize_filename, strip_suspicious_text

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)
configure_security(app)
logger = configure_logger("safeguard.backend.api", settings.log_file_path)

log_service = LogService(settings.database_path)
model_service = ModelService(
    settings.model_bundle_path, max_explanations=settings.max_explanations
)
auth_service = AuthService(settings.database_path)
ws_manager = WebSocketManager()
alert_service = AlertService(settings.database_path, websocket_manager=ws_manager)
rule_parser = RuleParser(settings.rules_path)

async def _publish_event(event_type: str, payload: dict[str, Any]) -> None:
    try:
        await ws_manager.broadcast(event_type, payload)
    except Exception as exc:
        logger.warning(
            f"Failed to publish websocket event {event_type}: {exc}",
            extra={"event_type": "websocket_error", "details": str(exc)},
        )

async def _publish_log_event(level: str, message: str, details: dict[str, Any] | None = None) -> None:
    await _publish_event(
        "log",
        {
            "level": level,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
detection_engine = DetectionEngine(
    alert_service=alert_service,
    rule_parser=rule_parser,
    model_service=model_service,
    shap_explainer=None,
)
packet_service = PacketCaptureService(detector=detection_engine)
response_engine = ResponseEngine(settings.database_path)
groq_assistant = GroqAssistant(
    api_url=settings.groq_api_url,
    api_key=settings.groq_api_key,
    model=settings.groq_model,
    rate_limit_per_minute=settings.groq_rate_limit_per_minute,
    max_retries=settings.groq_max_retries,
    cache_dir=settings.groq_cache_dir,
    cache_ttl_seconds=settings.groq_cache_ttl,
)

# Add detection callback for additional logging
async def detection_callback(detection):
    src_ip = detection.event_context.get("src_ip", "unknown") if detection.event_context else "unknown"
    payload = {
        "threat_type": detection.threat_type,
        "confidence": detection.confidence,
        "description": detection.description,
        "src_ip": src_ip,
        "severity": detection.severity,
        "alert_type": detection.alert_type,
        "score": detection.score,
        "timestamp": datetime.utcnow().isoformat(),
    }
    logger.warning(
        f"Threat detected: {detection.threat_type} - {detection.description}",
        extra={"event_type": "detection_event", "details": payload},
    )
    await _publish_event("traffic", payload)
    await _publish_event(
        "notification",
        {
            "level": "warning",
            "title": "Threat detected",
            "message": detection.description,
            "context": payload,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

packet_service.add_detection_callback(detection_callback)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning(
        "Request validation failed.",
        extra={
            "event_type": "validation_error",
            "details": {"path": str(request.url.path), "errors": exc.errors()},
        },
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail="Input validation failed.",
            error_code="validation_error",
            errors=[str(err.get("msg", "Invalid input.")) for err in exc.errors()],
        ).model_dump(mode="json"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "HTTP exception raised.",
        extra={
            "event_type": "http_error",
            "details": {
                "path": str(request.url.path),
                "status_code": exc.status_code,
                "detail": str(exc.detail),
            },
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            error_code="http_error",
        ).model_dump(mode="json"),
    )


@app.exception_handler(AuthenticationError)
async def authentication_exception_handler(
    request: Request, exc: AuthenticationError
) -> JSONResponse:
    logger.warning(
        "Authentication error raised.",
        extra={
            "event_type": "authentication_error",
            "details": {"path": str(request.url.path), "detail": exc.message},
        },
    )
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=ErrorResponse(
            detail=exc.message,
            error_code="authentication_error",
        ).model_dump(mode="json"),
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(InputValidationError)
async def input_validation_exception_handler(
    request: Request, exc: InputValidationError
) -> JSONResponse:
    logger.warning(
        "Input validation error raised.",
        extra={
            "event_type": "input_validation_error",
            "details": {"path": str(request.url.path), "errors": exc.errors},
        },
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            detail=exc.message,
            error_code="input_validation_error",
            errors=exc.errors,
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled server exception.",
        extra={
            "event_type": "unhandled_exception",
            "details": {"path": str(request.url.path)},
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="Internal server error.",
            error_code="internal_error",
        ).model_dump(mode="json"),
    )


def _json_records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="No records supplied.")
    return pd.DataFrame(records)


async def _upload_to_frame(file: UploadFile) -> pd.DataFrame:
    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Uploaded file is missing a filename."
        )
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV: {exc}"
        ) from exc


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "message": "Safeguard-AI Lite API is running.",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    logger.info("Health endpoint checked.", extra={"event_type": "health_check"})
    return HealthResponse(
        status="ok" if model_service.ping() and log_service.ping() else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=model_service.ping(),
        database_ok=log_service.ping(),
    )


@app.post("/auth/create-admin", response_model=UserResponse)
async def create_admin(
    payload: CreateAdminRequest,
    current_user: dict | None = Depends(get_optional_current_user),
) -> UserResponse:
    logger.info(
        "Create-admin endpoint called.",
        extra={"event_type": "create_admin_attempt", "username": payload.username},
    )
    if auth_service.admin_exists():
        if current_user is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to create additional admin users.",
            )
        if not current_user.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin privileges required.")
    username = strip_suspicious_text(payload.username, max_length=64)
    if username != payload.username:
        raise HTTPException(
            status_code=400, detail="Username contains disallowed characters."
        )
    user = auth_service.create_admin_user(username=username, password=payload.password)
    log_service.log_user_activity(
        action="create_admin",
        username=user["username"],
        details={"created_admin": user["username"]},
    )
    return UserResponse(**user)


@app.post("/auth/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    logger.info(
        "Login endpoint called.",
        extra={"event_type": "login_attempt", "username": payload.username},
    )
    username = strip_suspicious_text(payload.username, max_length=64)
    if username != payload.username:
        raise HTTPException(
            status_code=400, detail="Username contains disallowed characters."
        )
    user = auth_service.authenticate_user(username=username, password=payload.password)
    token, expires_in = auth_service.create_access_token(user)
    log_service.log_user_activity(
        action="login",
        username=user["username"],
        details={"is_admin": user["is_admin"]},
    )
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        username=user["username"],
        is_admin=bool(user["is_admin"]),
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info(
    current_user: dict = Depends(get_current_user),
) -> ModelInfoResponse:
    logger.info(
        "Model info retrieved.",
        extra={"event_type": "model_info", "username": current_user.get("username")},
    )
    return ModelInfoResponse(**model_service.get_model_info())


@app.get("/stats", response_model=StatsResponse)
async def stats(current_user: dict = Depends(get_current_user)) -> StatsResponse:
    logger.info(
        "Stats retrieved.",
        extra={"event_type": "stats_view", "username": current_user.get("username")},
    )
    return StatsResponse(**log_service.get_stats())


@app.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request,
    file: UploadFile | None = File(default=None),
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    logger.info(
        "Upload endpoint called.",
        extra={
            "event_type": "upload_attempt",
            "username": current_user.get("username"),
        },
    )
    source_type = "csv"
    if file is not None:
        df = await _upload_to_frame(file)
        df = model_service.sanitize_and_validate(df)
        records = df.to_dict(orient="records")
        source_name = sanitize_filename(file.filename or "uploaded.csv")
    else:
        try:
            payload = UploadRequest.model_validate(await request.json())
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Expected a CSV file or JSON payload with 'records'.",
            ) from exc
        records = model_service.sanitize_and_validate(
            _json_records_to_frame(payload.records)
        ).to_dict(orient="records")
        source_name = payload.source_name
        source_type = "json"

    if len(records) > settings.max_upload_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Upload exceeds max row limit of {settings.max_upload_rows}.",
        )

    upload_id = log_service.log_upload(
        source_name=source_name,
        source_type=source_type,
        records=records,
        username=current_user.get("username"),
    )
    logger.info(
        "Upload logged successfully.",
        extra={
            "event_type": "upload_success",
            "username": current_user.get("username"),
            "source_type": source_type,
            "details": {"row_count": len(records), "source_name": source_name},
        },
    )
    return UploadResponse(
        upload_id=upload_id,
        rows_logged=len(records),
        source_type=source_type,
        timestamp=datetime.utcnow(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None),
    current_user: dict = Depends(get_current_user),
) -> PredictionResponse:
    logger.info(
        "Predict endpoint called.",
        extra={
            "event_type": "predict_attempt",
            "username": current_user.get("username"),
        },
    )
    source_type = "csv"
    include_explanations = True
    explanation_top_k = 5

    if file is not None:
        df = await _upload_to_frame(file)
        df = model_service.sanitize_and_validate(df)
    else:
        try:
            payload = PredictionRequest.model_validate(await request.json())
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Expected a CSV file or JSON payload with 'records'.",
            ) from exc
        df = model_service.sanitize_and_validate(
            _json_records_to_frame(payload.records)
        )
        include_explanations = payload.include_explanations
        explanation_top_k = payload.explanation_top_k
        source_type = "json"

    try:
        result = model_service.predict(
            raw_df=df,
            include_explanations=include_explanations,
            explanation_top_k=explanation_top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {exc}"
        ) from exc

    log_service.log_predictions(
        result["predictions"],
        source_type=source_type,
        username=current_user.get("username"),
    )
    logger.info(
        "Prediction response returned.",
        extra={
            "event_type": "predict_success",
            "username": current_user.get("username"),
            "source_type": source_type,
            "details": {
                "row_count": len(result["predictions"]),
                "labels": result["summary"]["labels"],
            },
        },
    )
    return PredictionResponse(**result)


# Packet Capture Endpoints

@app.post("/api/v1/capture/start")
async def start_capture(
    interface: str = None,
    current_user: dict = Depends(get_current_user)
):
    """Start packet capture on specified interface"""
    try:
        await packet_service.start_monitoring(interface)
        logger.info(
            "Packet capture started.",
            extra={"event_type": "capture_start", "username": current_user.get("username"), "interface": interface}
        )
        return {"status": "started", "interface": packet_service.sniffer.interface}
    except Exception as e:
        logger.error(f"Failed to start capture: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start capture: {e}")


@app.post("/api/v1/capture/stop")
async def stop_capture(current_user: dict = Depends(get_current_user)):
    """Stop packet capture"""
    try:
        await packet_service.stop_monitoring()
        logger.info(
            "Packet capture stopped.",
            extra={"event_type": "capture_stop", "username": current_user.get("username")}
        )
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Failed to stop capture: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop capture: {e}")


@app.get("/api/v1/capture/stats")
async def get_capture_stats(current_user: dict = Depends(get_current_user)):
    """Get packet capture statistics"""
    try:
        stats = await packet_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@app.get("/api/v1/alerts")
async def list_detection_alerts(current_user: dict = Depends(get_current_user)):
    try:
        alerts = await alert_service.list_alerts(limit=100)
        await _publish_log_event(
            "info",
            "Listed detection alerts.",
            {"username": current_user.get("username"), "count": len(alerts)},
        )
        return [alert.model_dump() for alert in alerts]
    except Exception as e:
        logger.error(f"Failed to list detection alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list detection alerts: {e}")


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_detection_alert(
    alert_id: int,
    payload: AlertAcknowledgementRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        acknowledgement = await alert_service.acknowledge_alert(alert_id, payload)
        await _publish_log_event(
            "info",
            "Alert acknowledgement submitted.",
            {"username": current_user.get("username"), "alert_id": alert_id},
        )
        return {
            "status": "acknowledged",
            "alert": acknowledgement.model_dump(),
        }
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {e}")


@app.get("/api/v1/rules")
async def list_rules(current_user: dict = Depends(get_current_user)):
    try:
        return {"rules": rule_parser.rules}
    except Exception as e:
        logger.error(f"Failed to list rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list rules: {e}")


@app.post("/api/v1/rules/reload")
async def reload_rules(current_user: dict = Depends(get_current_user)):
    try:
        rule_parser.load_rules()
        return {"status": "reloaded", "rule_count": len(rule_parser.rules)}
    except Exception as e:
        logger.error(f"Failed to reload rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload rules: {e}")


@app.post("/api/v1/soc/analyze", response_model=GroqAssistantResponse)
async def analyze_soc_context(
    payload: GroqAssistantRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        result = await groq_assistant.analyze(payload)
        logger.info(
            "SOC analysis completed.",
            extra={
                "event_type": "soc_analysis",
                "username": current_user.get("username"),
                "alert_id": payload.alert_id,
            },
        )
        return result
    except Exception as e:
        logger.error(f"Failed to run SOC analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run SOC analysis: {e}")


@app.post("/api/v1/response/request")
async def request_response_action(
    payload: ResponseActionRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        action_id = response_engine.propose_action(payload)
        logger.info(
            "Response action requested.",
            extra={
                "event_type": "response_requested",
                "username": current_user.get("username"),
                "action_type": payload.action_type,
                "target": payload.target,
            },
        )
        return {"action_id": action_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Failed to queue response action: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue response action: {e}")


@app.get("/api/v1/response/pending")
async def get_pending_response_actions(
    current_user: dict = Depends(get_current_user),
):
    try:
        actions = response_engine.get_pending_actions()
        return [action.model_dump(mode="json") for action in actions]
    except Exception as e:
        logger.error(f"Failed to retrieve pending actions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve pending actions: {e}")


@app.post("/api/v1/response/approve")
async def approve_response_action(
    payload: ResponseActionApproval,
    current_user: dict = Depends(get_current_user),
):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin privileges required to approve response actions.")
    try:
        result = response_engine.approve_action(
            action_id=payload.action_id,
            approved_by=payload.approved_by,
            approved=payload.approved,
            justification=payload.justification,
        )
        logger.info(
            "Response action approval handled.",
            extra={
                "event_type": "response_approval",
                "username": current_user.get("username"),
                "action_id": payload.action_id,
                "approved": payload.approved,
            },
        )
        return result
    except Exception as e:
        logger.error(f"Failed to approve response action: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve response action: {e}")


@app.post("/api/v1/response/rollback")
async def rollback_response_action(
    payload: RollbackRequest,
    current_user: dict = Depends(get_current_user),
):
    if not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin privileges required to rollback response actions.")
    try:
        response = response_engine.rollback_action(
            action_id=payload.action_id,
            requested_by=payload.requested_by,
            reason=payload.reason,
        )
        logger.info(
            "Response action rolled back.",
            extra={
                "event_type": "response_rollback",
                "username": current_user.get("username"),
                "action_id": payload.action_id,
            },
        )
        return response
    except Exception as e:
        logger.error(f"Failed to rollback response action: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rollback response action: {e}")


from pydantic import BaseModel
from backend.network.active_scanner import ActiveScanner

active_scanner = ActiveScanner()

class ScanRequest(BaseModel):
    target: str

@app.post("/api/v1/scan")
async def run_active_scan(
    payload: ScanRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        logger.info(
            "Active scan requested.",
            extra={
                "event_type": "active_scan",
                "username": current_user.get("username"),
                "target": payload.target,
            },
        )
        result = await active_scanner.scan_target(payload.target)
        return result
    except Exception as e:
        logger.error(f"Failed to run active scan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run active scan: {e}")


@app.post("/api/v1/scan/analyze")
async def analyze_scan_results(
    payload: dict,
    current_user: dict = Depends(get_current_user),
):
    try:
        logger.info(
            "Scan analysis requested.",
            extra={
                "event_type": "scan_analyze",
                "username": current_user.get("username"),
            },
        )
        result = await groq_assistant.analyze_vulnerability(payload)
        return result
    except Exception as e:
        logger.error(f"Failed to analyze scan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze scan: {e}")


@app.websocket("/ws/realtime")
async def realtime_websocket(websocket: WebSocket, channels: str = ""):
    requested_channels = [item.strip() for item in channels.split(",") if item.strip()]
    await ws_manager.connect(websocket, requested_channels)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
            except ValueError:
                continue
            if payload.get("type") == "ack" and payload.get("alert_id"):
                try:
                    ack_payload = AlertAcknowledgementRequest(
                        acknowledged_by=payload.get("acknowledged_by", "analyst"),
                        comment=payload.get("comment"),
                    )
                    acknowledgement = await alert_service.acknowledge_alert(
                        int(payload["alert_id"]), ack_payload
                    )
                    await ws_manager.broadcast("alert_ack", acknowledgement.model_dump())
                except Exception:
                    continue
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


@app.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket, ["alerts"])
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
