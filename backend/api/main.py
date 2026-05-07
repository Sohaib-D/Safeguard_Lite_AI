from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.logging_config import configure_logger
from backend.dependencies.auth import get_current_user, get_optional_current_user
from backend.core.security import configure_security
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
from backend.services.auth_service import AuthService, AuthenticationError
from backend.services.log_service import LogService
from backend.services.model_service import ModelService
from backend.services.validation_service import InputValidationError
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
