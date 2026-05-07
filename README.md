# Safeguard-AI Lite

Lightweight intrusion detection and triage for campus and SME-style networks. Safeguard-AI Lite combines a `FastAPI` backend, a `Streamlit` analyst dashboard, classical tabular ML models, `SHAP` explainability, `SQLite` logging, and container-friendly deployment.

## Overview

This project is designed for low-resource environments where teams need:

- fast classification of uploaded or simulated traffic records
- interpretable predictions with feature-level explanations
- lightweight persistence for users, scans, alerts, and activity logs
- a practical UI for upload, live simulation, analytics, and recommendations
- CPU-friendly training and deployment without GPU requirements

The current stack supports multiclass intrusion labels such as `Normal`, `DDoS`, `PortScan`, `BruteForce`, and `Botnet`, with optional model optimization for deployment.

## Key Features

- `FastAPI` API with JWT authentication and protected prediction endpoints
- `Streamlit` dashboard with Upload, Live Predictions, Statistics, Analytics, and Explainability views
- CSV and JSON prediction workflows
- Rule-based response recommendations per detected attack class
- `SHAP` global and local explanations
- `SQLite` persistence for users, scan results, alerts, and activity logs
- Model comparison across `LogisticRegression`, `RandomForest`, optional `XGBoost`, and optional `LightGBM`
- Deployment optimization with feature pruning, float32 quantization, optional JAX inference metadata, and cached predictions
- Docker and GitHub Actions support

## Architecture

```text
                    +----------------------+
                    |  Streamlit Frontend  |
                    |  frontend/app.py     |
                    +----------+-----------+
                               |
                               | HTTP / JSON / CSV
                               v
                    +----------------------+
                    |   FastAPI Backend    |
                    |  backend/api/main.py |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
   +------------------------+       +------------------------+
   |  Model + SHAP Service  |       |   SQLite Log Store     |
   | backend/services/*     |       | backend/db/sqlite_*    |
   +-----------+------------+       +-----------+------------+
               |                                    |
               v                                    v
   +------------------------+       +------------------------+
   | Preprocessing / ML     |       | Users / Alerts / Logs  |
   | ml/preprocessing.py    |       | Scan results / Stats   |
   | ml/training.py         |       +------------------------+
   | ml/explainability.py   |
   +------------------------+
```

## Tech Stack

- `Frontend`: Streamlit, Matplotlib, Pandas
- `Backend`: FastAPI, Pydantic, Uvicorn
- `ML`: scikit-learn, SHAP, optional XGBoost, optional LightGBM, optional JAX
- `Storage`: SQLite
- `Security`: JWT auth, bcrypt/passlib hashing, input sanitization, parameterized SQL
- `DevOps`: Docker, Docker Compose, GitHub Actions, Black, Flake8, Mypy, Bandit, Pytest

## Folder Structure

```text
backend/
  api/                 FastAPI app and endpoints
  core/                config, logging, security
  db/                  SQLite persistence layer
  dependencies/        auth dependencies
  schemas/             Pydantic request/response models
  services/            auth, model, logging, validation, recommendations
  utils/               sanitization helpers
frontend/
  app.py               Streamlit dashboard entrypoint
  api_client.py        backend API client
  sample_data.py       fake live traffic generation
ml/
  preprocessing.py     cleaning, encoding, scaling, feature engineering
  training.py          model training, evaluation, selection
  optimization.py      feature pruning, quantization, JAX metadata
  explainability.py    SHAP utilities
data/
  processed/           smoke datasets and processed CSVs
models/
  trained_*            saved model bundles and reports
  preprocessing_*      saved preprocessing artifacts
  cache/               prediction cache
scripts/
  train_intrusion_models.py
  preprocess_dataset.py
  download_merge_intrusion_datasets.py
tests/
  API, preprocessing, training, E2E, and UI-adjacent tests
```

## Installation

### Local Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Start the backend.
5. Start the frontend.

```bash
git clone <your-repo-url>
cd PFAI

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Backend:

```bash
uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```bash
streamlit run frontend/app.py
```

Open:

- Frontend: `http://localhost:8501`
- Backend health: `http://localhost:8000/health`

### Docker Setup

Build and start everything:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`

Notes:

- The Compose stack includes a lightweight volume-backed `database` service because this app uses `SQLite`, not a standalone DB server.
- SQLite is fine for demos and lightweight deployments, but it is not ideal for high-write multi-instance production setups.

## Environment Variables

Backend:

```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_BUNDLE_PATH=models/trained_multiclass_smoke/best_model.pkl
SAFEGUARD_DB_PATH=safeguard_ai.db
BACKEND_LOG_FILE=logs/backend.log
PREDICTION_CACHE_DIR=models/cache/predictions
JWT_SECRET_KEY=replace-with-a-long-random-secret
ALLOWED_ORIGINS=http://localhost:8501
USE_JAX_INFERENCE=true
```

Frontend:

```env
SAFEGUARD_API_BASE_URL=http://127.0.0.1:8000
```

## Usage

### 1. Create the first admin

```bash
curl -X POST http://127.0.0.1:8000/auth/create-admin ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"admin_user\",\"password\":\"StrongPass123\"}"
```

### 2. Login

```bash
curl -X POST http://127.0.0.1:8000/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"admin_user\",\"password\":\"StrongPass123\"}"
```

Use the returned bearer token in authenticated requests.

### 3. Predict from JSON

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Authorization: Bearer <token>" ^
  -H "Content-Type: application/json" ^
  -d "{\"records\":[{\"f0\":0.1,\"f1\":0.2,\"f2\":0.3,\"f3\":0.4,\"f4\":0.5,\"f5\":0.6,\"f6\":0.7,\"f7\":0.8,\"f8\":0.9,\"f9\":1.0,\"service\":\"http\"}],\"include_explanations\":true,\"explanation_top_k\":5}"
```

### 4. Predict from CSV

Use the Upload page in Streamlit or send a multipart request to `/predict`.

Streamlit Upload flow:

1. Open the `Upload` tab.
2. Select a CSV log file.
3. Review the preview table.
4. Submit the file to the backend.
5. Inspect predictions, recommendations, and the first-row SHAP explanation.
6. Download enriched prediction results as CSV.

### Screenshot Placeholders

Add screenshots here when available:

- `docs/screenshots/home.png`
- `docs/screenshots/upload.png`
- `docs/screenshots/live_predictions.png`
- `docs/screenshots/explanations.png`

Example markdown:

```md
![Upload Page](docs/screenshots/upload.png)
![Explainability View](docs/screenshots/explanations.png)
```

## Model Training

Train and compare available classifiers:

```bash
python scripts/train_intrusion_models.py ^
  --input data/processed/smoke_train.csv ^
  --target attack_class ^
  --selection-metric f1 ^
  --random-state 42 ^
  --enable-jax-conversion
```

Optional deployment-oriented training:

```bash
python scripts/train_intrusion_models.py ^
  --input data/processed/smoke_train.csv ^
  --target attack_class ^
  --enable-feature-pruning ^
  --feature-prune-max-features 50 ^
  --optimization-max-metric-drop 0.02 ^
  --enable-jax-conversion
```

Outputs include:

- `best_model.pkl`
- `model_results.csv`
- `confusion_matrix.csv`
- `per_class_report.csv`

## Dataset Sources

Recommended public IDS datasets for this project:

- `NSL-KDD`
- `UNSW-NB15`
- `CICIDS2017`

Why they matter:

- `NSL-KDD` is useful as a baseline and teaching dataset.
- `UNSW-NB15` is a stronger modern tabular benchmark.
- `CICIDS2017` is the best fit for campus and SME-style traffic behavior.

Official sources:

- CICIDS2017: `https://www.unb.ca/cic/datasets/ids-2017.html`
- UNSW-NB15: `https://research.unsw.edu.au/projects/unsw-nb15-dataset`
- NSL-KDD reference: `https://fkie-cad.github.io/COMIDDS/content/datasets/nsl_kdd_dataset/`

## API Summary

Public:

- `GET /health`
- `POST /auth/create-admin`
- `POST /auth/login`

Protected:

- `POST /predict`
- `POST /upload`
- `GET /stats`
- `GET /model_info`

## Developer Guide

### Key Modules

- [backend/api/main.py](/d:/4th%20Sem/Projects/PFAI/backend/api/main.py)  
  Main API app, auth routes, upload/predict endpoints, exception handling.

- [backend/services/model_service.py](/d:/4th%20Sem/Projects/PFAI/backend/services/model_service.py)  
  Prediction pipeline, cached inference, JAX-aware scoring path, response assembly.

- [backend/services/validation_service.py](/d:/4th%20Sem/Projects/PFAI/backend/services/validation_service.py)  
  Schema validation, coercion, CSV safety checks.

- [backend/services/auth_service.py](/d:/4th%20Sem/Projects/PFAI/backend/services/auth_service.py)  
  JWT creation/validation and admin authentication.

- [backend/db/sqlite_store.py](/d:/4th%20Sem/Projects/PFAI/backend/db/sqlite_store.py)  
  Transaction-safe SQLite CRUD for users, scans, alerts, and activity logs.

- [frontend/app.py](/d:/4th%20Sem/Projects/PFAI/frontend/app.py)  
  Streamlit dashboard, upload flow, live simulation, analytics, explanations.

- [ml/preprocessing.py](/d:/4th%20Sem/Projects/PFAI/ml/preprocessing.py)  
  Data cleaning, missing handling, encoding, scaling, correlation filtering, PCA/tree selection.

- [ml/training.py](/d:/4th%20Sem/Projects/PFAI/ml/training.py)  
  Model comparison, evaluation, saving selected bundles and reports.

- [ml/optimization.py](/d:/4th%20Sem/Projects/PFAI/ml/optimization.py)  
  Feature pruning, quantization, ONNX/JAX metadata, bundle sizing helpers.

- [ml/explainability.py](/d:/4th%20Sem/Projects/PFAI/ml/explainability.py)  
  SHAP explainers, local/global importance utilities, Streamlit-ready figures.

### Quality and Validation

Run the main checks locally:

```bash
python -m black --check backend frontend ml scripts tests
python -m flake8 --jobs=1 backend frontend ml scripts tests
python -m mypy --config-file mypy.ini backend frontend ml scripts tests
python -m bandit -r backend frontend ml -f txt
pytest tests -q
```

### CI

GitHub Actions is configured to:

- run linting
- run tests
- train a CPU-only model
- rerun training with fixed seeds
- compare outputs for reproducibility

See [ci.yml](/d:/4th%20Sem/Projects/PFAI/.github/workflows/ci.yml).

## Known Limitations

- SQLite is not a horizontally scalable production database.
- ONNX export is optional and depends on extra packages not installed by default.
- Free-tier hosting is fine for demos, but persistent storage and uptime will be limited.

## License

Add your preferred license here before public release.
