# Safeguard-AI Lite: Project Blueprint

Safeguard-AI Lite is a professional, high-performance Cybersecurity Analyst Console designed for lightweight intrusion detection, network monitoring, and safe reconnaissance. It is built with a "Human-in-the-Loop" philosophy, focusing on providing actionable intelligence without performing offensive actions.

---

## 🛠️ Technology Stack

### **Backend (API Layer)**
- **Framework:** FastAPI (Python)
- **Server:** Uvicorn (with hot-reload for development)
- **Authentication:** JWT (JSON Web Tokens) with Bearer Token authorization.
- **Database:** **PostgreSQL (Cloud-hosted via Supabase)**.
- **ORM:** SQLAlchemy 2.0 (Relational mapping with connection pooling).
- **Network Logic:** Scapy, PyShark, and `dnspython` for packet analysis and safe recon.
- **AI Integration:** Groq SDK (Llama 3.1 models) for real-time SOC analyst assistance.

### **Frontend (Analyst Dashboard)**
- **Framework:** Streamlit
- **Design System:** Custom Glassmorphic CSS (Minimalist, dark-mode focused).
- **Visualizations:** Matplotlib and Pandas for traffic analytics and threat mix charts.
- **Navigation:** Dual-layer navigation (9 horizontal dashboard tabs + standalone sidebar pages).

### **Machine Learning (ML) Engine**
- **Library:** Scikit-learn (RandomForest and DecisionTree ensembles).
- **Interpretability:** SHAP (SHapley Additive exPlanations) for "Explainable AI" on every detection.
- **Data:** Trained on a multiclass intrusion detection dataset (Normal vs. Attack types).

---

## 📂 Project Structure

```text
PFAI/
├── .env                # Secret keys, Supabase URLs, and Admin credentials
├── .gitignore          # Comprehensive GitHub exclusion rules
├── requirements.txt    # Python dependencies (includes sqlalchemy, psycopg2-binary)
├── backend/
│   ├── api/            # FastAPI routes and main entry point (main.py)
│   ├── core/           # Security config, logging, and environment settings
│   ├── db/             # Database Layer
│   │   ├── database.py # SQLAlchemy engine and Base configuration
│   │   ├── models.py   # PostgreSQL SQLAlchemy models (Users, Alerts, etc.)
│   │   ├── session.py  # Dependency (get_db) and background session_scope
│   │   └── postgres_store.py # Data access layer (Postgres migration)
│   ├── ml/             # ML model loading and prediction logic
│   ├── network/        # Active scanner and packet capture services
│   ├── schemas/        # Pydantic models for API request/response
│   └── services/       # Refactored services (Auth, Alert, Log, Response)
├── frontend/
│   ├── App.py          # Main Dashboard entry point (9 horizontal tabs)
│   ├── api_utils.py    # Shared logic for API calls
│   ├── ui_components.py# Reusable glassmorphic UI elements
│   └── pages/          # Standalone sidebar pages (Active Scanner, Live Monitor)
├── models/             # Pre-trained .pkl models and SHAP explainers
└── logs/               # Persistent backend and frontend logs
```

---

## ⚙️ Core Operational Logic

### **1. Cloud Database & Connection Pooling**
- **Supabase Integration:** The system uses the **Supabase Transaction Pooler** (Port 6543) for reliable connections over IPv4 networks.
- **Resource Management:** 
    - **FastAPI Routes:** Use the `get_db` dependency for request-scoped sessions.
    - **Background Tasks:** (e.g., Packet Capture) Use the `session_scope` context manager to safely open/close sessions.
- **Auto-Migration:** On backend startup, `Base.metadata.create_all()` automatically initializes the PostgreSQL schema in Supabase.

### **2. Authentication & Security**
- **Admin Seeding:** If no admin exists in the PostgreSQL `users` table, the system seeds one using `ADMIN_USERNAME` and `ADMIN_PASSWORD` from `.env`.
- **JWT Flow:** Standard Bearer Token flow. Tokens are validated against the PostgreSQL user database.

### **3. Threat Detection Pipeline**
1. **Detection:** Scapy captures packets or users upload CSVs.
2. **Analysis:** ML models predict threats.
3. **Storage:** Results are written to the `scan_results` and `detection_alerts` tables in Supabase.
4. **Broadcasting:** Real-time updates are sent to the Streamlit frontend via WebSockets.

---

## 🚀 Deployment (Production)

### **Infrastructure**
- **Database:** Supabase (PostgreSQL).
- **Backend:** Render (FastAPI) or similar Linux environment.
- **Frontend:** Streamlit Cloud.

### **Environment Variables**
- `DATABASE_URL`: Must include `sslmode=require`.
- `GROQ_API_KEY`: Required for AI capabilities.
- `JWT_SECRET_KEY`: Keep this highly secure in production.

---

## 🎯 Development Roadmap
- [x] Migrate SQLite to **PostgreSQL (Supabase)**.
- [x] Implement robust SQLAlchemy session management.
- [x] Secure environment configuration and .gitignore.
- [ ] Implement full "Live Network Monitor" dashboard with historical trending.
