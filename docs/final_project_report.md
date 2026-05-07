# Safeguard-AI Lite: Lightweight Intrusion Detection and Explainable Triage for Campus and SME Networks

## Abstract

This project presents `Safeguard-AI Lite`, a lightweight intrusion detection and analyst-support platform designed for resource-constrained campus and small-to-medium enterprise (SME) environments. The system integrates a `FastAPI` backend, a `Streamlit` dashboard, classical machine learning classifiers, `SHAP`-based explainability, and `SQLite`-based operational logging. The motivation for the work is the persistent gap between high-performing research prototypes and practical, low-cost deployable intrusion detection systems (IDSs). To address this gap, the project emphasizes CPU-friendly models, modular preprocessing, secure API design, and human-readable recommendations. The implementation supports multiclass attack detection, including `Normal`, `DDoS`, `BruteForce`, and `PortScan` traffic categories, and includes optimization features such as feature pruning, model quantization, prediction caching, and optional JAX-assisted inference. On the current smoke evaluation artifact, the best model, a Random Forest classifier, achieved `83.33%` accuracy, `0.8300` weighted F1-score, and `0.9493` ROC-AUC, outperforming Logistic Regression on the same dataset split. The project demonstrates that an explainable, operationally usable IDS can be built with modest infrastructure while retaining strong extensibility for future research and deployment.

## 1. Introduction

### 1.1 Motivation

Modern organizations increasingly depend on continuous network connectivity, cloud-integrated services, and distributed endpoints. This same connectivity enlarges the attack surface, creating persistent exposure to denial-of-service attacks, reconnaissance, brute-force activity, credential abuse, and host compromise. Classical perimeter defenses such as firewalls remain necessary, but they are insufficient for identifying subtle or evolving attack patterns once traffic is allowed into the network. As noted in the foundational intrusion-detection model proposed by Denning, security violations can often be identified through deviations in system and network behavior rather than by signatures alone [1].

In campus and SME settings, the challenge is especially acute. These environments often lack dedicated security operations teams, large-scale monitoring infrastructure, and GPU-backed analytics platforms. Accordingly, there is strong practical value in a lightweight IDS that combines acceptable detection performance with easy deployment, interpretable outputs, and low operational cost.

### 1.2 Problem Statement

The problem addressed in this project is how to design and implement a production-oriented yet lightweight IDS platform that:

1. classifies network-flow or session-level records into multiple attack categories,
2. supports secure and auditable interaction through a web API,
3. provides analyst-facing explanations and recommended actions,
4. remains deployable on commodity CPU-only infrastructure, and
5. supports reproducible experimentation and simple containerized deployment.

### 1.3 Project Objectives

The specific objectives of `Safeguard-AI Lite` are:

1. to build an end-to-end intrusion detection workflow from data preprocessing through inference and visualization,
2. to compare multiple classical machine learning models for multiclass intrusion detection,
3. to integrate explainability and rule-based mitigation guidance into analyst workflows,
4. to persist users, scan results, alerts, and activity logs using a lightweight relational store, and
5. to package the system for local, containerized, and free-tier cloud deployment.

## 2. Related Work

Intrusion detection research has evolved from statistical anomaly detection to machine learning, deep learning, and hybrid intelligent systems. Denning’s early model established the core notion that abnormal usage patterns extracted from audit records can reveal security violations in real time [1]. This work laid the conceptual basis for later anomaly-based IDS designs.

As machine learning matured, IDS research increasingly shifted toward supervised and semi-supervised approaches. Buczak and Guven surveyed data mining and machine learning methods for cyber intrusion detection and highlighted a recurring tension between detection accuracy, computational cost, feature engineering complexity, and dataset quality [2]. Their survey remains relevant because many practical systems still need models that are interpretable, efficient, and robust to imperfect benchmark data.

Benchmark datasets have strongly influenced the field. KDD Cup 1999 was widely used for years, but Tavallaee et al. showed that the dataset suffered from redundancy and evaluation bias, motivating the introduction of `NSL-KDD` as a more balanced alternative [3]. Although still useful for educational baselines, NSL-KDD does not fully reflect contemporary traffic and attack behaviors. In response to these limitations, Moustafa and Slay introduced `UNSW-NB15`, which incorporates modern normal traffic and a broader set of contemporary attacks [4]. Similarly, Sharafaldin, Lashkari, and Ghorbani described `CICIDS2017` as a contemporary dataset generation effort with richer benign behavior, broader threat diversity, and more realistic traffic characteristics [5].

The choice of learning algorithm is also shaped by operational constraints. Random Forests remain attractive for IDS tasks because they are comparatively robust, support nonlinear decision boundaries, and expose feature importance estimates [6]. Gradient-boosted tree models such as `XGBoost` often deliver strong tabular classification performance with efficient training and inference [7]. However, high-performing models alone are insufficient in security operations, where analysts often need explanations before taking containment actions. SHAP, proposed by Lundberg and Lee, provides a theoretically grounded mechanism for local and global feature attribution and is particularly useful in security settings where interpretability affects trust and response quality [8]. More recent IDS surveys continue to observe that explainability, generalizability, and real-world deployment constraints remain open issues despite significant progress in model accuracy [9].

The present project builds on this literature by focusing on a practical middle ground: modern benchmark-inspired preprocessing, classical tabular classifiers suitable for CPU deployment, explainability through SHAP, and a web-based operational interface that supports both prediction and triage.

## 3. Methodology

### 3.1 System Design Strategy

The project follows a modular pipeline:

1. ingest raw CSV or JSON records,
2. sanitize and validate inputs,
3. apply the trained preprocessing stack,
4. classify records with the selected best model,
5. compute explainability outputs when requested,
6. attach rule-based response recommendations,
7. store scan results and alerts, and
8. present outputs in the Streamlit dashboard.

This separation of concerns allows the system to remain maintainable while supporting experimentation with preprocessing, models, and deployment optimizations.

### 3.2 Data Preprocessing

The preprocessing stage handles several issues common in IDS tabular data:

- missing values through row dropping or imputation,
- mixed numeric and categorical columns,
- target label encoding,
- numeric feature scaling,
- optional correlation-based feature removal,
- optional dimensionality reduction through PCA, and
- optional tree-based feature selection.

The pipeline persists the fitted preprocessor, label encoder, correlation filter, and feature-engineering components using `joblib`, ensuring that training-time transformations are reused exactly at inference time.

### 3.3 Model Training and Selection

The training module compares multiple classical classifiers:

- Logistic Regression,
- Random Forest,
- optional XGBoost,
- optional LightGBM.

The data split is stratified when label distributions permit it, and all stochastic components use fixed random seeds for reproducibility. Model evaluation uses:

- Accuracy,
- Precision,
- Recall,
- F1-score,
- ROC-AUC.

For multiclass tasks, the system additionally generates a confusion matrix and a per-class report. The best model is selected by a configurable metric, with `F1-score` used by default in the current implementation.

### 3.4 Explainability and Recommendations

For interpretability, the project integrates SHAP explanations. Global explanations summarize the features that drive model predictions overall, while local explanations show which features contributed most to a specific decision. These explanations are exposed through both the API and the Streamlit UI.

To support actionability, a rule-based recommendation layer maps predicted attack classes to human-readable guidance. For example, `DDoS` predictions map to suggestions such as blocking offending sources and enabling rate limiting, while `BruteForce` predictions map to account lockout and password reset actions.

### 3.5 Deployment Optimization

Because the target use case emphasizes low-resource deployment, the project also includes:

- feature pruning after model selection,
- float32-oriented quantization of selected learned parameters,
- compressed model bundle persistence,
- prediction caching via `joblib.Memory`,
- optional JAX-backed inference metadata for Logistic Regression.

These measures are intended to reduce storage footprint and repeated inference cost while preserving acceptable accuracy.

## 4. Implementation Details

### 4.1 Backend Architecture

The backend is implemented in `FastAPI` and centered in `backend/api/main.py`. It provides:

- `POST /predict`
- `POST /upload`
- `GET /stats`
- `GET /model_info`
- `GET /health`
- authentication endpoints for admin creation and login

The backend uses structured exception handling, JWT-based authentication, request validation via Pydantic schemas, and security middleware such as CORS configuration and sanitized input handling.

### 4.2 Database and Logging

The persistence layer uses `SQLite` with transactional, parameterized CRUD helpers. The schema includes tables for:

- users,
- scan results,
- alerts,
- activity logs.

This design supports lightweight deployment while preserving essential auditability. Rotating JSON logging is used in both backend and frontend paths to capture login attempts, suspicious input, prediction events, and detected attacks.

### 4.3 Frontend Dashboard

The frontend is implemented in `Streamlit` and includes the following functional views:

- Home,
- Upload,
- Live Predictions,
- Statistics,
- Analytics,
- Explanations,
- About.

The Upload page supports CSV submission, preview, prediction display, CSV result export, and first-row explanation rendering. The Live Predictions page simulates traffic every two seconds and updates recent events and alert summaries. The Analytics and Explainability pages provide operational summaries and feature-attribution views.

### 4.4 Dataset Used in the Current Prototype

The current repository contains smoke datasets and saved evaluation artifacts rather than the full public benchmark corpora. However, the project design, preprocessing scripts, and report are aligned with public IDS datasets such as `NSL-KDD`, `UNSW-NB15`, and `CICIDS2017`. In this design, `CICIDS2017` is the most suitable primary target for campus and SME-like traffic because it better reflects modern attack diversity and mixed benign behavior [5].

### 4.5 Security Controls

Security-related implementation details include:

- bcrypt/passlib-based password hashing,
- JWT bearer authentication,
- column and type validation for CSV uploads,
- suspicious-input stripping and sanitization,
- parameterized SQLite access,
- structured error responses,
- static analysis with Black, Flake8, Mypy, and Bandit,
- protected inference and statistics endpoints.

## 5. Results

### 5.1 Model Comparison

The current saved multiclass smoke evaluation artifact reports the following metrics:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8333 | 0.8471 | 0.8333 | 0.8300 | 0.9493 |
| Logistic Regression | 0.7778 | 0.7903 | 0.7778 | 0.7777 | 0.9414 |

The Random Forest model was selected as the best available model under the current saved evaluation because it achieved the strongest weighted F1-score and the highest ROC-AUC.

### 5.2 Per-Class Performance

The current per-class report for the selected Random Forest model is shown below:

| Class | Support | Per-class Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| BruteForce | 18 | 0.8889 | 0.8000 | 0.8889 | 0.8421 |
| DDoS | 18 | 0.9444 | 0.9444 | 0.9444 | 0.9444 |
| Normal | 18 | 0.8889 | 0.7273 | 0.8889 | 0.8000 |
| PortScan | 18 | 0.6111 | 0.9167 | 0.6111 | 0.7333 |

These results suggest that the current model distinguishes `DDoS` traffic most reliably, while `PortScan` remains the most challenging class in the current evaluation artifact. The relatively lower recall for `PortScan` is reflected in the confusion matrix, where several PortScan samples are misclassified as `Normal`.

### 5.3 Confusion Matrix Interpretation

The saved confusion matrix for the selected model is:

| Actual \\ Predicted | BruteForce | DDoS | Normal | PortScan |
|---|---:|---:|---:|---:|
| BruteForce | 16 | 0 | 1 | 1 |
| DDoS | 1 | 17 | 0 | 0 |
| Normal | 2 | 0 | 16 | 0 |
| PortScan | 1 | 1 | 5 | 11 |

The most notable error concentration is the confusion of `PortScan` traffic with `Normal` traffic. This suggests either partial feature overlap in the current dataset slice or insufficient representation of PortScan behavior in the current training configuration.

### 5.4 Suggested Figures for Submission

The written report can be accompanied by the following figures generated from the saved artifacts:

1. **Figure 1. System Architecture Diagram**  
   A block diagram showing Streamlit, FastAPI, the ML service, SHAP, and SQLite.

2. **Figure 2. Model Comparison Bar Chart**  
   A grouped bar chart comparing accuracy, F1-score, and ROC-AUC for Logistic Regression and Random Forest.

3. **Figure 3. Confusion Matrix Heatmap**  
   A heatmap derived from `confusion_matrix.csv`.

4. **Figure 4. Per-Class F1-score Chart**  
   A bar chart showing class-level F1 for BruteForce, DDoS, Normal, and PortScan.

5. **Figure 5. SHAP Global Feature Importance Plot**  
   A summary of the most influential features in the selected model.

6. **Figure 6. Streamlit Upload and Explainability Screenshots**  
   Screenshots of the CSV upload workflow and first-row explanation panel.

## 6. Conclusion

This project demonstrates that a lightweight, explainable, and operationally oriented intrusion detection platform can be built using classical machine learning, modern web tooling, and modest infrastructure. The resulting system does more than classify records: it authenticates users, validates uploaded data, logs security-relevant events, explains predictions, and presents response guidance in a usable dashboard.

The current results indicate that the Random Forest model provides stronger multiclass performance than Logistic Regression on the saved evaluation artifact, particularly in ROC-AUC and weighted F1-score. At the same time, the confusion matrix reveals a meaningful limitation in the detection of PortScan traffic, showing that model quality must be understood not only in aggregate but also at the class level.

From an engineering perspective, the project successfully integrates machine learning experimentation with backend security, frontend usability, logging, testing, static analysis, CI, containerization, and cloud deployment guidance. This makes `Safeguard-AI Lite` suitable as both an academic prototype and a foundation for further applied work.

## 7. Future Work

Several directions can strengthen the project:

1. replace SQLite with PostgreSQL or another managed database for durable multi-user deployment,
2. train on the full `CICIDS2017` and `UNSW-NB15` corpora rather than smoke data,
3. add temporal or session-aware models for improved generalization,
4. evaluate optional `XGBoost` and `LightGBM` under full benchmark settings,
5. add threshold tuning and cost-sensitive learning to reduce false negatives on difficult classes such as PortScan,
6. integrate ONNX export and runtime benchmarking for edge deployment,
7. extend recommendation logic with asset-aware context and automated response playbooks,
8. explore semi-supervised or online anomaly detection for previously unseen attacks.

## References

[1] D. E. Denning, “An Intrusion-Detection Model,” *IEEE Transactions on Software Engineering*, vol. SE-13, no. 2, pp. 222–232, 1987. DOI: `10.1109/TSE.1987.232894`  
https://www.cerias.purdue.edu/apps/reports_and_papers/view/979

[2] A. L. Buczak and E. Guven, “A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection,” *IEEE Communications Surveys & Tutorials*, vol. 18, no. 2, pp. 1153–1176, 2016. DOI: `10.1109/COMST.2015.2494502`  
https://www.researchgate.net/publication/283811300_A_Survey_of_Data_Mining_and_Machine_Learning_Methods_for_Cyber_Security_Intrusion_Detection

[3] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” in *Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications*, 2009. DOI: `10.1109/CISDA.2009.5356528`  
https://www.researchgate.net/publication/48446353_A_detailed_analysis_of_the_KDD_CUP_99_data_set

[4] N. Moustafa and J. Slay, “UNSW-NB15: a comprehensive data set for network intrusion detection systems,” in *2015 Military Communications and Information Systems Conference (MilCIS)*, 2015. DOI: `10.1109/MilCIS.2015.7348942`  
https://research.unsw.edu.au/projects/unsw-nb15-dataset

[5] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” in *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 2018.  
https://www.unb.ca/cic/datasets/ids-2017.html

[6] L. Breiman, “Random Forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. DOI: `10.1023/A:1010933404324`  
https://www.researchgate.net/publication/275342330_Random_Forests

[7] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016. DOI: `10.1145/2939672.2939785`  
https://dl.acm.org/doi/10.1145/2939672.2939785

[8] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017.  
https://arxiv.org/abs/1705.07874

[9] H. Liu and B. Lang, “Machine Learning and Deep Learning Methods for Intrusion Detection Systems: A Survey,” *Applied Sciences*, vol. 9, no. 20, 4396, 2019. DOI: `10.3390/app9204396`  
https://www.mdpi.com/2076-3417/9/20/4396
