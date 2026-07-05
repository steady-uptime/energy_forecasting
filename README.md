# Modular AutoML Energy Forecasting System

![Python](https://img.shields.io/badge/python-3.9+-blue)
![MLOps](https://img.shields.io/badge/MLOps-Production_Grade-orange)
![Architecture](https://img.shields.io/badge/Architecture-Service_Oriented-green)

## 🚀 Project Overview
This project is a production-grade, configuration-driven MLOps framework designed for energy-consumption forecasting. Unlike standard ML scripts, this system treats Machine Learning as a set of **decoupled, modular services**. 

The goal is to solve the "Hidden Technical Debt in ML Systems" by enforcing strict engineering principles: **reproducibility, testability, and portability.**

### 🎯 Core Engineering Laws
To ensure production readiness, the system strictly adheres to five architectural laws:
1.  **Zero Hardcoding:** No absolute paths; all paths are resolved dynamically via a central configuration loader.
2.  **Config-Driven Architecture:** All hyperparameters, file paths, and feature schemas reside in `.yaml` files.
3.  **Portability First:** Designed to run on any machine by using `pathlib` for cross-platform compatibility.
4.  **Decoupled Logic:** Complete separation of Data Engineering, Model Engineering, and Orchestration.
5.  **Singleton Config:** A single source of truth for the project root and global settings, ensuring consistency across all modules.

---

## 🏗 System Architecture

The system is organized into five distinct layers to separate concerns:

### A. Data & Configuration Layer
*   **Config Module:** Centralized YAML configuration management.
*   **Data Access Module:** Abstracted data repository (supports local, SQL, and Cloud storage).

### B. Core ML Lifecycle (Service-Oriented)
Each phase is a discrete service with clear input/output contracts:
*   **Ingestion $\rightarrow$ Preprocessing $\rightarrow$ Feature Engineering $\rightarrow$ Model Search $\rightarrow$ Evaluation $\rightarrow$ Registry**

### C. Orchestration Layer
Pipelines are defined as **Code-as-DAGs**. Instead of notebooks, the system uses Python scripts to orchestrate service execution:
*   `train_pipeline.py`: The primary production training flow.
*   `retrain_pipeline.py`: Triggered by drift signals.
*   `backfill_pipeline.py`: Handles historical data updates.

### D. Serving & Interface Layer
*   **FastAPI Inference API:** Endpoints for `/predict` and `/model_metadata`.
*   **CLI Tools:** Command-line interface for manual pipeline execution and champion selection.

### E. Observability & Governance
*   **Experiment Tracking:** Automated logging of hyperparameters and metrics.
*   **Data Validation Gatekeepers:** Strict schema validation between every pipeline stage.
*   **Model Cards:** Auto-generated documentation for every registered model.

---

## 🛠 Tech Stack
*   **Language:** Python (Type Hinted)
*   **Orchestration:** Custom Pipeline Orchestrator (Modular Service Pattern)
*   **Data Handling:** Pandas, NumPy, Pathlib
*   **ML Frameworks:** Scikit-Learn (Random Forest, etc.)
*   **Logging:** Loguru (Structured logging)
*   **Configuration:** PyYAML, Dataclasses
*   **API:** FastAPI (Planned)
*   **Deployment:** Docker (Planned)

---

## 📈 Project Milestones & Roadmap

### ✅ Completed Milestones
- [x] **Singleton Configuration Loader:** Implemented a type-safe, environment-aware config loader using Dataclasses.
- [x] **Service-Oriented Architecture:** Built standalone services for Ingestion, Preprocessing, and Feature Engineering.
- [x] **Data Validation Gatekeepers:** Implemented `DataValidator` to enforce schema contracts between pipeline stages (Raw $\rightarrow$ Sanitized $\rightarrow$ Engineered).
- [x] **Model Worker Factory:** Implemented the Factory Pattern to decouple the orchestrator from specific model implementations (e.g., Random Forest).
- [x] **Train Pipeline:** A fully functional `train_pipeline.py` that executes the end-to-end flow from raw CSV to saved model and metrics.
- [x] **Structured Logging:** Integrated `loguru` with `run_id` tracking for easy debugging in production.

### 🏗 In Progress / Planned
- [ ] **Model Registry:** A persistent storage system to version models, tags, and metadata.
- [ ] **Monitoring Service:** Implementation of drift detection signals to trigger retraining.
- [ ] **Retrain Pipeline:** Automated DAG for model updating based on production performance.
- [ ] **FastAPI Deployment:** Wrapping the model loader in a production-ready REST API.
- [ ] **Dockerization:** Multi-stage Docker builds for consistent environment deployment.

---

## 💻 Getting Started

### Prerequisites
*   Python 3.9+
*   Virtual Environment (`venv` or `conda`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/energy-forecasting-mlops.git
   cd energy-forecasting-mlops
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline:
   ```bash
   python -m pipelines.train_pipeline
   ```

---

## 📂 Directory Structure
```text
.
├── configs/                # YAML configuration files
├── data/                   # Raw, processed, and interim data
├── logs/                   # Structured pipeline logs
├── models/                 # Saved model artifacts
├── pipelines/             # Pipeline Orchestrators (DAGs)
├── src/                   # Core Logic
│   ├── core/              # Configuration, Services, Validation
│   ├── infra/             # Logging, Artifact Management, Repositories
│   └── utils/             # Helper functions
└── requirements.txt
```

⚖️ Legal & Compliance
This project uses the Electricity Load Diagrams 2011–2014 dataset, sourced from the UCI Machine Learning Repository. The data is licensed under CC BY 4.0. All modifications to the data are for educational and portfolio purposes only.

_Note: The pipeline includes a DataValidator gatekeeper to ensure the input data conforms to the expected schema before processing, ensuring robustness against data drift or schema changes._