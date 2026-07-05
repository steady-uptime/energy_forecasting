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

## 🏗 System Architecture & Blueprints

The system is organized into five distinct layers to separate concerns, moving beyond a simple script to a **Service-Oriented Architecture (SOA)**.

### 1. High‑Level System Architecture (Layers)

This diagram represents the **System Topology**. It separates the "Core" (ML Logic) from the "Infra" (Utilities) and "Orchestration" (Execution), ensuring that the ML logic is decoupled from the underlying infrastructure.

![System Topology](./assets/diagrams/architecture.png)

<details>
<summary>View Architecture Logic (Mermaid Code)</summary>

```text

graph TD

    subgraph UI["User Interface / Entry Points"]
        CLI[CLI Tools]
        API[FastAPI Inference]
    end

    subgraph ORCH["Orchestration Layer"]
        TrainPipe[Train Pipeline]
        RetrainPipe[Retrain Pipeline]
        MonitorPipe[Monitor Pipeline]
    end

    subgraph CORE["Core ML Logic"]
        Ingestion[Ingestion Service]
        Preprocessing[Preprocessing Service]
        Features[Feature Engineering]
        Search[Model Search / AutoML]
        Eval[Evaluation Service]
        Registry[Model Registry]
    end

    subgraph INFRA["Infrastructure Layer"]
        Storage[Model Store & Data Repo]
        Tracker[Experiment Tracker]
        Logger[Structured Logger]
        Secrets[Secrets Management]
    end

    subgraph CONFIG["Configuration Layer"]
        Config[YAML Configs]
    end

    Config -.-> TrainPipe
    Config -.-> Registry
    
    TrainPipe --> Ingestion
    Ingestion --> Preprocessing
    Preprocessing --> Features
    Features --> Search
    Search --> Eval
    Eval --> Registry
    
    Registry --> API
    API --> Prediction[Prediction]
    Prediction --> MonitorPipe
    MonitorPipe -- "Drift Detected" --> RetrainPipe
    RetrainPipe --> TrainPipe

    Ingestion --- Storage
    Preprocessing --- Storage
    Features --- Storage
    Search --- Storage
    Eval --- Storage
    Registry --- Storage

    Ingestion --- Tracker
    Preprocessing --- Tracker
    Features --- Tracker
    Search --- Tracker
    Eval --- Tracker
    Registry --- Tracker

    Ingestion --- Logger
    Preprocessing --- Logger
    Features --- Logger
    Search --- Logger
    Eval --- Logger
    Registry --- Logger
```

</details>

---

### 2. The Training Pipeline (Data Flow)

This diagram illustrates the **Worker Pattern** and **Data Flow**. It highlights the "Fail-Fast" principle via **Validation Gates**, ensuring that data integrity is verified at every transition point between Data Engineering and ML Engineering.

![System Topology](./assets/diagrams/ml-data-flow.png)

<details>
<summary>View Training Pipeline (Mermaid Code)</summary>

```text

flowchart TD

subgraph DE["Data Engineering"]
    RawData[(Raw Data)] --> Load[Loader]
    Load --> Val1{Validator}
    Val1 -- Fail --> Error[RuntimeError]
    Val1 -- Pass --> Sanitized[Sanitized Data]
    Sanitized --> Pre[Preprocessor]
    Pre --> Processed[(Processed Data)]
end

subgraph FE["Feature Engineering"]
    Processed --> Feat[Feature Engineer]
    Feat --> Matrix[Feature Matrix]
    Matrix --> Val2{Validator}
    Val2 -- Fail --> Error
    Val2 -- Pass --> Split[Train/Test Split]
end

subgraph ME["Model Engineering (AutoML)"]
    Split --> Search[Search Space Loop]
    Search --> Train[Model Worker]
    Train --> Metrics[Metrics Calculation]
    Metrics --> Eval[Evaluation Logic]
    Eval --> Champion{Champion Selection}
end

subgraph RP["Registry & Persistence"]
    Champion -- "Winner" --> Reg[Register Model]
    Reg --> Store[(Model Store)]
    Reg --> Tracker[Experiment Tracker]
end

style Val1 fill:#f96,stroke:#333
style Val2 fill:#f96,stroke:#333
style Champion fill:#f96,stroke:#333
style Error fill:#ff9999
```

</details>

---

### 3. The Production Feedback Loop (MLOps Cycle)

This diagram demonstrates the **Observability & Automation** lifecycle. It shows how the system behaves in production, where the **Model Registry** acts as the "Single Source of Truth" to trigger automated retraining based on live drift signals.

![System Topology](./assets/diagrams/feedback-loop.png)

<details>
<summary>View Production Feedback Loop (Mermaid Code)</summary>

```text

graph TD

    subgraph PROD["Production Environment"]
        LiveData[(Live Data)] --> API[Inference API]
        API --> Prediction[Prediction]
        Prediction --> Monitor[Monitoring Service]
    end

    subgraph OBS["Observability & Governance"]
        Monitor --> Drift{Drift / Error?}
        Drift -- No --> Prediction
        Drift -- Yes --> Alert[Alert / Trigger]
    end

    subgraph AUTO["Automated Retraining"]
        Alert --> RetrainPipe[Retrain Pipeline]
        RetrainPipe --> TrainPipe[Training Pipeline]
        TrainPipe --> Registry[Model Registry]
        Registry -- "Update Champion" --> API
    end

    style Drift fill:#f96,stroke:#333
    style Alert fill:#ff9999
    style Registry fill:#bbf,stroke:#333
```

</details>


### 🔑 Key Architectural Notes

- **The Registry as the Heart:** Notice in Diagrams 1 and 3, the **Registry** is the central hub. It is the **Single Source of Truth** for what the API should use. The API never looks at the raw Model Store directly; it only asks the Registry for the current "Champion."
- **Validation Gates:** In Diagram 2, the highlighted **Validators** are your primary tool for enforcing **Data Contracts**. This prevents "silent failures" where corrupted data propagates downstream.
- **Decoupled Search:** The **Model Search** block is isolated. This allows you to swap a Random Search for Bayesian Optimization or a Genetic Algorithm just by changing the `search_space.yaml` without modifying the core `train_pipeline.py`.
- **The "Worker" Pattern:** The **Model Worker** is a clean abstraction. The Orchestrator doesn't care _how_ the model trains; it only knows that it takes a Feature Matrix and returns a standardized Model object.



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
```text
This project uses the Electricity Load Diagrams 2011–2014 dataset, sourced from the UCI Machine Learning Repository. The data is licensed under CC BY 4.0. All modifications to the data are for educational and portfolio purposes only.
```

_Note: The pipeline includes a DataValidator gatekeeper to ensure the input data conforms to the expected schema before processing, ensuring robustness against data drift or schema changes._