



Dataset
Name: Electricity Load Diagrams 2011–2014
Source: UCI Machine Learning Repository
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Description: This dataset contains hourly electricity load data for 370 clients, used to train the energy forecasting model.



# Modular AutoML Energy Forecasting System

![Project Status](https://img.shields.io/badge/Status-Production_Ready-blueviolet?style=for-the-badge)

This repository demonstrates a modular, production-ready MLOps architecture designed for scalability, portability, and maintainability. It moves beyond standard research notebooks by implementing strict **Software Engineering** principles, ensuring a seamless transition from model development to production deployment via a **Kubernetes-native orchestration layer.**

## 🏗 Architecture Principles

To ensure production readiness, this project adheres to five core engineering constraints:

1.  **Strongly-Typed Configuration Contracts**: Zero hardcoding. All hyperparameters, file paths, and transformation rules are externalized in `configs/` and mapped to **Dataclasses**. This enforces a "Fail-Fast" mechanism where configuration errors are caught during the system's boot phase.
2.  **Decoupled Logic (Separation of Concerns)**: Strict separation between Data Engineering (ETL), Model Engineering (Training/Inference), and Orchestration (Workflow Management). Components interact via defined interfaces, not shared state.
3.  **Portability & Dynamic Resolution**: All filesystem paths are resolved dynamically via a **Singleton Configuration Loader** using `pathlib`. This ensures the project is environment-agnostic and runs on any OS without modification to the source code.
4.  **Dynamic Contract Validation**: A dedicated validation layer enforces data integrity by validating the **runtime-generated schema** emitted by the Feature Engineering worker. This ensures the "Data Contract" is honored before data is handed off to the Model Worker.
5.  **Observability & Traceability**: Centralized logging using `loguru` for structured, leveled telemetry. Automated metadata serialization ensures experiment lineage (hyperparameters, metrics, and model URIs) is captured for every run.

## 🚀 Project Status

- [x] **Phase 1: Configuration Engine** (Singleton Config Loader, Dataclass Schema Enforcement)
- [x] **Phase 2: Data Engineering Pipeline** (Worker-based ETL, Dynamic Preprocessing)
- [x] **Phase 3: Model Engineering** (Trainable Workers, Artifact Management)
- [x] **Phase 4: Evaluation & Persistence** (Metrics, Logging, Metadata Serialization)
- [x] **Phase 5: Containerization & Testing** (Docker Environment Parity, Unit Testing)
- [x] **Phase 6: Orchestration** (Kubernetes Jobs, ConfigMaps, Persistent Volume Mapping)

## 🛠 Key Engineering Achievements

- **Singleton Pattern**: Implemented for Configuration Management to provide a **Single Source of Truth** and prevent redundant I/O operations.
- **Abstract Base Classes (ABC)**: Defines a strict contract for all Model Workers, ensuring interchangeability of different ML algorithms (**Polymorphism**).
- **Dependency Injection (DI)**: Preprocessing rules and persistence engines are injected into workers, decoupling the "How" (logic) from the "What" (configuration).
- **Contract Validation**: A dedicated validation layer ensures the data "contract" is honored at every gate in the pipeline, preventing silent failures and downstream errors.
- **Kubernetes Orchestration**: Transitioned from Docker Compose (Composition) to K8s Jobs (Orchestration), allowing for managed lifecycle execution and **ConfigMap-driven** environment injection.
- **Persistent State Management**: Implemented a **Pathing Contract** using Kubernetes Volumes to bridge the gap between host-machine storage and containerized file systems, ensuring data persistence across ephemeral container lifecycles.

## ⚙️ Component Breakdown

### 1. Configuration Management (`src/core/`)
- **DataClasses**: Provides type safety and schema enforcement for all configuration parameters.
- **Singleton Loader**: Dynamically calculates the `project_root` and maps YAML files to Dataclasses, providing a centralized, thread-safe configuration object.

### 2. Data Engineering (`src/core/`, `src/infra/`)
- **Worker Pattern**: Data loading and preprocessing are abstracted into dedicated worker classes.
- **Feature Engineering**: Isolated module for extracting features and encoding categorical variables.
- **Validator Module**: A generic, immutable utility that performs **Source Contract Validation** (verifying raw data integrity) and **Transformation Contract Validation** (verifying engineered features) at every gate.

### 3. Model Engineering (`src/core/`)
- **Contract Definition**: Uses ABCs to ensure every model implementation supports a standardized `.train()`, `.predict()`, and `.save()` interface.
- **Persistence Layer**: Standardized serialization using `joblib` for model weights and `json` for evaluation metadata.

### 4. Orchestration (`pipelines/`)
- **Orchestrator Pattern**: `pipelines/train_pipeline.py` acts as the primary entry point, delegating tasks to specific modules. It manages the workflow topology without knowing the internal implementation details of the workers.

## ⚙️ Development & Deployment Setup

### 1. Local Development (Virtual Environment)

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Execute training pipeline
python -m pipelines.train_pipeline
```

### 2. Kubernetes Deployment (Production Simulation)
This project utilizes a **Kubernetes Job** to execute the training lifecycle.

**Deploy Configuration & Job:**
```bash
# Apply the ConfigMap (Environment Variables)
kubectl apply -f k8s/configmap.yaml

# Apply the Training Job
kubectl apply -f k8s/job.yaml
```

**Monitor Execution:**
```bash
# View pod status
kubectl get pods -l job=energy-forecast-train-job

# Stream logs from the training worker
kubectl logs -f $(kubectl get pods -l job=energy-forecast-train-job -o jsonpath='{.items[0].metadata.name}')
```

## 🐳 Infrastructure Stack

- **Containerization**: Docker (Production-slim Python base).
- **Orchestration**: Kubernetes (K8s) Jobs for one-off training tasks.
- **Configuration Management**: K8s ConfigMaps for environment variable injection.
- **Storage**: Kubernetes Volumes with `hostPath` mapping for persistent data/model/log storage.
- **Observability**: Structured logging via `loguru`.

## 🧪 Testing & Quality Assurance

- **Unit/Integration Tests**: Automated test suite in `tests/` using `pytest`.
- **Schema Validation**: Contract validation gates between Data Engineering and Model Engineering modules.
- **Logging**: Centralized logging routed to the `logs/` directory with structured, leveled telemetry.
