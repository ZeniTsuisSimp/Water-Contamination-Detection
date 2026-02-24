---
title: Water Quality Monitor
emoji: 💧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 💧 Smart Water Contamination Detection

AI-powered water quality monitoring system using **IoT sensor data**, **Machine Learning**, **Docker**, and **CI/CD** — built with a professional MLOps pipeline.

## 🏗️ Project Structure

```
├── app/                    # Streamlit web application
│   └── main.py
├── src/                    # ML pipeline source code
│   ├── data_preprocessing.py   # Data loading, imputation, splitting
│   ├── train.py                # Model training + MLflow logging
│   ├── evaluate.py             # Evaluation + report generation
│   └── predict.py              # Prediction + anomaly detection
├── data/
│   ├── raw/                # Original, unprocessed CSVs
│   └── processed/          # Classified & cleaned datasets
├── notebooks/
│   └── model_training.ipynb    # Exploratory notebook (EDA + training)
├── tests/
│   └── validate_model.py       # CI model validation test
├── reports/                # Auto-generated metrics, plots
├── mlruns/                 # MLflow experiment tracking data
├── params.yaml             # Centralized hyperparameters & config
├── dvc.yaml                # DVC pipeline definition
├── Dockerfile              # Docker containerization
├── docker-compose.yml      # Docker Compose for local dev
├── requirements.txt        # Python dependencies
└── .github/workflows/
    └── ci-cd.yml           # GitHub Actions CI/CD pipeline
```

## 🚀 Quick Start

### 1. Setup
```bash
# Clone the repository
git clone <repo-url>
cd Water-Contamination-Detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python -m src.train
```
This trains RF + SVM models, logs to MLflow, and saves `water_model.pkl`.

### 3. Evaluate
```bash
python -m src.evaluate
```
Generates classification reports and confusion matrix plots in `reports/`.

### 4. Run the App
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run with Docker
```bash
docker-compose up --build
```

## 🔬 ML Pipeline

| Stage | Command | Output |
|-------|---------|--------|
| Preprocess | `python -m src.data_preprocessing` | `reports/data_summary.json` |
| Train | `python -m src.train` | `water_model.pkl`, `reports/metrics.json` |
| Evaluate | `python -m src.evaluate` | `reports/eval_metrics.json`, `reports/confusion_matrix.png` |

### DVC Pipeline
```bash
dvc repro        # Run full pipeline
dvc status       # Check if pipeline is up-to-date
dvc dag          # Visualize pipeline DAG
```

### MLflow
```bash
mlflow ui        # Open experiment tracker at http://localhost:5000
```

## 🔧 Configuration

All hyperparameters and thresholds are in [`params.yaml`](params.yaml):

- **Data paths** — raw/processed data locations
- **Feature names** — pH, Solids (TDS)
- **Model hyperparameters** — RF trees, SVM kernel
- **Anomaly thresholds** — critical pH/TDS limits

## 🧠 Hybrid Prediction Logic

1. **Rule-Based Safety Check** — Catches critical anomalies (pH < 4 or > 10, TDS > 3000)
2. **ML Model** — Random Forest or SVM predicts potability for non-critical inputs
3. **Confidence Smoothing** — Prevents unrealistic 100%/0% probabilities

## 🚢 CI/CD Pipeline

| Stage | Description |
|-------|-------------|
| **Lint & Test** | Flake8 linting + model validation test |
| **Docker Build** | Build image + health check |
| **Deploy** | Push to Hugging Face Spaces (main branch only) |

## 📊 Tech Stack

- **ML**: scikit-learn (Random Forest, SVM)
- **App**: Streamlit + Altair charts
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Containerization**: Docker
- **CI/CD**: GitHub Actions → Hugging Face Spaces