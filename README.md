# 🌫️ AQI Intelligence

> An end-to-end Air Quality Index (AQI) prediction and monitoring system with anomaly detection, health risk classification, model explainability, and a live interactive Streamlit dashboard.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://aqi-intelligence-yuifhmspuc2b9taqfyysoq.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Open%20Source-green?style=for-the-badge)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Methodology](#methodology)
- [AQI Health Risk Scale](#aqi-health-risk-scale)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 🔍 Overview

Air pollution is a critical public health issue affecting millions globally. **AQI Intelligence** is a full-stack machine learning system designed to:

- **Predict** the Air Quality Index (AQI) from pollutant and meteorological readings using an ensemble of Random Forest and XGBoost regressors
- **Detect anomalies** in air quality data using Isolation Forest
- **Classify health risk** levels from predicted AQI values following EPA standards
- **Explain model predictions** using SHAP (SHapley Additive exPlanations) values
- **Visualize** everything through an interactive Streamlit web dashboard

---

## 🚀 Live Demo

The app is deployed and publicly accessible:

**👉 [https://aqi-intelligence-yuifhmspuc2b9taqfyysoq.streamlit.app](https://aqi-intelligence-yuifhmspuc2b9taqfyysoq.streamlit.app)**

---

## 📂 Dataset

Three data files power this project:

| File | Description |
|---|---|
| `city_day.csv` | Daily city-level AQI and pollutant readings |
| `station_day.csv` | Daily station-level pollutant measurements |
| `stations.csv` | Metadata about monitoring stations |

**Input features used for modelling:**

| Feature | Description |
|---|---|
| `PM2.5` | Fine particulate matter (μg/m³) |
| `PM10` | Coarse particulate matter (μg/m³) |
| `SO2` | Sulphur dioxide (μg/m³) |
| `NO2` | Nitrogen dioxide (μg/m³) |
| `CO` | Carbon monoxide (mg/m³) |
| `O3` | Ozone (μg/m³) |
| `TEMP` | Ambient temperature (°C) |
| `PRES` | Atmospheric pressure (hPa) |
| `DEWP` | Dew point temperature (°C) |
| `RAIN` | Precipitation (mm) |
| **`AQI`** | **Target variable** |

---

## 🗂️ Project Structure

```
AQI-intelligence/
│
├── dashboard/                  # Streamlit dashboard app
│
├── .devcontainer/              # Dev container configuration
│
├── preprocess.py               # Data loading, imputation & scaling
├── train_model.py              # Ensemble model training (RF + XGBoost)
├── anomaly_detection.py        # Isolation Forest anomaly detection
├── explainability.py           # SHAP-based model explainability
├── health_risk.py              # AQI → health risk classification
│
├── city_day.csv                # City-level daily AQI data
├── station_day.csv             # Station-level daily pollutant data
├── stations.csv                # Monitoring station metadata
│
└── requirements.txt            # Python dependencies
```

---

## 🧩 Modules

### `preprocess.py` — Data Preprocessing
- Loads raw CSV data
- Selects the 10 pollutant and weather features
- Applies **mean imputation** for missing values (`SimpleImputer`)
- Applies **standard scaling** (`StandardScaler`) for model readiness

### `train_model.py` — Model Training
- Trains two independent regressors on the preprocessed data:
  - **Random Forest** (`n_estimators=100`)
  - **XGBoost** (`n_estimators=200`, `learning_rate=0.05`, `max_depth=6`)
- Generates an **ensemble prediction** by averaging both models' outputs
- Evaluates performance using **RMSE** on the held-out test set
- Persists both trained models to disk via `joblib` for reuse in the dashboard

### `anomaly_detection.py` — Anomaly Detection
- Uses **Isolation Forest** (`contamination=0.05`) to flag unusual air quality readings
- Returns per-sample predictions: `1` (normal) or `-1` (anomaly)

### `explainability.py` — Model Explainability
- Loads the saved XGBoost model
- Computes **SHAP values** using `shap.Explainer`
- Generates a **SHAP summary plot** showing global feature importance and impact direction

### `health_risk.py` — Health Risk Classification
Maps predicted AQI values to standard EPA health risk categories:

```python
health_risk_level(aqi)  # returns one of 6 risk categories
```

---

## 🔄 Methodology

```
Raw Data (city_day.csv / station_day.csv)
          ↓
    preprocess.py
  ┌─────────────────────────┐
  │ Feature selection       │
  │ Mean imputation         │
  │ Standard scaling        │
  └─────────────────────────┘
          ↓
    train_model.py
  ┌──────────────────────────────────────┐
  │ Random Forest Regressor              │
  │ XGBoost Regressor                    │
  │ Ensemble avg → RMSE evaluation       │
  │ Save: random_forest.pkl, xgboost.pkl │
  └──────────────────────────────────────┘
          ↓
  ┌──────────────┬──────────────┬──────────────────┬──────────────┐
  │              │              │                  │              │
anomaly_      health_       explainability.py   dashboard/
detection.py  risk.py       SHAP summary plot   Streamlit app
Isolation    AQI → Risk
Forest       Category
```

---

## 🏥 AQI Health Risk Scale

| AQI Range | Category | Health Implication |
|---|---|---|
| 0 – 50 | ✅ Good | Air quality is satisfactory |
| 51 – 100 | 🟡 Moderate | Acceptable; minor concern for very few individuals |
| 101 – 150 | 🟠 Unhealthy for Sensitive Groups | Sensitive individuals may experience health effects |
| 151 – 200 | 🔴 Unhealthy | Everyone may begin to experience health effects |
| 201 – 300 | 🟣 Very Unhealthy | Health alert — serious effects for the general public |
| 300+ | ⚫ Hazardous | Emergency conditions; entire population at risk |

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/SoumyadeepChattopadhyay2004/AQI-intelligence.git
cd AQI-intelligence
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Preprocess data and train models

```bash
python preprocess.py
python train_model.py
```

### Step 2 — Run anomaly detection

```bash
python anomaly_detection.py
```

### Step 3 — Generate SHAP explainability plots

```bash
python explainability.py
```

### Step 4 — Launch the Streamlit dashboard locally

```bash
streamlit run dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-blue)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly)

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `scikit-learn` | Preprocessing, Random Forest, Isolation Forest |
| `xgboost` | Gradient boosted AQI regression |
| `shap` | Model explainability via SHAP values |
| `streamlit` | Interactive web dashboard |
| `plotly` / `matplotlib` | Data visualizations |
| `joblib` | Model serialization and loading |

---

## 👤 Author

**Soumyadeep Chattopadhyay**

[![GitHub](https://img.shields.io/badge/GitHub-SoumyadeepChattopadhyay2004-181717?logo=github)](https://github.com/SoumyadeepChattopadhyay2004)

---

## 📄 License

This project is open-source and available for educational and research purposes.

---

> ⭐ If you found this project useful, please consider giving it a star!
