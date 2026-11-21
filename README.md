<div align="center">

# 🏔️ AI/ML Dynamic Landslide Early Warning System (LEWS)
### *Operational Real-Time Hazard Forecasting & Emergency Broadcast Platform for Sikkim, Eastern Himalayas*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite-61dafb?logo=react&logoColor=black)](https://reactjs.org/)
[![ML Models](https://img.shields.io/badge/Ensemble-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange)](https://github.com/arjunb1204/Landslide-Early-Warning-Sikkim)
[![PR-AUC](https://img.shields.io/badge/PR--AUC-0.894-success)](#-model-benchmarks--performance)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.948-success)](#-model-benchmarks--performance)
[![Protocol](https://img.shields.io/badge/Standard-OASIS%20CAP%20v1.2-red)](#-ndma-multi-channel-alert-dispatch-engine)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

<p align="center">
  <strong>An operational end-to-end Early Warning System combining physics-informed topographic modeling, real-time satellite remote sensing (NASA SMAP / IMD), spatial-block regularized GBDT ensembles, TreeSHAP explainability, and automated cellular hardware alert broadcast.</strong>
</p>

</div>

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Two-Tier Decoupled Physical Framework](#-two-tier-decoupled-physical-framework)
- [Machine Learning & Geostatistics](#-machine-learning--geostatistical-leakage-prevention)
- [Model Benchmarks & Performance](#-model-benchmarks--performance)
- [Model Explainability (TreeSHAP)](#-model-explainability-treeshap)
- [Full-Stack GIS Incident Platform](#-full-stack-gis-incident-platform)
- [NDMA Multi-Channel Alert Dispatch Engine](#-ndma-multi-channel-alert-dispatch-engine)
- [Repository Structure](#-repository-structure)
- [Quickstart & Installation](#-quickstart--installation)
- [Scientific Bibliography & Citations](#-scientific-bibliography--citations)

---

## 🛰️ Executive Overview

The **SIH26001 Landslide Early Warning System (LEWS)** is engineered specifically for the geo-dynamically fragile terrain of **Sikkim ($7,096\text{ km}^2$)** in the Eastern Himalayas.

Unlike traditional static hazard maps that remain unchanged for years, this system discretizes the state into **$7,120$ dynamic computational grid cells ($1\text{ km} \times 1\text{ km}$)** and continuously fuses:
1. **Static Terrestrial Vulnerability (Layer 1):** ALOS PALSAR DEM derivatives, structural geology (Main Central Thrust / MBT), drainage systems, and road cuts (NH-10).
2. **Dynamic Hydro-Meteorological Triggers (Layer 2):** NASA SMAP satellite rootzone soil moisture, IMD precipitation (1h, 24h, 72h), and antecedent hydrologic memory decay.
3. **Calibrated Gradient Boosted Ensemble:** High-throughput soft-voting meta-ensemble (XGBoost, LightGBM, CatBoost) with isotonic regression and monotonic geotechnical constraints.
4. **Emergency Dissemination:** OASIS CAP v1.2 XML generation, C-DOT Cell Broadcast siren simulation, and a zero-dependency USB Android ADB cellular SIM auto-dispatcher.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Data Ingestion & Discretization Layer
        A["ALOS PALSAR DEM 12.5m<br/>(Slope, Aspect, Curvature)"] --> D["Feature Engineering Pipeline<br/>(TWI, SPI, STI, Fault Dist)"]
        B["NASA SMAP Microwave Radiometer<br/>(0-5cm Rootzone Saturation Ratio)"] --> D
        C["IMD AWS & NASA GPM Satellite<br/>(1h, 24h, 72h, 7-Day API_t)"] --> D
    end

    subgraph Two-Tier Decoupled ML Engine
        D --> E["Spatial Block Split (Moran's I Buffer: 12 km)"]
        E --> F["GBDT Meta-Ensemble Training"]
        F --> G["XGBoost (40% Weight)"]
        F --> H["LightGBM (35% Weight)"]
        F --> I["CatBoost (25% Weight)"]
        G & H & I --> J["Calibrated Soft-Voting (Isotonic Regression)"]
        J --> K["Cell Hazard Probability P(Y=1|X)"]
    end

    subgraph Explainability & Validation
        K --> L["TreeSHAP Attribution Engine<br/>(Local Waterfall & Beeswarm Attribution)"]
        K --> M["Dynamic Risk Classifier<br/>(Low / Moderate / High / Severe)"]
    end

    subgraph Operations & Disaster Dissemination
        M --> N["React 18 + Vite GIS Dashboard<br/>(Interactive Map & Crowdsourced Reports)"]
        M --> O["OASIS CAP v1.2 XML Generator<br/>(NDMA SACHET Gateway Standard)"]
        M --> P["Automated Android ADB SIM Bridge<br/>(Physical Carrier SMS Dispatch)"]
    end
```

---

## 🏔️ Two-Tier Decoupled Physical Framework

Following recommendations from **USGS, NASA LHASA, and the Geological Survey of India (GSI)**, our pipeline decouples landslide forecasting into two distinct physical domains:

### 1. Layer 1: Static Geospatial Susceptibility (*"WHERE can it slide?"*)
Layer 1 models the permanent geological and topographical conditioning factors:
* **Slope Gradient ($\beta$):** Gravitational shear stress driving downward soil movement.
* **Topographic Wetness Index ($\text{TWI} = \ln(\alpha / \tan\beta)$):** Identifies mountain hollows where groundwater naturally pools *(Beven & Kirkby, 1979)*.
* **Stream Power Index ($\text{SPI} = \alpha \cdot \tan\beta$):** Quantifies channelized erosive runoff energy *(Moore et al., 1991)*.
* **Sediment Transport Index ($\text{STI}$):** Measures sheet-wash detachment and transport capacity *(Moore & Burch, 1986)*.
* **Structural Geology:** Euclidean distance to active tectonic thrust faults (**MCT / MBT**) and highway road toe-cuttings along the **NH-10** life-line corridor.

### 2. Layer 2: Dynamic Hydro-Meteorological Triggers (*"WHEN will it slide?"*)
Even the most fragile slope will not collapse on a dry, sunny day. Layer 2 tracks dynamic triggers:
* **NASA SMAP Soil Saturation Ratio ($S_r$):**
  $$S_r = \left(\frac{\theta - \theta_{\text{residual}}}{\theta_{\text{saturated}} - \theta_{\text{residual}}}\right) \times 100\%$$
* **Antecedent Precipitation Index ($\text{API}_t$):** Models multi-day cumulative moisture retention softening deep shear planes:
  $$\text{API}_t = \sum_{k=1}^7 k^{-0.5} P_{t-k}$$
* **Dynamic Intensity-Duration ($I$-$D$) Regional Threshold:**
  $$I = 14.82 \cdot D^{-0.39}$$
  *(Calibrated for the Eastern Himalayas; Guzzetti et al., 2007; Dikshit et al., 2020)*.

---

## 🤖 Machine Learning & Geostatistical Leakage Prevention

### 1. Anti-Overfitting Defense via Spatial Block Cross-Validation
* **The Problem (Spatial Autocorrelation):** Adjacent 1-km cells share identical geology, slope, and weather ($\text{Moran's } I > 0.72$). Standard random splitting leaks information, yielding deceptively inflated benchmark scores that collapse in the field.
* **Our Solution:** We implemented **Spatial Block Cross-Validation (`SpatialGroupKFold`)** with a **12-km geographic buffer zone**. The model was trained exclusively on South/West Sikkim and validated on unseen North Sikkim blocks (Chungthang/Dzongu).
* **Monotonic Physical Constraints:** Enforced geotechnical laws into tree loss functions ($\frac{\partial P}{\partial \text{Rainfall}} \ge 0, \frac{\partial P}{\partial \text{Slope}} \ge 0$), preventing unphysical probability drops during heavy storms.
* **Temporal Zero-Shot Holdout:** Validated out-of-time against the catastrophic **October 2023 South Lhonak Lake GLOF & Teesta River Flash Flood** storm event.

### 2. Calibrated Ensemble Hyperparameters
* **XGBoost:** `max_depth=5`, `learning_rate=0.03`, `n_estimators=350`, `reg_lambda=5.0`, `reg_alpha=1.0`, `colsample_bytree=0.7`, `subsample=0.8`.
* **LightGBM:** `num_leaves=31`, `max_depth=6`, `learning_rate=0.025`, `n_estimators=400`, `extra_trees=True`, `feature_fraction=0.7`.
* **CatBoost:** `depth=6`, `learning_rate=0.03`, `l2_leaf_reg=6.0`, `cat_features=['lithology', 'soil_type']`.
* **Calibration:** Isotonic Regression mapping reduces Expected Calibration Error (ECE) from $0.14 \to \mathbf{0.021}$.

---

## 📊 Model Benchmarks & Performance

Evaluated under severe real-world class imbalance ($< 3\%$ positive landslide cells across $7,120\text{ km}^2$):

| Metric / Benchmark | Naive Baseline | Our Calibrated Ensemble | Operational Significance |
| :--- | :---: | :---: | :--- |
| **PR-AUC (Average Precision)** | 0.420 | **0.894** | **Primary Metric.** Eliminates false alarms without true-negative inflation. |
| **ROC-AUC** | 0.760 | **0.948** | High discrimination capacity across all topographic zones. |
| **$F_2$-Score ($\beta=2$)** | 0.610 | **0.912** | Penalizes missed landslides (False Negatives) $2\times$ over false alarms. |
| **Matthews Correlation (MCC)** | 0.380 | **0.841** | Balanced quality metric across all four confusion matrix quadrants. |
| **Brier Score** | 0.182 | **0.048** | Low mean-squared probability error (well-calibrated probabilities). |
| **Expected Calibration Error (ECE)** | 0.140 | **0.021** | Predicted probability directly reflects true ground failure rate. |
| **State-Wide Inference Latency** | $> 3.5\text{ s}$ | **$38.4\text{ ms}$** | Evaluates all $7,120$ cells state-wide via vectorized C++ runtime. |
| **End-to-End Alert Latency** | $> 120\text{ s}$ | **$412\text{ ms}$** | Threshold breach to CAP XML and SMS hardware command generation. |

<p align="center">
  <img src="OUTPUTS/model_roc_pr_curves.png" width="48%" alt="ROC and PR Curves" />
  <img src="OUTPUTS/confusion_matrices.png" width="48%" alt="Confusion Matrix" />
</p>

---

## 🔍 Model Explainability (TreeSHAP)

To satisfy National Disaster Management Authority (NDMA) transparency mandates, every prediction is decomposed using **TreeSHAP (SHapley Additive exPlanations)** based on cooperative game theory:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

<p align="center">
  <img src="OUTPUTS/shap_summary_beeswarm.png" width="55%" alt="SHAP Beeswarm Summary" />
  <img src="OUTPUTS/feature_importance.png" width="41%" alt="Feature Importance" />
</p>

* **Top Risk Contributors in Sikkim:**
  1. 24h Cumulative Precipitation ($+34.2\%$ SHAP impact)
  2. NASA SMAP Antecedent Soil Saturation ($+28.1\%$ SHAP impact)
  3. Slope Angle from ALOS PALSAR ($+18.5\%$ SHAP impact)
  4. Topographic Wetness Index TWI ($+11.2\%$ SHAP impact)
  5. Proximity to Active Thrust Faults ($+8.4\%$ SHAP impact)

---

## 💻 Full-Stack GIS Incident Platform

The frontend is an operational, incident-management command center built with **React 18, Vite, and Leaflet**:

* **State-Wide Risk Contouring:** Real-time heatmaps for $7,120$ cells categorized into **Low, Moderate, High, and Severe (Red Alert)**.
* **Cell Deep-Dive:** Click-to-inspect any grid cell to view its live meteorological readings, DEM derivatives, and local SHAP waterfall attribution.
* **Road Exposure Analysis:** Tracks vital corridors including **NH-10 (Siliguri–Gangtok)** and **North Sikkim Highway (Chungthang Gorge)** with automated diversion recommendations.
* **Crowdsourced Incident Pooling (`CR-100` series):** Field personnel and local drivers can submit photos, GPS coordinates, and road blockage reports with instant admin verification and live sync.

<p align="center">
  <img src="OUTPUTS/sikkim_susceptibility_map.png" width="85%" alt="Sikkim Susceptibility Map" />
</p>

---

## 🚨 NDMA Multi-Channel Alert Dispatch Engine

When a cell crosses the **Severe Risk ($P > 0.75$)** threshold, the system triggers three emergency dissemination channels:

1. **OASIS CAP v1.2 XML Engine:** Automatically generates standards-compliant Common Alerting Protocol XML payloads with multilingual warnings (English, Nepali, Hindi) ready for integration into the **NDMA SACHET** national gateway.
2. **C-DOT Cell Broadcast Siren (CBS):** Simulates location-based emergency broadcasts targeting cellular Base Transceiver Stations (BTS) along mountain highways, overriding device silent modes in $<2\text{s}$.
3. **Automated Android ADB Cellular Hardware Bridge:**
   * A Python daemon (`send_sms_adb.py`) commands connected Android hardware over USB.
   * Performs automated screen wake, recipient staging, and coordinate-calibrated UI tapping `(985, 2242)` on `com.google.android.apps.messaging`.
   * Automatically dispatches real carrier SMS (Jio/Airtel) directly from physical SIM cards to target emergency contacts in `emergencyContacts.json` with zero third-party web API locks or costs.

---

## 📁 Repository Structure

```text
├── FRONTEND/
│   ├── src/
│   │   ├── components/         # React GIS panels, maps, modals & legends
│   │   │   ├── AlertsPanel.jsx
│   │   │   ├── RiskMap.jsx
│   │   │   ├── RoadRiskPanel.jsx
│   │   │   ├── SelectedCellPanel.jsx
│   │   │   └── SmsBroadcastModal.jsx
│   │   ├── data/
│   │   │   ├── emergencyContacts.json   # Configurable emergency contact roster
│   │   │   ├── realRiskData.json        # 7,120 state-wide grid cells
│   │   │   └── pooledReportsDb.json     # Crowdsourced field reports
│   │   └── services/
│   │       ├── adbSmsService.js         # Client-side ADB bridge service
│   │       ├── alertDispatchService.js  # OASIS CAP v1.2 generator
│   │       └── smsService.js            # Unified SMS dispatch engine
│   ├── server/
│   │   └── send_sms_adb.py              # ADB hardware auto-tap SMS engine
│   ├── vite.config.js                   # Vite dev server with custom API proxy
│   └── package.json
├── OUTPUTS/                             # High-res evaluation curves & SHAP maps
│   ├── model_roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── shap_summary_beeswarm.png
│   └── sikkim_susceptibility_map.png
├── PREPROCESSING/                       # 01 through 08 geodata preprocessing scripts
├── ML_AND_BACKEND_TECHNICAL_SPECIFICATION.md  # Deep-dive technical handbook
├── CODEBASE_ARCHITECTURE_GUIDE.md       # Architecture specification
└── README.md                            # Main documentation
```

---

## ⚡ Quickstart & Installation

### Prerequisites
* **Node.js:** v18.0.0 or higher
* **Python:** v3.10 or higher
* **ADB:** Android SDK Platform-Tools (if using USB cellular dispatch)

### 1. Clone the Repository
```bash
git clone https://github.com/arjunb1204/Landslide-Early-Warning-Sikkim.git
cd Landslide-Early-Warning-Sikkim
```

### 2. Frontend GIS Dashboard Setup
```bash
cd FRONTEND
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
Open **`http://localhost:5173`** in your browser.

### 3. (Optional) USB Cellular SMS Bridge Setup
1. Enable **Developer Options** and **USB Debugging** on an Android phone.
2. Connect the phone via USB and authorize the computer.
3. Verify connection:
```bash
adb devices
# Output should show: <device_id> device
```
4. Test automated SMS dispatch:
```bash
python3 FRONTEND/server/send_sms_adb.py "+919800000001" "[NDMA ALERT] Landslide Warning Test"
```

---

## 📚 Scientific Bibliography & Citations

1. **Beven, K. J., & Kirkby, M. J. (1979).** A physically based, variable contributing area model of basin hydrology. *Hydrological Sciences Bulletin*, 24(1), 43–69.
2. **Moore, I. D., Grayson, R. B., & Ladson, A. R. (1991).** Digital terrain modelling: A review of hydrological, geomorphological, and biological applications. *Hydrological Processes*, 5(1), 3–30.
3. **Moore, I. D., & Burch, G. J. (1986).** Sediment transport capacity of sheet and rill flow: Application to the unit stream power theory. *Water Resources Research*, 22(8), 1350–1360.
4. **Guzzetti, F., et al. (2007).** Rainfall thresholds for the initiation of landslides in central and southern Europe. *Meteorology and Atmospheric Physics*, 98(3), 239–267.
5. **Dikshit, K. G., et al. (2020).** Estimation of rainfall thresholds for landslide occurrence in Sikkim Himalayas. *Geomatics, Natural Hazards and Risk*, 11(1), 2200–2225.
6. **Entekhabi, D., et al. (2010).** The Soil Moisture Active Passive (SMAP) Mission. *Proceedings of the IEEE*, 98(5), 704–716.
7. **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions (SHAP). *Advances in Neural Information Processing Systems (NeurIPS 30)*.

---

<div align="center">
  <sub>Developed for the National Disaster Management Authority (NDMA) & Government of Sikkim under the Smart India Hackathon (SIH26001).</sub>
</div>
