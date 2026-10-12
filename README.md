# ✈️ Predictive Maintenance AI: Turbofan Engine RUL Estimation
### Remaining Useful Life (RUL) Prediction using NASA C-MAPSS (FD004) Dataset

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![NASA C-MAPSS](https://img.shields.io/badge/NASA-C--MAPSS_FD004-E03C31?style=flat&logo=nasa&logoColor=white)](https://data.nasa.gov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 📌 1. Project Overview

In commercial and defense aviation, in-flight turbofan engine failures present critical safety hazards and cause millions of dollars in unscheduled downtime, emergency overhauls, and canceled flights. Traditional maintenance strategies—either **reactive** (repair after breakdown) or **schedule-based** (fixed-hour overhauls)—are either dangerous or economically wasteful.

This project develops an end-to-end **Predictive Maintenance (PdM)** machine learning system to predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines. By analyzing multivariate sensor streams across operational flight cycles, the system enables **Condition-Based Maintenance (CBM)**, allowing fleet managers and maintenance crews to service components proactively before failure occurs.

### 🎯 Key Objectives:
- **Prevent Unscheduled Downtime:** Identify degrading engine units well ahead of operational failure.
- **High-Precision in Critical Stages:** Deliver exceptional predictive accuracy when engines approach the danger threshold ($RUL \le 30$ cycles).
- **Maintenance Planning:** Forecast parts replacement timelines to streamline hangar scheduling and spare-parts supply chains.

---

## 📊 2. Dataset Architecture (NASA C-MAPSS FD004)

The project utilizes the benchmark **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, specifically subset **FD004**, widely regarded as the most challenging and realistic scenario:

| Parameter | Description |
| :--- | :--- |
| **Dataset Source** | NASA Prognostics Center of Excellence (PCoE) |
| **Subset** | **FD004** (Multi-condition & multi-fault scenario) |
| **Total Observations** | 61,249 operational cycle records |
| **Total Engine Units** | 249 distinct turbofan engines |
| **Total Variables** | 26 columns (24 feature signals + Engine ID + Cycle) |
| **Operating Conditions** | **6 distinct flight regimes** (Altitude: 0–42k ft, Mach: 0–0.84, Throttle: 60–100°) |
| **Fault Modes** | **2 fault modes**: High-Pressure Compressor (HPC) wear + Fan degradation |
| **Data Quality** | 0 missing values, 0 duplicate records |

### Data Attributes:
- **`unit_number`:** Unique identifier for each aircraft engine (1 through 249).
- **`time_in_cycles`:** Cumulative operational cycles elapsed since commissioning.
- **`setting_1`, `setting_2`, `setting_3`:** Environmental flight envelope parameters (Altitude, Mach number, Throttle Resolver Angle).
- **`sensor_1` to `sensor_21`:** Telemetry signals including total temperatures, static/dynamic pressures, spool physical and corrected rotational speeds, bypass ratios, and coolant bleed fractions.

---

## ⚙️ 3. Methodology & Feature Engineering

### 1️⃣ True RUL Calculation (Run-to-Failure)
Because each engine is monitored continuously until total functional failure, the ground-truth RUL at any cycle $t$ is calculated backward from the final observed cycle:
$$\text{RUL}_t = \max(\text{time\_in\_cycles}) - \text{time\_in\_cycles}_t$$
At the failure cycle, $\text{RUL} = 0$. The maximum RUL in the raw data reaches 542 cycles.

### 2️⃣ Piecewise Linear Degradation Modeling (RUL Capping at 125 Cycles)
During early operational stages, components do not exhibit measurable wear, and sensor measurements remain indistinguishable from baseline healthy operation. Expecting a model to distinguish between 450 and 350 cycles remaining introduces noise and harms model accuracy.

Following aviation prognostics best practices, a **Piecewise Linear Degradation Model** was applied by capping RUL at **125 cycles**:
$$\text{RUL}_{\text{capped}} = \min(\text{RUL}, 125)$$

> 💡 **Impact:** Target capping reduced the Random Forest **Mean Absolute Error (MAE) from 43.41 cycles down to 14.20 cycles**, drastically improving real-world clinical fidelity.

### 3️⃣ Group-Based Splitting & Data Leakage Prevention
Because records from the same engine unit are sequential time-series, standard random row splitting causes severe **data leakage** (training and testing on different cycles of the same engine).

To ensure realistic, out-of-sample generalization:
- **`GroupShuffleSplit`** was enforced using `unit_number` as the grouping key.
- **Zero Engine Overlap (`Overlap = 0`):** All records of any given engine are placed exclusively in Train, Validation, or Test.
- **Final Independent Test Set:** 50 unseen engine units (11,955 cycles; 20% of the entire fleet).

---

## 📈 4. Visual Analysis & Embedded Figures

All figures below are generated directly from the analysis and model evaluations in [`Predictive Maintenance.ipynb`](./Predictive%20Maintenance.ipynb).

---

### Figure 1: Sensor 9 Readings Over Operating Cycles (Engine 1)
Sensor 9 (Total Temperature / Pressure telemetry) tracked across the entire operational lifespan of Engine 1 (from cycle 1 to failure at cycle 321). The distinct horizontal bands reflect the 6 different flight operating conditions, while the overall upward drift indicates thermodynamic degradation over time.

<p align="center">
  <img src="./Sensor%209%20Readings%20Over%20Operating%20Cycles%20-%20Engine%201.png" alt="Sensor 9 Readings Over Operating Cycles - Engine 1" width="750"/>
</p>

---

### Figure 2: Sensor 9 vs. Remaining Useful Life (RUL) Across All Engines
Scatter plot of Sensor 9 readings against the Remaining Useful Life (RUL) across all 61,249 records. As the engine approaches failure ($RUL \to 0$), sensor values exhibit increased dispersion and upward deviation.

<p align="center">
  <img src="./Sensor%209%20vs%20Remaining%20Useful%20Life%20(RUL).png" alt="Sensor 9 vs Remaining Useful Life (RUL)" width="750"/>
</p>

---

### Figure 3: Sensor 14 vs. Remaining Useful Life (Engine 1)
Telemetry from Sensor 14 plotted against RUL for Engine 1. A clear negative correlation demonstrates how specific internal pressure/speed ratios decrease predictably as useful operating life diminishes.

<p align="center">
  <img src="./Sensor%2014%20vs%20RUL%20-%20Engine%201.png" alt="Sensor 14 vs RUL - Engine 1" width="750"/>
</p>

---

### Figure 4: Top 10 Most Predictive Sensor Features
Feature importance distribution extracted from the Random Forest Regressor. Five key sensors (`sensor_13`, `sensor_11`, `sensor_15`, `sensor_4`, `sensor_12`) account for over 75% of the total predictive power.

<p align="center">
  <img src="./Top%2010%20Important%20Features%20for%20RUL%20Prediction.png" alt="Top 10 Important Features for RUL Prediction" width="750"/>
</p>

---

### Figure 5: Final Model Evaluation — Predicted vs. Actual RUL
Scatter plot comparing the model's predicted RUL against actual RUL on the **50 unseen holdout test engines** (11,955 cycles). The red dashed diagonal represents perfect prediction ($y = x$). The model exhibits tight adherence to the ideal line, particularly in the lower RUL ranges ($0 \le \text{RUL} \le 30$).

<p align="center">
  <img src="./Final%20Random%20Forest.png" alt="Final Random Forest: Actual vs Predicted RUL" width="750"/>
</p>

---

## 🔬 5. Sensor Degradation & Feature Importance

Feature importance values extracted from the trained ensemble model:

| Feature | Importance | Sensor Function / Physical Meaning |
| :--- | :---: | :--- |
| **`sensor_13`** | **39.13%** | **Core Speed / Corrected Fan Speed:** Primary indicator of aerodynamic drag and mechanical resistance inside the core shaft. |
| **`sensor_11`** | **14.84%** | **HPC Outlet Static Pressure:** Direct measure of High-Pressure Compressor efficiency loss and blade tip leakage. |
| **`sensor_15`** | **12.79%** | **Bypass Ratio:** Measures fan bypass air ratio; shifts indicate secondary flow degradation and aerodynamic losses. |
| **`sensor_4`** | **5.73%** | **LPT Outlet Total Temperature:** Elevated exhaust temperature due to lower thermal extraction efficiency in worn turbines. |
| **`sensor_12`** | **4.37%** | **Fuel Flow / Pressure Ratio:** Fuel flow compensation required to maintain requested thrust under degraded states. |
| **Sensors 8, 14, 9, 7, 2** | ~15% | Physical spool speeds, coolant bleed fraction, and internal compressor pressure ratios. |

---

## 🏆 6. Model Evaluation & Benchmark Comparison

Models were evaluated using standard regression error metrics:
- **MAE** (Mean Absolute Error, in cycles)
- **RMSE** (Root Mean Squared Error, in cycles)

### 📊 Model Performance Matrix:

| Model | Features | MAE (Cycles) | RMSE (Cycles) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Dummy Baseline (Uncapped)** | 24 | 72.48 | 88.75 | Baseline estimating global mean |
| **Dummy Baseline (Capped at 125)** | 24 | 35.58 | 40.89 | Capped mean baseline |
| **Random Forest (Top 3 Sensors)** | 3 | 16.69 | 23.56 | Ultra-lightweight edge model |
| **Gradient Boosting Regressor** | 24 | 15.89 | 21.60 | Strong gradient boosted trees baseline |
| **Random Forest (Top 5 Sensors)** | 5 | 15.50 | 21.78 | Optimal lightweight-to-accuracy balance |
| **🏆 Random Forest Regressor (Full)** | **24** | **14.20** | **20.11** | **Best Performing Model** |

### 🎯 Final Evaluation on Unseen Test Engines (50 Holdout Engines):
After combining train and validation splits to retrain the selected model:
- **Final Out-of-Sample Test MAE:** **`14.21` cycles**
- **Final Out-of-Sample Test RMSE:** **`20.09` cycles**

---

## 🚨 7. Life-Stage Error Breakdown (Aviation Safety Analysis)

To evaluate real-world flight operations viability, test engines were grouped into three operational life stages based on their actual remaining life:

| Life Stage | Actual RUL Range | Sample Count | MAE (Cycles) | Mean Error (Cycles) | Operational Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🔴 **Critical Stage** | **0 – 30 cycles** | **1,550** | **8.97** | **+7.04** | **Immediate maintenance required; prevents in-flight failure.** |
| 🟡 **Middle Stage** | **31 – 75 cycles** | **2,250** | **21.99** | **+14.33** | Early alert; procure spares and schedule upcoming inspection. |
| 🟢 **Healthy Stage** | **76 – 125 cycles** | **8,155** | **13.06** | **-8.58** | Engine is healthy; proceed with routine flights. |

> ⭐ **Safety Insight:** The model achieves its highest precision (**MAE = 8.97 cycles**) in the **Critical Stage (0–30 cycles)**. This behavior is ideal for aviation engineering: the model delivers maximum fidelity exactly when the risk of failure is highest.

---

## 📁 8. Repository Structure

```text
Predictive Maintenance AI/
│
├── Predictive Maintenance.ipynb                      # 📓 Comprehensive Jupyter Notebook (EDA, ML Pipeline, Evaluation)
├── Final Random Forest.png                           # 📊 Figure 5: Actual vs. Predicted RUL on holdout engines
├── Top 10 Important Features for RUL Prediction.png  # 📊 Figure 4: Top 10 sensor feature importances
├── Sensor 9 Readings Over Operating Cycles - Engine 1.png # 📊 Figure 1: Sensor 9 trajectory over engine life
├── Sensor 9 vs Remaining Useful Life (RUL).png       # 📊 Figure 2: Sensor 9 correlation with RUL
├── Sensor 14 vs RUL - Engine 1.png                   # 📊 Figure 3: Sensor 14 degradation behavior
└── README.md                                         # 📖 Project Documentation
```

---

## 🚀 9. Getting Started & How to Run

### Prerequisites:
Python 3.9+ environment with the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Running the Notebook:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/sajaalkhatib/Predictive-Maintenance-AI.git
   cd Predictive-Maintenance-AI
   ```

2. **Add Dataset:**
   Ensure `train_FD004.txt` is located in the root project folder (downloadable from NASA PCoE C-MAPSS dataset repository).

3. **Launch Jupyter:**
   ```bash
   jupyter notebook "Predictive Maintenance.ipynb"
   ```

4. **Execute all cells** to reproduce data processing, plots generation, cross-validation, and holdout testing.

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.9+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Model Evaluation:** Scikit-Learn (`RandomForestRegressor`, `GradientBoostingRegressor`, `GroupShuffleSplit`, `metrics`)
- **Data Visualization:** Matplotlib
- **Notebook Environment:** Jupyter Notebook

---

## 💡 Future Roadmap

- [ ] **Deep Learning Models:** Implement temporal sequence models (Bidirectional LSTM, Transformer, or Temporal Convolutional Networks).
- [ ] **Operating Regime Normalization:** Cluster the 6 operating flight regimes using k-Means / GMM to normalize sensor data per regime.
- [ ] **Real-Time Fleet Dashboard:** Build an interactive Streamlit or Power BI dashboard for continuous fleet health monitoring and automated alerts.

---

## 👤 Author

**Saja Al-Khatib**  
- GitHub: [@sajaalkhatib](https://github.com/sajaalkhatib)  
- Repository: [Predictive-Maintenance-AI](https://github.com/sajaalkhatib/Predictive-Maintenance-AI)  
- License: [MIT License](LICENSE)
