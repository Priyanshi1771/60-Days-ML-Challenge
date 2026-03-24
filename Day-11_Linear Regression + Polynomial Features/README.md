<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0D47A1&height=250&section=header&text=Day%2011%20%E2%80%94%20ICU%20Mortality%20Prediction&fontSize=38&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%F0%9F%8F%A5%20Polynomial%20Regression%20%2B%20GPU%20Neural%20Net&descSize=18&descAlignY=55&descColor=90CAF9" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=42A5F5&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%8F%A5+Phase+2+Begins%3A+Regression+%26+Time-Series;%F0%9F%93%88+Polynomial+Features+%2B+Regression+Metrics;%E2%9A%A1+GPU-Accelerated+Neural+Net+Regressor" alt="Typing SVG" /></a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Day](https://img.shields.io/badge/Day-11%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)

<br/>

[![Phase2](https://img.shields.io/badge/Phase%202-Regression%20Begins!-00C853?style=flat-square&logo=rocket&logoColor=white)](#)
[![Regression](https://img.shields.io/badge/Task-Regression-FF6F00?style=flat-square&logo=chartdotjs&logoColor=white)](#)
[![GPU](https://img.shields.io/badge/GPU-Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)](#)
[![Polynomial](https://img.shields.io/badge/Technique-Polynomial%20Features-AB47BC?style=flat-square&logo=databricks&logoColor=white)](#)

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

<br/>

## 🏥 Project Overview

> **Predict ICU patient mortality risk score from 22 clinical features using Polynomial Regression and a GPU-accelerated Neural Net — the first regression task in the 60-day challenge.**

Phase 2 shifts from **classification** (Days 1-10) to **regression** — predicting continuous values instead of categories. ICU mortality risk is a continuous score (0.0 = safe → 1.0 = critical), making it the perfect entry point for regression techniques.

<div align="center">

```
🏥 ICU Patient Data → 📈 Regression Models → 🎯 Risk Score (0.0 — 1.0)

  👤 Demographics        💉 Vitals              🧪 Labs
  ─────────────         ────────              ────────
   Age                   Heart Rate             BUN / Creatinine
   Prev ICU Stays        BP (Sys/Dia)           Sodium / Potassium
   LOS Before ICU        Resp Rate              Hemoglobin / WBC
                          Temperature            Platelets / Lactate
  🔧 Interventions       SpO₂ / GCS             PaO₂/FiO₂ ratio
  ─────────────         Urine Output
   Ventilator (Y/N)
   Vasopressor (Y/N)         ↓
                         📈 Mortality Risk
                         ═══════════════
                          0.0 ════════ 1.0
                          Safe        Critical
```

</div>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📈 Key Learning: Polynomial Features

### 🔢 How Polynomial Features Work

```
Original features:    [age, lactate]

Degree 1 (Linear):   [age, lactate]                    → 2 features
Degree 2 (Quadratic): [age, lactate, age², age·lactate, lactate²]  → 5 features
Degree 3 (Cubic):    [...all above + age³, age²·lactate, ...]      → 9 features

With 22 original features:
  Degree 1 →     22 features
  Degree 2 →    275 features   (12× explosion!)
  Degree 3 →  2,324 features   (105× explosion! 💥)
```

### 🎯 Why This Matters

| Concept | Detail |
|:--------|:-------|
| **Linear regression can't curve** | y = w₁·age + w₂·lactate misses age² and age×lactate effects |
| **Poly features add curves** | y = w₁·age + w₂·lactate + w₃·age² + w₄·age·lactate captures nonlinearity |
| **Feature explosion** | 22 features → 2,324 at degree 3 — regularization (Ridge) is mandatory |
| **Overfitting risk** | More features than samples → model memorizes noise |
| **Sweet spot** | Usually degree 2 — captures interactions without explosion |

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📊 Regression Metrics Explained

> **Classification uses Accuracy/F1. Regression uses MAE/RMSE/R².**

| Metric | Formula | Interpretation |
|:-------|:--------|:---------------|
| **MAE** | mean(\|actual - predicted\|) | Average error magnitude. Easy to interpret. |
| **RMSE** | √mean((actual - predicted)²) | Penalizes large errors MORE than MAE |
| **R²** | 1 - SS_res/SS_total | % of variance explained (1.0 = perfect, 0.0 = useless) |
| **MSE** | mean((actual - predicted)²) | Raw squared error. Used as loss function. |

```
Example:
  Patient actual risk = 0.75
  Model predicts      = 0.68
  
  Error = |0.75 - 0.68| = 0.07     ← MAE contribution
  Error² = 0.07² = 0.0049          ← MSE contribution
  
  If another patient has error = 0.20:
  MAE contribution = 0.20           (2.9× the first)
  MSE contribution = 0.04           (8.2× the first! RMSE punishes more)
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 🧠 Models

<div align="center">

```mermaid
graph LR
    A["📥 22 ICU Features"] --> B["🔢 Poly Features<br/>Degree 1/2/3"]
    A --> E["🧠 GPU Neural Net<br/>22→128→64→1"]
    B --> C["📈 Ridge Regression<br/>(L2 regularized)"]
    C --> D["🎯 Risk Score"]
    E --> D

    style A fill:#1a1b27,stroke:#42A5F5,color:#fff
    style B fill:#1a1b27,stroke:#AB47BC,color:#fff
    style C fill:#1a1b27,stroke:#66BB6A,color:#fff
    style D fill:#1a1b27,stroke:#EF5350,color:#fff
    style E fill:#1a1b27,stroke:#FF7043,color:#fff
```

</div>

| Model | Type | Features | Why |
|:------|:-----|:---------|:----|
| **Linear (Degree 1)** | Ridge Regression | 22 | Interpretable baseline |
| **Poly Degree 2** | Ridge + PolynomialFeatures | 275 | Captures interactions + quadratics |
| **Poly Degree 3** | Ridge + PolynomialFeatures | 2,324 | Tests feature explosion limits |
| **GPU Neural Net** | PyTorch 2-layer MLP | 22 (raw) | Learns nonlinearity automatically |

<br/>

## 📊 Dataset

| Property | Detail |
|:---------|:-------|
| **Inspired by** | MIMIC-IV ICU database |
| **Samples** | 4,200 ICU patients |
| **Features** | 22 (vitals, labs, demographics, interventions) |
| **Target** | Mortality risk score (continuous, 0.0 — 1.0) |
| **Nonlinearity** | Target includes interaction terms (age×lactate) + polynomial (age²) |
| **Noise** | Gaussian noise added for realism |

<br/>

## 🏗️ Project Structure

```
day11_icu_mortality/
├── 📄 main.py              ← Entry point
├── 📄 config.py             ← Poly degrees, NN arch, GPU device
├── 📄 data_pipeline.py      ← ICU data generation + preprocessing
├── 📄 model_training.py     ← Poly regression + GPU neural net
├── 📄 evaluation.py         ← Regression metrics, residuals, comparison
├── 📄 README.md
├── 📁 data/    ├── 📁 models/    ├── 📁 plots/
├── 📁 logs/    └── 📁 outputs/
```

<br/>

## ⚡ Quick Start

```bash
cd day11_icu_mortality
python main.py
```

**Pipeline:**
1. 🏥 Generate 4,200 ICU patients (22 features + nonlinear target)
2. 📊 EDA: target distribution + top correlated features
3. 🔢 Train Poly(1), Poly(2), Poly(3) Ridge regression with CV
4. 🧠 Train GPU neural net (128→64→1) with AMP + early stopping
5. 📈 Evaluate all: MAE, RMSE, R² + residual analysis
6. 🔬 Feature importance from linear regression coefficients

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📈 Generated Visualizations

| # | Plot | What It Shows |
|:-:|:-----|:-------------|
| 01 | EDA Overview | Target distribution + top 5 feature correlations |
| 02 | Poly Comparison | RMSE, R², feature count across degrees 1/2/3 |
| 03 | NN Training | Train vs val loss curves over epochs |
| 04 | Predictions | Actual vs predicted scatter + residual plots (4-panel) |
| 05 | Model Comparison | R² bar chart for all models |
| 06 | Coefficients | Linear regression weights (which features drive risk) |

<br/>

## ⚡ GPU Optimizations

| Optimization | Where | Impact |
|:-------------|:------|:-------|
| `float32` everywhere | data + model | Standard GPU dtype |
| `autocast` (AMP) | NN training + eval | Mixed precision on GPU |
| `GradScaler` | NN training | Prevents FP16 underflow |
| `set_to_none=True` | zero_grad | Faster than zeroing |
| `non_blocking=True` | .to(device) | Async transfer |
| `drop_last=True` | Train DataLoader | BatchNorm stability |
| `ReduceLROnPlateau` | Scheduler | Auto LR decay |
| Early stopping (patience=10) | Training loop | Save compute |
| `n_jobs=-1` | sklearn CV | Parallel CPU for poly models |
| `compress=3` joblib | Model saving | Smaller files |
| `rasterized=True` | Scatter plots | Smaller plot files |

<br/>

## 🩺 Clinical Significance

> **ICU mortality prediction scores help doctors triage patients — who needs the most aggressive intervention? A well-calibrated risk score (not just "high/low") enables proportional care allocation.**

The residual analysis reveals systematic prediction errors — e.g., if the model consistently under-predicts risk for elderly patients, that's a dangerous blind spot that needs fixing before deployment.

<br/>

## 💡 Lessons Learned

| Lesson | Detail |
|:-------|:-------|
| **Poly degree 2 sweet spot** | Captures interactions without massive feature explosion |
| **Ridge is mandatory** | Without L2 regularization, poly degree 3 overfits catastrophically |
| **Feature explosion** | 22 → 2,324 features at degree 3 — curse of dimensionality |
| **Neural nets skip poly** | MLP learns nonlinearity automatically — no manual feature engineering |
| **RMSE > MAE for safety** | In ICU, large errors are disproportionately dangerous |
| **Residual plots matter** | They reveal patterns that aggregate metrics (R²) hide |
| **GPU speeds up NN** | But sklearn poly regression is CPU-only (still fast at this scale) |

<br/>

## 📦 Dependencies

```bash
numpy>=1.24
torch>=2.0
scikit-learn>=1.3
matplotlib>=3.7
pandas>=2.0
joblib>=1.3
```

<br/>

## 🔗 Part of 60 Days of ML & DL Challenge

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 10: Malaria CNN](../day10_malaria_classification/) | **🏥 Day 11: ICU Mortality** | [Day 12: Blood Pressure](../day12_blood_pressure/) |
| 🎉 First CNN! | Polynomial Regression + GPU NN | Ridge Regression + Multicollinearity |

</div>

<br/>

<div align="center">

```
╔════════════════════════════════════════════════════╗
║                                                    ║
║   📈 PHASE 2: REGRESSION & TIME-SERIES             ║
║   Day 11 of 10 — Now predicting NUMBERS,           ║
║   not categories!                                   ║
║                                                    ║
╚════════════════════════════════════════════════════╝
```

</div>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0D47A1&height=150&section=footer&animation=twinkling" width="100%"/>

<br/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=16&duration=4000&pause=2000&color=42A5F5&center=true&vCenter=true&repeat=true&width=500&lines=%F0%9F%8F%A5+Predicting+risk+scores+%7C+Saving+ICU+lives" alt="Footer" /></a>

</div>
