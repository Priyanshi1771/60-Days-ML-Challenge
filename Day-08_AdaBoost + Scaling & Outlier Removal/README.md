<div align="center">

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- 🩸 ANIMATED HEADER — ANEMIA / BLOOD THEME 🔬 -->
<!-- ═══════════════════════════════════════════════════════════════ -->

<img src="https://capsule-render.vercel.app/api?type=waving&color=B71C1C&height=250&section=header&text=Day%2008%20%E2%80%94%20Anemia%20Detection&fontSize=40&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=%F0%9F%A9%B8%20AdaBoost%20%2B%20Scaling%20%26%20Outlier%20Removal&descSize=18&descAlignY=55&descColor=EF9A9A" width="100%"/>

<!-- ═══════════════ ANIMATED TYPING ═══════════════ -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=EF5350&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%A9%B8+Detecting+Anemia+from+Blood+Tests;%F0%9F%A7%AA+Ablation+Study%3A+4+Outlier+%C3%97+4+Scaling+Methods;%E2%9A%A1+AdaBoost%3A+Weak+Learners+%E2%86%92+Strong+Ensemble" alt="Typing SVG" /></a>

<br/>

<!-- ═══════════════ BADGES ═══════════════ -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Day](https://img.shields.io/badge/Day-08%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge&logo=checkmarx&logoColor=white)](#)

<br/>

[![AdaBoost](https://img.shields.io/badge/Model-AdaBoost-EF5350?style=flat-square&logo=probot&logoColor=white)](#-key-learning-adaboost)
[![Outliers](https://img.shields.io/badge/Focus-Outlier%20Removal-FF6F00?style=flat-square&logo=alertmanager&logoColor=white)](#-outlier-removal-methods)
[![Scaling](https://img.shields.io/badge/Focus-Feature%20Scaling-4FC3F7?style=flat-square&logo=chartdotjs&logoColor=white)](#-scaling-strategies)
[![Ablation](https://img.shields.io/badge/Study-16%20Combos-AB47BC?style=flat-square&logo=experiment&logoColor=white)](#-ablation-study)

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

<br/>

## 🩸 Project Overview

> **Detect anemia from Complete Blood Count (CBC) results using AdaBoost, with a systematic ablation study comparing 4 outlier removal methods × 4 scaling strategies = 16 preprocessing pipelines.**

Anemia affects ~1.6 billion people globally — it's the most common blood disorder. A CBC blood test is cheap and fast, but interpreting results across multiple correlated biomarkers is where ML excels. This project focuses on the often-overlooked but critical preprocessing step: how outliers and feature scaling affect model performance.

<div align="center">

```
🩸 Complete Blood Count (CBC)
═════════════════════════════════════════════════════
  
  🔴 Red Cell Markers           🔬 Cell Indices
  ─────────────────           ─────────────
   Hemoglobin (g/dL)           MCV (fL)     ← Cell volume
   RBC Count (M/μL)            MCH (pg)     ← Hemoglobin/cell
                                MCHC (g/dL)  ← Concentration
  
  ⚪ White Cells                🟡 Platelets
  ─────────────                ──────────
   WBC Count (K/μL)             Platelet Count (K/μL)
  
  ═════════════════════════════════════════════════════
  🎯 Target: Hemoglobin < 12-13 g/dL = ANEMIC
```

</div>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## ⚡ Key Learning: AdaBoost

<div align="center">

```mermaid
graph TD
    A["🌱 Round 1<br/>Train weak learner on data"] --> B["❌ Find misclassified samples"]
    B --> C["⚖️ Increase their weights"]
    C --> D["🌱 Round 2<br/>Train NEW learner on reweighted data"]
    D --> E["❌ Find NEW mistakes"]
    E --> F["⚖️ Increase weights again"]
    F --> G["🔄 Repeat N times"]
    G --> H["🏆 Final = Weighted vote<br/>of ALL weak learners"]

    style A fill:#1a1b27,stroke:#66BB6A,color:#fff
    style B fill:#1a1b27,stroke:#EF5350,color:#fff
    style C fill:#1a1b27,stroke:#FFB74D,color:#fff
    style D fill:#1a1b27,stroke:#66BB6A,color:#fff
    style E fill:#1a1b27,stroke:#EF5350,color:#fff
    style F fill:#1a1b27,stroke:#FFB74D,color:#fff
    style G fill:#1a1b27,stroke:#4FC3F7,color:#fff
    style H fill:#1a1b27,stroke:#AB47BC,color:#fff
```

</div>

### 🧠 How AdaBoost Differs from Other Ensembles

| Method | Strategy | Key Idea |
|:-------|:---------|:---------|
| **AdaBoost** | Sequential, adaptive | Each learner fixes MISTAKES of the previous one |
| Random Forest | Parallel, bagging | Each tree sees random data subset independently |
| XGBoost | Sequential, gradient | Each tree fits the RESIDUAL error (gradient) |

### 🎛️ AdaBoost Hyperparameters

| Parameter | Values Tested | Effect |
|:----------|:-------------|:-------|
| **`n_estimators`** | 50, 100, 200, 300, 500 | More learners = more complex (risk overfitting) |
| **`learning_rate`** | 0.001, 0.01, 0.05, 0.1, 0.5, 1.0 | Shrinks each learner's contribution (lower = more robust) |

> **30 combinations** × 10-fold CV = **300 fits** in GridSearch

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 🧪 Ablation Study

> **The core experiment: systematically test every combination of outlier removal and scaling to find the optimal preprocessing pipeline.**

### 🔍 Outlier Removal Methods

| Method | How It Works | Best For |
|:-------|:------------|:---------|
| **None** | Keep all data | When outliers are real extreme cases |
| **IQR** | Remove if outside Q1-1.5×IQR to Q3+1.5×IQR | Normal-ish distributions, simple |
| **Z-Score** | Remove if \|z\| > 3 standard deviations | Assumes normality |
| **Isolation Forest** | ML anomaly detection (no distribution assumption) | Multivariate outliers |

### 📏 Scaling Strategies

| Method | Formula | Best For |
|:-------|:--------|:---------|
| **None** | Raw values | Tree-based models (don't need scaling) |
| **StandardScaler** | (x - mean) / std | Normal distributions |
| **MinMaxScaler** | (x - min) / (max - min) | Bounded features |
| **RobustScaler** | (x - median) / IQR | **Data WITH outliers** — uses median, not mean! |

### 🗺️ Ablation Heatmap (4 × 4 = 16 combos)

```
                    none    standard   minmax    robust
              ┌─────────┬──────────┬─────────┬─────────┐
  none        │  0.9511  │  0.9511  │  0.9511 │  0.9511 │
  iqr         │  0.9486  │  0.9486  │  0.9486 │  0.9486 │
  zscore      │  0.9529  │  0.9529  │  0.9529 │ 🏆0.9529│
  iso_forest  │  0.9512  │  0.9512  │  0.9512 │  0.9512 │
              └─────────┴──────────┴─────────┴─────────┘
              
  Key insight: Scaling has NO effect on AdaBoost (tree-based!)
  but Z-score outlier removal helps the most (+0.18%)
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📊 Dataset

| Property | Detail |
|:---------|:-------|
| **Source** | Kaggle — Anemia Detection Dataset |
| **Samples** | 1,421 patients |
| **Features** | 8 (Gender + 7 CBC biomarkers) |
| **Target** | Binary: Anemic (38%) vs Not Anemic (62%) |
| **Outliers** | ~3% injected lab errors (realistic extreme values) |
| **Label Noise** | ~2% diagnostic disagreement |

### 🔬 Feature Discriminability

```
  Hemoglobin      ████████████████████  CRITICAL  ← Primary diagnostic marker
  MCH             ████████████████░░░░  HIGH      ← Hemoglobin per red cell
  MCHC            ██████████████░░░░░░  HIGH      ← Concentration
  MCV             █████████████░░░░░░░  MEDIUM    ← Cell volume (micro/macro)
  RBC Count       ████████████░░░░░░░░  MEDIUM    ← Red cell quantity
  Platelet Count  ████░░░░░░░░░░░░░░░░  LOW       ← Reactive thrombocytosis
  WBC Count       ███░░░░░░░░░░░░░░░░░  LOW       ← Minimal signal
  Gender          ██████░░░░░░░░░░░░░░  MODERATE  ← Different thresholds M/F
```

<br/>

## 🏗️ Project Structure

```
day08_anemia_detection/
├── 📄 main.py                ← Entry point
├── 📄 config.py              ← AdaBoost grid, outlier/scaling method lists
├── 📄 data_pipeline.py       ← CBC data, outlier detection, scaling functions
├── 📄 model_training.py      ← Ablation study + AdaBoost GridSearch + baselines
├── 📄 evaluation.py          ← Metrics, confusion matrices, ROC, error analysis
├── 📄 README.md              ← You are here
├── 📁 data/                  ├── 📁 models/
├── 📁 plots/                 ├── 📁 logs/
└── 📁 outputs/               ← Results CSV + ablation CSV + report
```

<br/>

## ⚡ Quick Start

```bash
cd day08_anemia_detection
python main.py
```

**Pipeline (5 phases):**
1. 🩸 Load CBC blood test data (1,421 patients)
2. 🧪 Ablation study: 4 outlier × 4 scaling = 16 combos tested
3. ⚡ AdaBoost GridSearchCV with best preprocessing
4. 📊 Train 4 baselines (LR, RF, SVM, DT) for comparison
5. 📈 Full evaluation + error analysis + 7 plots saved

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📈 Generated Visualizations

| # | Plot | What It Shows |
|:-:|:-----|:-------------|
| 01 | EDA Distributions | Class balance + all 7 CBC features by anemia status |
| 02 | Boxplots | Outliers visible as red dots for each feature |
| 03 | Ablation Heatmap | 4×4 grid: outlier method × scaling method → F1 |
| 04 | AdaBoost Landscape | n_estimators × learning_rate + overfitting check |
| 05 | Confusion Matrices | All 5 models side by side |
| 06 | ROC Curves | AUC comparison across all models |
| 07 | Model Comparison | Bar chart: AdaBoost vs all baselines |

<br/>

## 🔬 Models Compared

| Model | Role | Test F1 |
|:------|:-----|:-------:|
| **Random Forest** 🥇 | Tree ensemble | ~0.979 |
| **AdaBoost (Tuned)** 🥈 | Primary — sequential boosting | ~0.961 |
| Logistic Regression 🥉 | Linear baseline | ~0.958 |
| SVM (RBF) | Non-linear baseline | ~0.954 |
| Decision Tree | Interpretable baseline | ~0.943 |

<br/>

## 🧠 Engineering Principles

```
✅ No Data Leakage        → Scaler + outlier detection fit on TRAIN ONLY
✅ Stratified Splits       → Anemic ratio preserved in both sets
✅ Ablation Study          → Systematic comparison of ALL preprocessing combos
✅ 10-Fold CV              → Robust model selection
✅ Multiple Metrics        → Accuracy + F1 + Precision + Recall + AUC-ROC
✅ Error Analysis          → FN = missed anemia patients!
✅ Full Logging            → Timestamped, persistent
✅ Modular Code            → Clean 5-file separation
✅ Save Everything         → Models + scaler + grid results + ablation data
```

<br/>

## 💡 Lessons Learned

| Lesson | Detail |
|:-------|:-------|
| **Scaling doesn't affect trees** | AdaBoost uses decision stumps — splits are threshold-based, scale-invariant |
| **Outlier removal helps** | Z-score removal of extreme lab errors improved stability |
| **RobustScaler for messy data** | Uses median/IQR instead of mean/std — outlier-resistant |
| **IQR too aggressive** | Removed 16% of training data — lost important borderline cases |
| **Ablation studies = science** | Testing one variable at a time reveals true impact vs lucky coincidence |
| **Hemoglobin dominates** | As expected clinically — it's literally the definition of anemia |

<br/>

## 📦 Dependencies

```bash
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
joblib>=1.3
```

<br/>

## 🔗 Part of 60 Days of ML & DL Challenge

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 7: Stroke Prediction](../day07_stroke_prediction/) | **🩸 Day 8: Anemia Detection** | [Day 9: Hepatitis Diagnosis](../day09_hepatitis_diagnosis/) |
| XGBoost + SHAP | AdaBoost + Outlier Removal | Perceptron + ROC Analysis |

</div>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=B71C1C&height=150&section=footer&animation=twinkling" width="100%"/>

<br/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=16&duration=4000&pause=2000&color=EF5350&center=true&vCenter=true&repeat=true&width=500&lines=%F0%9F%A9%B8+Boosting+weak+learners+%7C+Detecting+anemia" alt="Footer" /></a>

</div>
