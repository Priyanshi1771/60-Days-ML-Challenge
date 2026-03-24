<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=311B92&height=250&section=header&text=Day%2016%20%E2%80%94%20Telomere%20Length%20Prediction&fontSize=36&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%F0%9F%A7%AC%20SVR%20%2B%20Feature%20Selection%20Techniques&descSize=18&descAlignY=55&descColor=B39DDB" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=7E57C2&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%A7%AC+Predicting+Biological+Aging+from+Biomarkers;%F0%9F%94%AC+6+Feature+Selection+Methods+Compared;%F0%9F%8E%AF+SVR%3A+Finding+Signal+in+a+Sea+of+Noise" alt="Typing SVG" /></a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Day](https://img.shields.io/badge/Day-16%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)

<br/>

[![SVR](https://img.shields.io/badge/Model-SVR%20(RBF)-7E57C2?style=flat-square&logo=target&logoColor=white)](#-chapter-3-svr)
[![Selection](https://img.shields.io/badge/Focus-6%20Selection%20Methods-EF5350?style=flat-square&logo=filter&logoColor=white)](#-chapter-2-the-detective)
[![GPU](https://img.shields.io/badge/GPU-Neural%20Net-76B900?style=flat-square&logo=nvidia&logoColor=white)](#)
[![DNA](https://img.shields.io/badge/Biology-Telomere%20Aging-4FC3F7?style=flat-square&logo=moleculer&logoColor=white)](#-prologue-the-biological-clock)

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

<br/>

---

## 📖 The Story of Day 16

*Inside every cell, a clock is ticking. At the ends of your chromosomes, protective caps called telomeres shorten with each cell division. When they're gone, the cell dies. Telomere length is the closest thing biology has to an "age meter" — and today, we predict it.*

---

<br/>

## 🧬 Prologue: The Biological Clock

<div align="center">

```
🧬 Chromosome with Telomeres

  Young Cell (telomere = 10+ kb)
  ╔══════════════════════════════════════════════╗
  ║▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓║  ← Long telomere caps
  ╚══════════════════════════════════════════════╝
   ↑ telomere                        telomere ↑

  Aging Cell (telomere = 5-7 kb)
  ╔══════════════════════════════════════════════╗
  ║▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓║  ← Shorter caps
  ╚══════════════════════════════════════════════╝

  Old Cell (telomere < 4 kb) → Cell death / senescence
  ╔══════════════════════════════════════════════╗
  ║▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓║  ← Critical!
  ╚══════════════════════════════════════════════╝

  Shorter telomeres → aging, cancer risk, cardiovascular disease
  Longer telomeres  → cellular youth, longevity
```

</div>

> **The mission:** predict telomere length (in kilobases) from 40 clinical and genomic features — but 7 of those features are pure noise. Can our algorithms separate the biological signal from the irrelevant?

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 🕵️ Chapter 1: The Trap — 7 Hidden Noise Features

> Of our 40 features, 33 are real biomarkers. 7 are **completely useless** — shoe size, eye color, birth month, zip code, a favorite number, and two random noise columns. If the model can't filter these out, it will fail.

```
40 Features:
  ✅ 33 Real Biomarkers            ❌ 7 Noise Traps
  ─────────────────────            ──────────────────
   Age, BMI, Blood Pressure         Shoe size
   Inflammatory markers (CRP, IL-6)  Eye color
   Telomerase activity               Birth month
   Hormones (cortisol, DHEA)        Zip code first digit
   Vitamins (D, B12, folate)         Favorite number
   Exercise, sleep, stress           Random noise × 2
   
  The question: can feature selection find the 7 impostors?
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 🔬 Chapter 2: The Detective — 6 Feature Selection Methods

<div align="center">

```mermaid
graph TD
    A["📥 40 Features<br/>(33 real + 7 noise)"] --> B["🔍 Variance Threshold<br/>Remove near-constant features"]
    A --> C["📊 Correlation Filter<br/>Keep top |corr| with target"]
    A --> D["🧠 Mutual Information<br/>Nonlinear dependency scoring"]
    A --> E["🔄 RFECV<br/>Recursive elimination with CV"]
    A --> F["🔪 Lasso L1<br/>Zeros out unimportant features"]
    B --> G["🎯 Top 15 features"]
    C --> G
    D --> G
    E --> G
    F --> G

    style A fill:#1a1b27,stroke:#7E57C2,color:#fff
    style B fill:#1a1b27,stroke:#EF5350,color:#fff
    style C fill:#1a1b27,stroke:#FFB74D,color:#fff
    style D fill:#1a1b27,stroke:#66BB6A,color:#fff
    style E fill:#1a1b27,stroke:#4FC3F7,color:#fff
    style F fill:#1a1b27,stroke:#AB47BC,color:#fff
    style G fill:#1a1b27,stroke:#7E57C2,color:#fff
```

</div>

### 📋 The 6 Methods Explained

| Method | How It Works | Catches Noise? | Speed |
|:-------|:------------|:---------------|:------|
| **None** | Keep all 40 features (baseline) | ❌ Keeps everything | ⚡ Instant |
| **Variance** | Remove features with near-zero variance | ⚠️ High variance ≠ useful | ⚡ Fast |
| **Correlation** | Keep top K features by \|correlation\| with target | ✅ Good for linear | ⚡ Fast |
| **Mutual Info** | Nonlinear dependency score (information theory) | ✅✅ Catches nonlinear too | 🔬 Medium |
| **RFECV** | Recursively remove worst feature, validate with CV | ✅✅✅ Gold standard | 🐌 Slow |
| **Lasso L1** | Regularization zeros out useless features | ✅✅ Simultaneous select+train | ⚡ Fast |

### 🎯 The Experiment Result

```
Method          → Features Kept  | Real ✅ | Noise 🔴 | Verdict
──────────────────────────────────────────────────────────────────
none            → 40             |   33    |    7      | ❌ Keeps all noise
variance        → 15             |   12    |    3      | ⚠️ Weak filter  
correlation     → 15             |   15    |    0      | ✅ Clean!
mutual_info     → 15             |   15    |    0      | ✅ Clean!
rfecv           → 15             |   14    |    1      | ✅ Nearly perfect
lasso           → 15             |   15    |    0      | ✅ Clean!
```

> **Lesson:** Variance threshold is the weakest — a random noise column can have high variance! Correlation, Mutual Info, and Lasso all correctly identified the noise features.

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 🎯 Chapter 3: SVR — Support Vector Regression

> SVR finds a tube (ε-tube) around the regression line. Points inside the tube don't contribute to the loss. Only "support vectors" outside the tube shape the model.

```
SVR with ε-tube:

  y ┤         ●                    ● = support vector (outside tube)
    │     ●  ╱───────────╲        ○ = inside tube (ignored!)
    │    ╱──╱  ○  ○  ○    ╲──╲
    │   ╱  ╱  ○  ○  ○  ○   ╲  ╲  ← ε-tube width
    │  ╱──╱ ○  ○  ○  ○  ○   ╲──╲
    │ ╱  ╱────────────────────╲  ╲
    │●  ╱                      ╲  ●
    │──╱                        ╲──
    ┼────────────────────────────────→ x

  RBF Kernel: maps features to higher dimension
  where a linear tube can capture nonlinear patterns.
  
  Why SVR hates noise: RBF kernel computes distances between ALL
  features. Noise features add "random distance" → blurs the signal.
  Feature selection removes noise → distances become meaningful.
```

### 🎛️ SVR Hyperparameters

| Param | Values | Effect |
|:------|:-------|:-------|
| **C** | 0.1, 1, 10, 100 | Penalty for errors outside ε-tube (high C = fit harder) |
| **ε** | 0.01, 0.05, 0.1 | Tube width (larger = more tolerant of errors) |
| **kernel** | rbf, linear | RBF = nonlinear, linear = simple |
| **gamma** | scale, auto | RBF kernel width (controls flexibility) |

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📊 Chapter 4: The Data

| Property | Detail |
|:---------|:-------|
| **Subjects** | 3,500 individuals |
| **Features** | 40 total (33 real biomarkers + 7 deliberate noise) |
| **Target** | Telomere length (kilobases), continuous 3-14 kb |
| **Key predictors** | Age (-), telomerase activity (+), CRP (-), exercise (+) |
| **Noise features** | Shoe size, eye color, birth month, zip digit, favorite number, 2× random |
| **Challenge** | SVR must find 33 signals among 40 features, or noise degrades predictions |

<br/>

## 🏗️ Project Structure

```
day16_telomere_prediction/
├── 📄 main.py              ← Entry point
├── 📄 config.py             ← 6 selection methods, SVR grid, noise feature list
├── 📄 data_pipeline.py      ← Biomarker data + feature selection comparison
├── 📄 model_training.py     ← SVR per selection method + GridSearch + GPU NN
├── 📄 evaluation.py         ← Metrics + telomere-vs-age plot + error analysis
├── 📄 README.md
├── 📁 data/    ├── 📁 models/    ├── 📁 plots/
├── 📁 logs/    └── 📁 outputs/
```

<br/>

## ⚡ Quick Start

```bash
cd day16_telomere_prediction
python main.py
```

**Pipeline:**
1. 🧬 Generate 3,500 subjects (33 biomarkers + 7 noise traps)
2. 📊 EDA: telomere distribution + top biological predictors
3. 🔬 Run 6 feature selection methods (variance, corr, MI, RFECV, Lasso, none)
4. 🎯 Train SVR on EACH feature subset → proves which selection is best
5. 🎯 Full GridSearchCV on best subset (C, ε, kernel, gamma)
6. 🧠 GPU neural net on same subset for comparison
7. 📈 Evaluate + telomere-vs-age plot colored by prediction error

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## 📈 Chapter 5: The Visualizations

| # | Plot | The Story It Tells |
|:-:|:-----|:------------------|
| 01 | EDA | 🧬 Telomere distribution + top 5 biological predictors |
| 02 | **Selection Comparison** | 🔬 Stacked bar (real vs noise kept) + feature selection heatmap |
| 03 | **Selection vs Performance** | 🎯 SVR RMSE and R² for each of the 6 methods |
| 04 | NN Training | 🧠 Loss curves |
| 05 | Predictions | 📈 Actual vs predicted + residuals + telomere-vs-age colored by error |
| 06 | Comparison | 🏆 All models ranked |

<br/>

## ⚡ Tech Stack & Optimizations

| Optimization | Impact |
|:-------------|:-------|
| `float32` everywhere | 50% memory |
| `n_jobs=-1` for SVR GridSearch + RFECV | Full CPU parallelism |
| `AMP autocast` for GPU NN | Mixed precision speedup |
| `compress=3` joblib | Smaller saved models |
| `rasterized=True` scatter | Smaller plot files |
| Early stopping (patience=10) | No wasted NN epochs |
| Linear SVR for RFECV | Fast feature elimination |
| Best mask saved with model | Reproduce feature selection at inference |

<br/>

## 💡 Chapter 6: The Moral

| Lesson | Detail |
|:-------|:-------|
| **Feature selection before SVR** | SVR with RBF kernel is HURT by noise features (distance pollution) |
| **Variance is misleading** | High variance ≠ high predictive power (random noise has high variance!) |
| **Correlation catches linear** | Simple but misses nonlinear dependencies |
| **Mutual info catches nonlinear** | Information-theoretic — no distribution assumptions |
| **RFECV is gold standard** | Wraps model training into selection — but slowest |
| **Lasso = select + train** | L1 penalty zeros out noise — fast and effective |
| **Age = strongest predictor** | Every year ≈ -0.04 kb telomere length (biology confirmed) |
| **Noise features degrade SVR** | Going from 40 → 15 features IMPROVES accuracy |

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
| [Day 15: BMI Prediction](../day15_bmi_prediction/) | **🧬 Day 16: Telomere Length** | [Day 17: Readmission Risk](../day17_readmission_risk/) |
| RF + Interaction Terms | SVR + Feature Selection | Logistic Regression + Time-Based Splits |

</div>

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=311B92&height=150&section=footer&animation=twinkling" width="100%"/>

<br/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=16&duration=4000&pause=2000&color=7E57C2&center=true&vCenter=true&repeat=true&width=600&lines=%F0%9F%A7%AC+Measuring+chromosomes+%7C+Predicting+aging+%7C+One+kilobase+at+a+time+%F0%9F%94%AC" alt="Footer" /></a>

</div>
