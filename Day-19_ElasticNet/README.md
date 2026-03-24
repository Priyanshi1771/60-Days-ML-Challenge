<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=BF360C&height=250&section=header&text=Day%2019%20%E2%80%94%20Radiosensitivity%20Prediction&fontSize=35&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%E2%98%A2%EF%B8%8F%20Elastic%20Net%20%2B%20Cross-Dataset%20Validation&descSize=18&descAlignY=55&descColor=FFAB91" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=FF7043&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%E2%98%A2%EF%B8%8F+Predicting+Tumor+Radiation+Response;%F0%9F%94%AC+Train+on+Lab+%E2%86%92+Test+on+Clinical;%F0%9F%A7%AC+Elastic+Net%3A+L1+%2B+L2+Combined" alt="Typing SVG" /></a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Day](https://img.shields.io/badge/Day-19%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)

<br/>
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## The Story

*A tumor sits under a radiation beam. Will it shrink or resist? We have two datasets: cell lines tested in the lab and real tumors from the clinic. The model trained on lab data must predict clinical outcomes -- the hardest test in translational medicine.*

### The Cross-Dataset Experiment

```
  Dataset A (Lab)                    Dataset B (Clinical)
  3000 cell lines                    1500 real tumors
  Low noise, controlled              High noise, messy
  SF2 mean = 0.45                    SF2 mean = 0.55 (SHIFTED!)

  Strategy 1: Train A -> Test A      (same domain -- easy)
  Strategy 2: Train B -> Test B      (same domain -- medium)
  Strategy 3: Train A -> Test B      (cross-domain -- HARD!)
              ^^^^^^^^^^^^^^^^^
              This is the real test!
```

### Elastic Net = Best of Both Worlds

```
  l1_ratio = 0.0  -->  Pure Ridge (L2 only)  -->  shrink all coefficients
  l1_ratio = 0.5  -->  Equal L1 + L2         -->  select + shrink
  l1_ratio = 1.0  -->  Pure Lasso (L1 only)  -->  zero out features
  
  Elastic Net searches the entire l1_ratio x alpha grid to find
  the optimal balance for each dataset.
```

### Key Takeaway

Cross-dataset R2 **always drops** compared to within-dataset R2. This gap IS the domain shift -- and it's the most honest measure of whether your model will work in the real world.

## Quick Start
```bash
cd day19_radiosensitivity && python main.py
```

## Project Structure
```
day19_radiosensitivity/
+-- main.py           +-- config.py          +-- data_pipeline.py
+-- model_training.py +-- evaluation.py      +-- README.md
+-- data/ models/ plots/ logs/ outputs/
```

## Dependencies
```
numpy, pandas, torch, scikit-learn, matplotlib, joblib
```

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 18: Gene Expression](../day18_gene_expression/) | **Day 19: Radiosensitivity** | [Day 20: Viral Load](../day20_viral_load/) |

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
<img src="https://capsule-render.vercel.app/api?type=waving&color=BF360C&height=120&section=footer&animation=twinkling" width="100%"/>
</div>
