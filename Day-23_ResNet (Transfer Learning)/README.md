<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=6A1B9A&height=250&section=header&text=Day%2023%20%E2%80%94%20Skin%20Lesion%20Classification&fontSize=35&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%F0%9F%94%AC%20ResNet%20Transfer%20Learning%20%E2%80%94%203%20Strategies%20Compared&descSize=17&descAlignY=55&descColor=CE93D8" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=AB47BC&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%94%AC+Classifying+7+Types+of+Skin+Lesions;%F0%9F%A7%A0+Scratch+vs+Frozen+vs+Fine-Tune;%F0%9F%8E%AF+First+Transfer+Learning+Project!" alt="Typing SVG" /></a>

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ResNet](https://img.shields.io/badge/ResNet--18-ImageNet-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#)
[![Day](https://img.shields.io/badge/Day-23%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)

<br/>
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## The Story

*A dermatologist examines a suspicious mole. Is it benign or melanoma? There are 7 possible diagnoses. Training a CNN from scratch needs millions of skin images we don't have. But a ResNet trained on 14 million ImageNet photos already knows edges, textures, and shapes. Can we borrow that knowledge? Today we find out -- by racing 3 transfer learning strategies against each other.*

<br/>

## The 3-Way Experiment

```
  Strategy 1: SCRATCH                    Strategy 2: FROZEN
  ========================               ========================
  ResNet-18 (random weights)             ResNet-18 (ImageNet pretrained)
  Train ALL 11M parameters               FREEZE conv layers (10.8M frozen)
  LR = 1e-3 (high)                       Train ONLY FC head (200K trainable)
  Needs many epochs                      LR = 1e-3 (only FC is learning)
  Slow convergence                       Fast convergence, limited capacity

  Strategy 3: FINE-TUNE (usually wins!)
  =====================================
  ResNet-18 (ImageNet pretrained)
  UNFREEZE all layers (11M trainable)
  LR = 1e-4 (LOW -- preserve pretrained knowledge!)
  Best of both worlds: pretrained features + task adaptation
```

### Why Transfer Learning Works

```
  ImageNet features (learned from 14M photos):
    Layer 1:  edges, corners, colors         <-- universal!
    Layer 2:  textures, gradients            <-- universal!
    Layer 3:  patterns, parts                <-- mostly universal
    Layer 4:  object-level features          <-- need adaptation
    FC:       1000 ImageNet classes          <-- REPLACE for 7 skin classes

  These low-level features are IDENTICAL whether you're looking at
  a cat photo or a dermoscopy image. That's why transfer works.
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## The 7 Skin Lesion Classes

```
  Class                     Danger Level    Visual Signature
  =====                     ============    ================
  Melanocytic Nevi          Benign          Uniform brown, round
  Melanoma                  MALIGNANT!      Dark, irregular, asymmetric
  Benign Keratosis          Benign          Tan, rough surface
  Basal Cell Carcinoma      Malignant       Pearly/pink, rolled edges
  Actinic Keratosis         Pre-malignant   Rough, scaly, red/tan
  Vascular Lesion           Benign          Red streaks, blood vessels
  Dermatofibroma            Benign          Firm brown nodule
```

> Melanoma is the deadliest -- early detection saves lives. A model that confuses melanoma with a benign nevus could be fatal.

<br/>

## Dataset

| Property | Detail |
|:---------|:-------|
| **Images** | 3,500 synthetic skin lesion images (224x224 RGB) |
| **Classes** | 7 (500 per class, balanced) |
| **Lesion features** | Class-specific colors, shapes, textures |
| **Melanoma** | Dark, irregular boundary, asymmetric color patches |
| **Augmentation** | HFlip, VFlip, rotation(20), color jitter, ImageNet normalization |

<br/>

## Project Structure

```
day23_skin_lesion/
+-- main.py           +-- config.py          +-- data_pipeline.py
+-- model_training.py +-- evaluation.py      +-- README.md
+-- data/ models/ plots/ logs/ outputs/
```

## Quick Start

```bash
cd day23_skin_lesion && python main.py
```

**Pipeline:**
1. Generate 3,500 skin lesion images (7 classes, class-specific morphology)
2. EDA: sample grid showing all 7 lesion types
3. Build ResNet-18 three ways: scratch | frozen | fine-tune
4. Train all 3 on GPU (AMP, early stopping) -- same data, same epochs
5. Evaluate: accuracy, F1, confusion matrices, per-class analysis
6. Speed vs accuracy tradeoff plot

<br/>

## Generated Visualizations

| # | Plot | What It Shows |
|:-:|:-----|:-------------|
| 01 | Samples | 4 images per class, all 7 lesion types |
| 02 | **Strategy Comparison** | Loss curves + accuracy curves + winner bar chart |
| 03 | **Confusion Matrices** | All 3 strategies side-by-side (7x7 matrices) |
| 04 | **Per-Class Accuracy** | Which classes benefit most from transfer learning |
| 05 | Final Comparison | Accuracy bars + speed vs accuracy scatter |

<br/>

## GPU Optimizations

| Optimization | Impact |
|:-------------|:-------|
| AMP autocast + GradScaler | ~2x GPU speedup |
| `filter(requires_grad)` in optimizer | Frozen strategy only updates FC head |
| Lower LR for fine-tune (1e-4) | Preserves pretrained features |
| ImageNet normalization | Required for pretrained weights |
| `non_blocking=True` | Async CPU-GPU |
| Gradient clipping (1.0) | Stable training |
| Early stopping (patience=6) | Stop when converged |

<br/>

## Lessons Learned

| Lesson | Detail |
|:-------|:-------|
| **Fine-tune usually wins** | Pretrained features + task-specific adaptation |
| **Frozen is fastest** | Only FC head trains, but limited capacity |
| **Scratch needs more data** | Without pretraining, 3,500 images may not be enough |
| **Low LR for fine-tuning** | High LR destroys pretrained weights (catastrophic forgetting) |
| **ImageNet features transfer** | Edges and textures are universal across domains |
| **Per-class analysis reveals** | Transfer helps rare/hard classes disproportionately |
| **Melanoma detection is critical** | FN on melanoma = missed cancer |

<br/>

## Dependencies

```
numpy, torch, torchvision, scikit-learn, matplotlib, pandas
```

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 22: Brain Tumor Seg](../day22_brain_tumor_segmentation/) | **Day 23: Skin Lesion** | [Day 24: Diabetic Retinopathy](../day24_diabetic_retinopathy/) |
| U-Net Segmentation | ResNet Transfer Learning | VGG16 Fine-Tuning |

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
<br/><br/>
<img src="https://capsule-render.vercel.app/api?type=waving&color=6A1B9A&height=120&section=footer&animation=twinkling" width="100%"/>
<br/>
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=16&duration=4000&pause=2000&color=AB47BC&center=true&vCenter=true&repeat=true&width=600&lines=%F0%9F%94%AC+Transfer+learning+%7C+7+lesion+types+%7C+Saving+skin+%F0%9F%A7%A0" alt="Footer" /></a>
</div>
