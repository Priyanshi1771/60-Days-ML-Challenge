<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=00695C&height=250&section=header&text=Day%2022%20%E2%80%94%20Brain%20Tumor%20Segmentation&fontSize=35&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%F0%9F%A7%A0%20U-Net%20%E2%80%94%20First%20Segmentation%20Model!&descSize=18&descAlignY=55&descColor=80CBC4" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=26A69A&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%A7%A0+Segmenting+Brain+Tumors+from+MRI;%F0%9F%8F%97%EF%B8%8F+U-Net%3A+Encoder+%2B+Decoder+%2B+Skip+Connections;%F0%9F%8E%AF+Dice+%2B+BCE+Loss+for+Pixel-Level+Prediction" alt="Typing SVG" /></a>

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#)
[![Day](https://img.shields.io/badge/Day-22%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)
[![Segmentation](https://img.shields.io/badge/Task-Segmentation-00C853?style=for-the-badge)](#)

<br/>
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## The Story

*An MRI scan arrives. Somewhere in the brain, a tumor hides. A radiologist must outline it precisely -- pixel by pixel -- for the surgeon to plan the operation. This takes 30-60 minutes per scan. A U-Net does it in milliseconds. Today we build that U-Net from scratch.*

### Classification vs Segmentation

```
  Day 21 (Classification):           Day 22 (Segmentation):
  "Is there pneumonia?"              "WHERE exactly is the tumor?"
  
  Input:  Image                       Input:  Image (128x128)
  Output: Single label (0 or 1)       Output: Mask (128x128 pixels)
                                              Each pixel = tumor or not
  
  +---------+     "Pneumonia"         +---------+     +---------+
  | Chest   | --> [0.92]              | Brain   | --> | :::     |
  | X-ray   |                        | MRI     |     | :tumor: |
  +---------+                        +---------+     +---------+
                                                      pixel-level!
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## U-Net Architecture

```
  The "U" shape: Encoder compresses, Decoder expands, Skip connections preserve detail.

  Input (1x128x128)
    |
    v
  [Enc1: 1->32]  ----skip1--->  [Dec1: 64->32]  -> Output (1x128x128)
    |  MaxPool                       ^  UpConv
  [Enc2: 32->64]  ---skip2--->  [Dec2: 128->64]
    |  MaxPool                       ^  UpConv
  [Enc3: 64->128] ---skip3--->  [Dec3: 256->128]
    |  MaxPool                       ^  UpConv
  [Enc4: 128->256] --skip4-->   [Dec4: 512->256]
    |  MaxPool                       ^  UpConv
    v                                |
  [Bottleneck: 256->512] ----------->
  
  Skip connections: CONCATENATE encoder features with decoder features
  This preserves spatial detail that pooling would destroy!
```

### Why Skip Connections Matter

```
  WITHOUT skip connections:
    Encoder compresses 128x128 -> 8x8 -> loses WHERE the tumor was
    Decoder guesses approximate location -> blurry boundaries
  
  WITH skip connections:
    Encoder features (edges, textures) are passed directly to decoder
    Decoder knows both WHAT (from bottleneck) and WHERE (from skips)
    Result: sharp, precise tumor boundaries
```

### Dice + BCE Loss

```
  Problem: tumor = ~5% of pixels, background = ~95%
  BCE alone: model learns to predict "no tumor everywhere" (95% accurate!)
  
  Dice loss fixes this:
    Dice = 2 * |Prediction AND Truth| / (|Prediction| + |Truth|)
    Dice = 1.0 -> perfect overlap
    Dice = 0.0 -> no overlap at all
    
  Combined: Loss = BCE + (1 - Dice)
  BCE gives pixel-level gradients, Dice handles class imbalance.
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## The Data

| Property | Detail |
|:---------|:-------|
| **Samples** | 2,000 synthetic brain MRI slices (128x128, grayscale) |
| **Tumor slices** | ~65% contain tumors, ~35% are healthy |
| **Tumor features** | Bright heterogeneous mass + dark necrotic core |
| **Brain features** | Skull ring, gray/white matter, ventricles |
| **Evaluation** | Dice coefficient + IoU (Intersection over Union) |

### What the Synthetic MRI Contains

```
  +--Brain MRI Anatomy--+
  |   /skull ring\       |    Tumor characteristics:
  |  / __________ \      |    - Bright heterogeneous mass
  | / / brain    \ \     |    - Dark necrotic core center
  | | | ventricle | |    |    - Located inside brain parenchyma
  | | |  (dark)   | |    |    - Variable size (6-22 pixel radius)
  | | |           | |    |
  | | |   [TUMOR] | |    |    Color coding in overlay:
  | \ \__________/ /     |      Green = True Positive
  |  \____________/      |      Red   = False Positive
  |    background        |      Blue  = False Negative
  +---------------------+
```

<br/>

## Project Structure

```
day22_brain_tumor_segmentation/
+-- main.py           +-- config.py          +-- data_pipeline.py
+-- model_training.py +-- evaluation.py      +-- README.md
+-- data/ models/ plots/ logs/ outputs/
```

## Quick Start

```bash
cd day22_brain_tumor_segmentation && python main.py
```

**Pipeline:**
1. Generate 2,000 synthetic brain MRI slices with tumors
2. EDA: sample images + masks + overlays + tumor size distribution
3. Train U-Net on GPU (Dice+BCE loss, AMP, early stopping)
4. Evaluate: Dice, IoU + visual overlay (green=TP, red=FP, blue=FN)
5. Show worst-to-best predictions ranked by Dice score

<br/>

## Generated Visualizations

| # | Plot | What It Shows |
|:-:|:-----|:-------------|
| 01 | Samples | 6 MRI slices + ground truth masks + red overlays |
| 02 | Tumor Stats | Tumor size distribution + slice balance |
| 03 | Training | Dice+BCE loss + Dice score curves |
| 04 | **Predictions** | Worst-to-best: MRI + color overlay (G=TP, R=FP, B=FN) |
| 05 | Metrics | Dice/IoU distributions + summary bars |

<br/>

## GPU Optimizations

| Optimization | Impact |
|:-------------|:-------|
| `AMP autocast + GradScaler` | ~2x GPU speedup |
| `bias=False` in Conv2d (with BN) | Fewer parameters |
| `Kaiming init` | Faster convergence |
| `set_to_none=True` | Faster grad zeroing |
| `non_blocking=True` | Async CPU-GPU transfer |
| `gradient clipping (1.0)` | Stable training |
| `del images, masks` after loaders | Free memory |
| `drop_last=True` train loader | BatchNorm stability |

<br/>

## Key Segmentation Metrics

| Metric | Formula | Range | Meaning |
|:-------|:--------|:------|:--------|
| **Dice** | 2\|P*T\| / (\|P\|+\|T\|) | 0-1 | Overlap quality (1=perfect) |
| **IoU** | \|P*T\| / \|P+T-P*T\| | 0-1 | Intersection/Union (stricter) |
| Dice > 0.7 | | | Clinically useful |
| Dice > 0.85 | | | Publication quality |

<br/>

## Lessons Learned

| Lesson | Detail |
|:-------|:-------|
| **Skip connections are essential** | Without them, decoder loses spatial precision |
| **Dice loss beats BCE alone** | BCE ignores class imbalance, Dice directly optimizes overlap |
| **Segmentation != Classification** | Output is a full-resolution mask, not a single label |
| **U-Net generalizes** | Same architecture works for tumors, organs, lesions, cells |
| **Small tumors are hardest** | Low pixel count means Dice is very sensitive to FP/FN |
| **This is Day 22's foundation** | U-Net appears again in Days 26, B2, B13 with variations |

<br/>

## Dependencies

```
numpy, torch, matplotlib, pandas
```

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 21: Pneumonia](../day21_pneumonia_detection/) | **Day 22: Brain Tumor Seg** | [Day 23: Skin Lesion](../day23_skin_lesion/) |
| CNN Classification | U-Net Segmentation | ResNet Transfer Learning |

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
<br/><br/>
<img src="https://capsule-render.vercel.app/api?type=waving&color=00695C&height=120&section=footer&animation=twinkling" width="100%"/>
<br/>
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=16&duration=4000&pause=2000&color=26A69A&center=true&vCenter=true&repeat=true&width=500&lines=%F0%9F%A7%A0+Segmenting+tumors+%7C+Pixel+by+pixel+%7C+Saving+brains" alt="Footer" /></a>
</div>
