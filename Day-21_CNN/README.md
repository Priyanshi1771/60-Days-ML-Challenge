<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0277BD&height=250&section=header&text=Day%2021%20%E2%80%94%20Pneumonia%20Detection&fontSize=38&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%F0%9F%AB%81%20CNN%20%2B%20Data%20Augmentation%20%E2%80%94%20Phase%203%20Begins!&descSize=18&descAlignY=55&descColor=81D4FA" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=29B6F6&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%AB%81+Detecting+Pneumonia+from+Chest+X-Rays;%F0%9F%A7%A0+4-Block+CNN+with+Heavy+Augmentation;%F0%9F%8E%89+Phase+3%3A+Deep+Learning+%26+Medical+Imaging!" alt="Typing SVG" /></a>

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#)
[![Day](https://img.shields.io/badge/Day-21%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)
[![Phase3](https://img.shields.io/badge/Phase%203-BEGINS!-00C853?style=for-the-badge)](#)

<br/>

[![CNN](https://img.shields.io/badge/Model-4--Block%20CNN-29B6F6?style=flat-square&logo=pytorch&logoColor=white)](#-chapter-2-the-architecture)
[![Augmentation](https://img.shields.io/badge/Technique-Data%20Augmentation-FF7043?style=flat-square&logo=transform&logoColor=white)](#-chapter-3-data-augmentation)
[![Xray](https://img.shields.io/badge/Data-Chest%20X--Rays-66BB6A?style=flat-square&logo=image&logoColor=white)](#-chapter-1-the-x-ray)
[![Medical](https://img.shields.io/badge/Domain-Medical%20Imaging-AB47BC?style=flat-square&logo=heart&logoColor=white)](#)

<br/>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

<br/>

---

## The Story of Day 21

*A radiologist stares at 200 chest X-rays stacked in a queue. Each one needs a diagnosis: normal or pneumonia? It takes 15 minutes per scan. A CNN can do it in 50 milliseconds. Today we build that CNN -- the first project of Phase 3: Deep Learning & Medical Imaging.*

---

<br/>

## Chapter 1: The X-Ray

<div align="center">

```
  NORMAL Chest X-ray              PNEUMONIA Chest X-ray
  ===================             =====================

  +------------------+            +------------------+
  |                  |            |    ::::          |
  |  .----.  .----.  |            |  .::::.  .----.  |
  | /      \/      \ |            | /::::::\/      \ |
  ||  clear  clear  ||            ||  hazy    clear ||
  | \ lung / \ lung /|            | \:::::/ \ lung /|
  |  '----'  '----'  |            |  '::::'  '----'  |
  |    ribs visible   |            |  opacity hides   |
  +------------------+            +------------------+

  CNN learns: white/hazy patches in lung fields = PNEUMONIA
  Normal lungs appear dark and clear with visible rib lines
```

</div>

> **The stakes:** A missed pneumonia (false negative) can kill. A false alarm just means an extra CT scan. Recall matters more than precision here.

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## Chapter 2: The Architecture

```
4-Block CNN for Chest X-Ray Classification

  Input: Grayscale X-ray (1 x 128 x 128)
  
  Block 1: Conv2d(1->32)  + BN + ReLU + MaxPool(2)   --> 32 x 64 x 64
  Block 2: Conv2d(32->64) + BN + ReLU + MaxPool(2)   --> 64 x 32 x 32
  Block 3: Conv2d(64->128)+ BN + ReLU + MaxPool(2)   --> 128 x 16 x 16
  Block 4: Conv2d(128->256)+ BN + ReLU + MaxPool(2)  --> 256 x 8 x 8
  
  AdaptiveAvgPool2d(4)                                --> 256 x 4 x 4
  Flatten                                             --> 4096
  FC(4096 -> 512) + ReLU + Dropout(0.5)               --> 512
  FC(512 -> 2)                                        --> [Normal, Pneumonia]
  
  What each block detects:
    Block 1: Edges, intensity boundaries
    Block 2: Rib textures, lung borders
    Block 3: Opacity patterns, consolidation shapes
    Block 4: Disease-level features (normal vs abnormal patterns)
```

### Why These Design Choices?

| Choice | Reason |
|:-------|:-------|
| **Grayscale (1 channel)** | X-rays are inherently grayscale -- 3 channels waste memory |
| **4 blocks (not 3 or 5)** | 128->8 px in 4 halvings. 3 blocks = too little abstraction. 5 = overkill for 128px |
| **BatchNorm after every conv** | Stabilizes training on small medical datasets |
| **Dropout(0.5)** | Aggressive -- but necessary when you have <3000 training images |
| **AdaptiveAvgPool(4)** | Fixed 4x4 output regardless of input size -- flexible architecture |
| **Kaiming init** | Proper initialization for ReLU networks -- faster convergence |

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## Chapter 3: Data Augmentation

> With only ~1500 training images per class, the CNN would memorize every pixel without augmentation. We artificially expand the dataset by applying random transformations.

| Transform | What It Does | Why |
|:----------|:------------|:----|
| **HorizontalFlip (50%)** | Mirror left-right | X-rays can be flipped |
| **Rotation (10 deg)** | Slight tilt | Patient positioning varies |
| **Affine translate (5%)** | Shift image slightly | Off-center X-rays are common |
| **Brightness jitter (20%)** | Lighter/darker | Different X-ray machines |
| **Contrast jitter (20%)** | More/less contrast | Exposure variation |

```
  Original       Flipped        Rotated        Shifted        Brightened
  +------+       +------+       +------+       +------+       +------+
  |  ..  |       |  ..  |       | ..   |       |   .. |       |  ..  |
  | .  . |  -->  | .  . |  -->  |.  .  |  -->  |  .  .|  -->  | .  . |
  |  ..  |       |  ..  |       | ..   |       |   .. |       |  ..  |
  +------+       +------+       +------+       +------+       +------+
  
  1 image becomes ~50 unique variations during training!
  Augmentation is ONLY applied to training set -- never val/test.
```

<br/>

<div align="center">
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## Chapter 4: The Dataset

| Property | Detail |
|:---------|:-------|
| **Source** | Kaggle Chest X-Ray (or synthetic fallback) |
| **Total images** | 3,000 (1,500 Normal + 1,500 Pneumonia) |
| **Image size** | 128 x 128 grayscale |
| **Split** | 70% train / 15% val / 15% test |
| **Balance** | 50:50 (well balanced) |
| **Augmentation** | Flip + rotate + translate + color jitter (train only) |

<br/>

## Chapter 5: The Plots

| # | Plot | What It Shows |
|:-:|:-----|:-------------|
| 01 | **Sample X-Rays** | 6 Normal vs 6 Pneumonia side by side |
| 02 | **Training Curves** | Loss + accuracy over epochs (train vs val) |
| 03 | **Evaluation** | Confusion matrix + ROC curve + Precision-Recall curve |
| 04 | **Error Analysis** | Confidence histograms + TP/TN/FP/FN breakdown |

<br/>

## Project Structure

```
day21_pneumonia_detection/
+-- main.py              <-- Entry point
+-- config.py             <-- CNN arch, augmentation params, training config
+-- data_pipeline.py      <-- X-ray generation + augmentation + DataLoaders
+-- model_training.py     <-- PneumoniaCNN class + GPU training loop
+-- evaluation.py         <-- Metrics + ROC/PR + confidence + error analysis
+-- README.md
+-- data/  models/  plots/  logs/  outputs/
```

<br/>

## Quick Start

```bash
# Option A: With real Kaggle data
# Download chest_xray/ into data/ folder

# Option B: Synthetic fallback (automatic)
cd day21_pneumonia_detection
python main.py
```

**Pipeline:**
1. Load chest X-rays (real or synthetic)
2. Plot sample normal vs pneumonia images
3. Create augmented DataLoaders (train/val/test)
4. Train 4-block CNN on GPU with AMP + early stopping
5. Evaluate: accuracy, F1, AUC, PR curve, confusion matrix
6. Error analysis: confidence distributions + FP/FN breakdown

<br/>

## GPU Optimizations

| Optimization | Impact |
|:-------------|:-------|
| `AMP autocast + GradScaler` | ~2x GPU speedup |
| `non_blocking=True` | Async CPU->GPU transfer |
| `pin_memory=True` | Faster host->device via page-locked memory |
| `drop_last=True` | BatchNorm stability |
| `batch_size*2` for val/test | No gradients = bigger batches |
| `set_to_none=True` | Faster grad zeroing |
| `gradient clipping (1.0)` | Prevent exploding gradients |
| `Kaiming init` | Faster convergence |
| `del images` after loaders | Free raw images from RAM |
| `bias=False` in Conv2d+BN | BN has its own bias |

<br/>

## The Moral

| Lesson | Detail |
|:-------|:-------|
| **Augmentation is mandatory** | Small medical datasets + CNN = overfitting without augmentation |
| **Grayscale is enough** | X-rays have no color information -- 1 channel saves 3x memory |
| **Recall > Precision** | Missing pneumonia is worse than a false alarm |
| **Dropout(0.5) is aggressive** | But necessary for <3000 training images |
| **4 blocks for 128px** | Each MaxPool halves: 128->64->32->16->8. Adaptive pool fixes the rest |
| **Phase 3 foundation** | Same CNN pattern extends to Days 22-30 (U-Net, ResNet, DenseNet) |

<br/>

## Dependencies

```bash
numpy>=1.24
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
matplotlib>=3.7
Pillow>=9.0
```

<br/>

## Part of 60 Days of ML & DL Challenge

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 20: Viral Load](../day20_viral_load/) | **Day 21: Pneumonia Detection** | [Day 22: Brain Tumor Segmentation](../day22_brain_tumor_segmentation/) |
| LSTM vs GRU (Phase 2 Finale) | CNN + Augmentation (Phase 3 Begins!) | U-Net Segmentation |

<br/>

```
Phase 3: Deep Learning & Medical Imaging (Days 21-30)
=====================================================
Day 21: CNN Classification         <-- YOU ARE HERE
Day 22: U-Net Segmentation
Day 23: ResNet Transfer Learning
Day 24: VGG16 Fine-Tuning
Day 25: 3D CNN Volumetric
Day 26: DenseNet
Day 27: Custom CNN + Wavelets
Day 28: InceptionV3
Day 29: Multi-class CNN
Day 30: Attention Mechanisms
```

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<br/>
<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0277BD&height=120&section=footer&animation=twinkling" width="100%"/>

<br/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=16&duration=4000&pause=2000&color=29B6F6&center=true&vCenter=true&repeat=true&width=600&lines=%F0%9F%AB%81+Reading+X-rays+%7C+Detecting+pneumonia+%7C+One+pixel+at+a+time+%F0%9F%A7%A0" alt="Footer" /></a>

</div>
