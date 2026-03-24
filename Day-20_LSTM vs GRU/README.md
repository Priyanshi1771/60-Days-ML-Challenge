<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=880E4F&height=250&section=header&text=Day%2020%20%E2%80%94%20Viral%20Load%20Forecasting&fontSize=38&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&desc=%F0%9F%A6%A0%20LSTM%20vs%20GRU%20%E2%80%94%20Phase%202%20Finale!&descSize=18&descAlignY=55&descColor=F48FB1" width="100%"/>

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=3000&pause=1000&color=E91E63&center=true&vCenter=true&repeat=true&width=700&height=45&lines=%F0%9F%A6%A0+Forecasting+HIV+Viral+Load;%F0%9F%A7%A0+LSTM+vs+GRU+Memory+Battle;%F0%9F%8E%89+Phase+2+Complete!" alt="Typing SVG" /></a>

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](#)
[![Day](https://img.shields.io/badge/Day-20%20of%2060-7C4DFF?style=for-the-badge&logo=googlecalendar&logoColor=white)](#)
[![Phase2](https://img.shields.io/badge/Phase%202-FINALE!-E91E63?style=for-the-badge)](#)

<br/>
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
</div>

<br/>

## The Story

*An HIV patient starts therapy. Viral load crashes from 1 million to undetectable. But 30% of patients rebound weeks later. An LSTM watches 8 weeks of history and predicts what comes next. A GRU does the same with fewer parameters. Which wins?*

### The Trajectory Challenge
```
  Viral Load (log10)
  6 |*                               * = rebound patient
    |  *                            
  4 |    *  *               * *     
    |        *  *       * *    *    
  2 |              * * *          * * = suppressed
    +--+--+--+--+--+--+--+--+--+--+
    0     10    20    30    40    50   Weeks
    
    LSTM/GRU input: [week 1..8] -> predict: [week 9]
```

### LSTM vs GRU
| | LSTM | GRU |
|:---|:---|:---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | More | ~25% fewer |
| Long-term memory | Better | Good |
| Training speed | Slower | Faster |

## Quick Start
```bash
cd day20_viral_load && python main.py
```

## Structure
```
day20_viral_load/
+-- main.py, config.py, data_pipeline.py
+-- model_training.py, evaluation.py, README.md
+-- data/, models/, plots/, logs/, outputs/
```

<div align="center">

| Previous | Current | Next |
|:---------|:--------|:-----|
| [Day 19: Radiosensitivity](../day19_radiosensitivity/) | **Day 20: Viral Load** | [Day 21: Pneumonia](../day21_pneumonia_detection/) |

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">
<img src="https://capsule-render.vercel.app/api?type=waving&color=880E4F&height=120&section=footer&animation=twinkling" width="100%"/>
</div>
