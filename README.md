# Computer Vision Classification Benchmark

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-success.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

A professional, modular, and reproducible Computer Vision classification benchmark comparing scratch CNNs, classical transfer learning backbones, modern lightweight architectures, and Vision Transformers on the **CIFAR-100** dataset.

---

## 1. Executive Summary & Objective

This project systematically investigates the trade-offs in modern computer vision image classification:
- **Accuracy vs. Efficiency**: How do parameters and FLOPs translate to Top-1 / Top-5 classification accuracy and inference latency?
- **Inductive Bias vs. Representation Capacity**: How does a custom-crafted inductive-bias CNN trained from scratch compare against pretrained residual architectures, compound scaled CNNs, and patch-based Vision Transformers (ViT)?
- **Transfer Learning Paradigms**: Comparing linear probing (feature extraction with frozen backbone) against end-to-end fine-tuning.
- **Explainability**: Interpreting decisions via **Grad-CAM** for CNN representations and **Attention Rollout** for self-attention patch layers.

---

## 2. Benchmark Architectures

| Architecture | Family | Pretrained Weights | Primary Characteristics | Target Input Res |
| :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | Scratch CNN | None | 4-stage Conv-BN-ReLU-MaxPool block, Dropout, Dense head | 32x32 |
| **VGG16** | Classical CNN | `IMAGENET1K_V1` | Deep homogeneous 3x3 convolutions, heavy parameter footprint | 224x224 |
| **MobileNetV2** | Edge / Mobile | `IMAGENET1K_V2` | Inverted residuals & linear bottlenecks for edge latency | 224x224 |
| **ResNet50** | Residual CNN | `IMAGENET1K_V2` | Skip connections mitigating gradient vanishing, identity mapping | 224x224 |
| **EfficientNet-B0** | Scaled CNN | `IMAGENET1K_V1` | Neural architecture search, compound depth/width/resolution scaling | 224x224 |
| **ViT-B/16** | Vision Transformer | `IMAGENET1K_V1` | Multi-head self-attention over non-overlapping 16x16 image patches | 224x224 |

---

## 3. Repository Architecture

```
computer-vision-classification-benchmark/
├── configs/
│   ├── dataset.yaml            # Dataset splits, download paths, normalization
│   ├── models.yaml             # Architecture configurations and head overrides
│   └── training.yaml           # Optimizers, schedulers, AMP, early stopping
├── notebooks/colab/
│   ├── 01_environment_setup.ipynb     # Colab runtime validation & Drive mount
│   ├── 02_data_exploration.ipynb      # CIFAR-100 EDA, stats, class distribution
│   ├── 03_custom_cnn.ipynb            # Train custom CNN from scratch
│   ├── 04_transfer_learning.ipynb     # Feature extraction (frozen backbones)
│   ├── 05_fine_tuning.ipynb           # End-to-end fine-tuning
│   ├── 06_model_comparison.ipynb      # Benchmark aggregation & Pareto analysis
│   └── 07_model_explainability.ipynb  # Grad-CAM & Attention Rollout
├── src/cv_benchmark/
│   ├── data/                   # CIFAR-100 loader, splitters, transforms
│   ├── models/                 # Model factory, head builders, layer freezers
│   ├── training/               # Mixed precision engine, checkpoint recovery, early stopping
│   ├── evaluation/             # Top-1/5, Macro/Weighted F1, profiler
│   ├── explainability/         # Grad-CAM and ViT Attention Rollout
│   ├── visualization/          # Training curves, confusion matrices, Pareto frontiers
│   └── utils/                  # Hardware profiler, config parser, seeding
├── tests/                      # Automated test suite (run in Colab)
├── reports/                    # Benchmark metrics, generated tables, and figures
└── scripts/
    ├── run_smoke_test.py       # Quick end-to-end pipeline verification
    └── run_benchmark.py        # Comprehensive benchmark runner
```

---

## 4. Setup and Remote Colab Execution

### Remote Google Colab Workflow (Colab MCP)
Because deep learning training on CIFAR-100 across 6 architectures requires GPU acceleration, all execution is strictly offloaded to Google Colab.

1. **Clone and Install in Colab**:
```bash
!git clone https://github.com/tahadeab-ai/computer-vision-classification-benchmark.git
%cd computer-vision-classification-benchmark
!pip install -r requirements.txt
!pip install -e .
```

2. **Run Test Suites**:
```bash
!pytest tests/ -v
```

3. **Execute Benchmark / Smoke Test**:
```bash
!python scripts/run_smoke_test.py
!python scripts/run_benchmark.py --config configs/training.yaml
```

---

## 5. Benchmark Results & Trade-Off Analysis

*(Populated with verified metrics from remote Colab benchmark runs)*

| Model | Mode | Top-1 Acc (%) | Top-5 Acc (%) | Macro F1 | Params (M) | Latency (ms) | Throughput (img/s) | Checkpoint (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Custom CNN | Scratch | *TBD* | *TBD* | *TBD* | ~1.2 | *TBD* | *TBD* | ~5 |
| VGG16 | Feature Extract | *TBD* | *TBD* | *TBD* | 134.7 | *TBD* | *TBD* | ~515 |
| MobileNetV2 | Feature Extract | *TBD* | *TBD* | *TBD* | 2.4 | *TBD* | *TBD* | ~9.5 |
| ResNet50 | Feature Extract | *TBD* | *TBD* | *TBD* | 23.7 | *TBD* | *TBD* | ~90 |
| EfficientNet-B0 | Feature Extract | *TBD* | *TBD* | *TBD* | 4.1 | *TBD* | *TBD* | ~16 |
| ViT-B/16 | Feature Extract | *TBD* | *TBD* | *TBD* | 85.9 | *TBD* | *TBD* | ~330 |

---

## 6. Explainability Methodology
- **Grad-CAM (CNNs)**: Extracts activation maps from the final convolutional layer (e.g. `layer4[2].conv3` in ResNet50, `features[18]` in MobileNetV2) weighted by the gradients of the target class score.
- **Attention Rollout (ViT)**: Accounts for multi-head self-attention across transformer encoder blocks, computing the recursive flow of information from the `[CLS]` token to all image spatial tokens.

---

## 7. License
MIT License. Authored by Taha Dhiab.
