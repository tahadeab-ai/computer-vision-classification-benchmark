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

*(Empirically benchmarked on Google Colab hosted runtime with NVIDIA Tesla T4 15GB GPU, PyTorch 2.11+cu130, and CUDA 13.0 on CIFAR-100 test dataset)*

| Architecture | Paradigm | Top-1 Acc (%) | Top-5 Acc (%) | Macro F1 | Total Params | Trainable Params | Mean Latency (ms) | Throughput (img/s) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | Scratch (4-stage) | 36.88% | 68.75% | 0.3535 | 0.767 M | 0.767 M | **2.31 ms** | **432.9** | **34.5 MB** |
| **MobileNetV2** | Feature Extraction | 46.95% | 76.24% | 0.4544 | 2.352 M | 0.128 M | 5.15 ms | 194.3 | 38.4 MB |
| **ResNet50** | Feature Extraction | 60.25% | 87.01% | 0.5963 | 23.713 M | 0.205 M | 6.33 ms | 157.9 | 120.8 MB |
| **ResNet50** | **Fine-Tuning** | **71.67%** | **94.17%** | **0.7129** | 23.713 M | 23.713 M | 10.85 ms | 92.2 | 475.5 MB |
| **EfficientNet-B0** | Feature Extraction | 55.17% | 83.17% | 0.5446 | 4.136 M | 0.128 M | 8.05 ms | 124.3 | 46.0 MB |
| **VGG16** | Feature Extraction | 54.46% | 83.68% | 0.5400 | 134.670 M | 0.410 M | 16.83 ms | 59.4 | 562.5 MB |
| **ViT-B/16** | Feature Extraction | **80.31%** | **96.45%** | **0.8023** | 85.876 M | 0.077 M | 15.03 ms | 66.5 | 354.4 MB |

---

### Empirical Research Findings

1. **Classification Performance Ceiling**:
   * **Vision Transformer (ViT-B/16)** achieves the highest classification performance (**80.31% Top-1**, **96.45% Top-5**, **0.8023 Macro F1**). Self-attention over 16x16 patch embeddings captures rich global semantic contexts that transfer with exceptional fidelity to CIFAR-100.
2. **Inference Latency & Edge Suitability**:
   * **Custom CNN** delivers the lowest latency (**2.31 ms**, 432.9 fps) due to 32x32 native processing and compact channel dimensions.
   * Among 224x224 backbones, **MobileNetV2** demonstrates premier efficiency (**5.15 ms**, 194.3 fps, 38.4 MB VRAM), making it the optimal candidate for constrained edge deployment.
3. **Transfer Learning vs. Training From Scratch**:
   * Training from scratch on 100 classes with 500 samples/class reaches 36.88% in early epochs due to data scarcity. Transfer learning from ImageNet provides an immediate leap of **+10.07% to +43.43%**, showing that low-level and mid-level feature representations are highly reusable.
4. **Feature Extraction vs. End-to-End Fine-Tuning**:
   * Fine-tuning ResNet50 with differential learning rates (`backbone_lr = 1e-5`, `head_lr = 1e-4`) boosted Top-1 accuracy from **60.25% to 71.67%** (+11.42% absolute gain), unlocking domain adaptation in deep residual representations at the cost of higher training memory footprint (475.5 MB vs 120.8 MB).
5. **Parameter Efficiency**:
   * **VGG16** consumes 134.7M parameters but achieves only 54.46% Top-1 accuracy, requiring 562.5 MB VRAM. In contrast, **EfficientNet-B0** achieves superior accuracy (55.17%) with **32.5x fewer parameters** (4.1M) and **3x lower latency** (8.05 ms vs 16.83 ms).

---

## 6. Explainability Methodology
- **Grad-CAM (CNNs)**: Extracts activation maps from the final convolutional layer (e.g. `layer4[2].conv3` in ResNet50, `features[18]` in MobileNetV2) weighted by the gradients of the target class score.
- **Attention Rollout (ViT)**: Accounts for multi-head self-attention across transformer encoder blocks, computing the recursive flow of information from the `[CLS]` token to all image spatial tokens.

---

## 7. License
MIT License. Authored by Taha Dhiab.
