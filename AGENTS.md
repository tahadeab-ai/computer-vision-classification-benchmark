# AGENTS.md

## Repository Overview & Guidelines for AI Agents

Welcome to `computer-vision-classification-benchmark`. This repository houses a modular Computer Vision classification benchmarking suite comparing:
1. **Custom CNN** (trained from scratch)
2. **VGG16** (Transfer Learning / Fine-Tuning)
3. **MobileNetV2** (Transfer Learning / Fine-Tuning)
4. **ResNet50** (Transfer Learning / Fine-Tuning)
5. **EfficientNet-B0** (Transfer Learning / Fine-Tuning)
6. **Vision Transformer (ViT-B/16)** (Transfer Learning / Fine-Tuning)

---

### 1. Mandatory Execution Rule (Non-Negotiable)
- **Local Machine Compute Restriction**: Under NO circumstances should model training, inference, local testing (`pytest`), or heavy script execution be executed on the local host machine.
- **Compute Offloading**: All data processing, training loops, test verification, and benchmarking must be executed inside the **Google Colab Hosted Runtime** using the configured **Colab MCP server/tools**.
- **Local Host Actions Permitted**: File authoring/editing, git operations (`git commit`, `git push`, branch management), and static inspection.

---

### 2. Architecture & File Structure
```
computer-vision-classification-benchmark/
├── configs/                # YAML configs for dataset, models, training
│   ├── dataset.yaml
│   ├── models.yaml
│   └── training.yaml
├── notebooks/colab/        # Orchestration notebooks (01_ to 07_)
├── src/cv_benchmark/       # Reusable package code
│   ├── data/               # CIFAR-100 download, transforms, splits, dataloaders
│   ├── models/             # Custom CNN & pretrained model wrappers / factory
│   ├── training/           # Engine, loss, scheduler, checkpointing, early stopping
│   ├── evaluation/         # Metrics (Top-1/5, F1), confusion matrix, profiler
│   ├── explainability/     # Grad-CAM and ViT attention rollout
│   ├── visualization/      # Curves, confusion matrices, trade-off frontier plots
│   └── utils/              # Seeds, config loaders, hardware profiler
├── tests/                  # Unit and integration test suites (executed on Colab)
├── reports/                # Final benchmark tables, metric summaries, and plots
└── scripts/                # Execution scripts for remote Colab runner
```

---

### 3. Git & Commit Protocol
- Use Conventional Commits strictly:
  - `feat(<scope>): ...`
  - `fix(<scope>): ...`
  - `test(<scope>): ...`
  - `docs(<scope>): ...`
  - `refactor(<scope>): ...`
- Work on descriptive feature branches and merge into `main` after verification.
- Never commit raw datasets (`data/`) or model checkpoints (`*.pt`, `checkpoints/`).
