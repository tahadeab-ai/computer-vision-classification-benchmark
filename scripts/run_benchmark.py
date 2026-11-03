import argparse
import json
import os
import sys
import pandas as pd
import torch
import torch.nn as nn

from cv_benchmark.utils import set_seed, load_config, get_device, count_parameters
from cv_benchmark.data import get_dataloaders, get_smoke_test_dataloaders, CIFAR100_FINE_CLASSES
from cv_benchmark.models import create_model, get_model_target_layer
from cv_benchmark.models.factory import get_parameter_groups
from cv_benchmark.training import Trainer
from cv_benchmark.evaluation import evaluate_model, profile_model_efficiency
from cv_benchmark.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_accuracy_vs_latency,
    plot_accuracy_vs_parameters,
    plot_cam_comparison,
)
from cv_benchmark.explainability import GradCAM, overlay_cam_on_image


def main():
    parser = argparse.ArgumentParser(description="Execute CV Classification Benchmark on CIFAR-100")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--models", nargs="+", default=["custom_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0", "vgg16", "vit_b_16"])
    parser.add_argument("--smoke", action="store_true", help="Run with smoke test subsets for quick validation")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.get("training", {}).get("seed", 42))
    device = get_device("cuda")

    reports_dir = "./reports"
    figures_dir = os.path.join(reports_dir, "figures")
    tables_dir = os.path.join(reports_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING COMPUTER VISION BENCHMARK EXPERIMENT")
    print(f"Device: {device} | Models: {args.models} | Smoke Mode: {args.smoke}")
    print("=" * 70)

    benchmark_records = []

    for model_name in args.models:
        print(f"\n[{model_name.upper()}] Preparing Pipeline...")

        is_custom = model_name == "custom_cnn"
        image_size = (32, 32) if is_custom else (224, 224)
        mode = "scratch" if is_custom else "feature_extraction"
        batch_size = cfg.get("training", {}).get("batch_size", {}).get(model_name, 32)
        epochs = args.epochs or (1 if args.smoke else (15 if is_custom else 8))

        # 1. Dataloaders
        if args.smoke:
            loaders = get_smoke_test_dataloaders(
                model_name=model_name,
                batch_size=min(batch_size, 16),
                image_size=image_size,
            )
        else:
            loaders = get_dataloaders(
                model_name=model_name,
                batch_size=batch_size,
                image_size=image_size,
            )

        # 2. Model Creation
        model, head_names = create_model(
            model_name=model_name,
            num_classes=100,
            pretrained=(not is_custom),
            mode=mode,
        )
        total_p, trainable_p = count_parameters(model)
        print(f"Model: {model_name} | Total Params: {total_p:,} | Trainable: {trainable_p:,}")

        # 3. Optimizer & Scheduler
        lr = 1e-3
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # 4. Training
        trainer = Trainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            experiment_name=f"{model_name}_{mode}",
            output_dir=reports_dir,
            checkpoint_dir="./checkpoints",
            mixed_precision=(device.type == "cuda"),
        )
        history = trainer.fit(epochs=epochs, max_steps_per_epoch=(2 if args.smoke else None))

        # Save training curve plot
        curve_path = os.path.join(figures_dir, f"{model_name}_{mode}_curves.png")
        plot_training_curves(history, save_path=curve_path, title=f"{model_name.upper()} ({mode})")

        # 5. Evaluation
        print(f"Evaluating {model_name} on Test Set...")
        eval_metrics = evaluate_model(
            model=model,
            test_loader=loaders["test"],
            device=device,
            num_classes=100,
            max_steps=(2 if args.smoke else None),
        )

        # 6. Profiling
        print(f"Profiling Latency and Throughput for {model_name}...")
        efficiency = profile_model_efficiency(
            model=model,
            input_shape=(1, 3, image_size[0], image_size[1]),
            device=device,
            warmup_runs=(3 if args.smoke else 15),
            benchmark_runs=(10 if args.smoke else 40),
        )

        record = {
            "model_name": model_name,
            "mode": mode,
            "top1_accuracy": eval_metrics["top1_accuracy"],
            "top5_accuracy": eval_metrics["top5_accuracy"],
            "f1_macro": eval_metrics["f1_macro"],
            "f1_weighted": eval_metrics["f1_weighted"],
            "total_parameters_m": efficiency["total_parameters_m"],
            "trainable_parameters_m": efficiency["trainable_parameters_m"],
            "latency_mean_ms": efficiency["latency_mean_ms"],
            "latency_p95_ms": efficiency["latency_p95_ms"],
            "throughput_fps": efficiency["throughput_fps"],
            "peak_gpu_memory_mb": efficiency["peak_gpu_memory_mb"],
        }
        benchmark_records.append(record)

        # 7. Explainability (Grad-CAM for CNNs)
        target_layer = get_model_target_layer(model_name, model)
        if target_layer is not None:
            try:
                gradcam = GradCAM(model=model, target_layer=target_layer)
                # Take 1 sample from test loader
                sample_imgs, sample_targets = next(iter(loaders["test"]))
                img_t = sample_imgs[0:1].to(device)
                cam = gradcam.generate(img_t, target_class=int(sample_targets[0].item()))

                raw_img = sample_imgs[0].permute(1, 2, 0).numpy()
                raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
                overlay = overlay_cam_on_image(raw_img, cam)

                class_name = CIFAR100_FINE_CLASSES[int(sample_targets[0].item())]
                cam_path = os.path.join(figures_dir, f"{model_name}_gradcam.png")
                plot_cam_comparison(raw_img, cam, overlay, class_name, save_path=cam_path)
                gradcam.remove_hooks()
            except Exception as e:
                print(f"Explainability generation notice for {model_name}: {e}")

    # Aggregated Summary Tables & Plots
    df = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(tables_dir, "benchmark_summary.csv")
    json_path = os.path.join(tables_dir, "benchmark_summary.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY:")
    print("=" * 70)
    print(df.to_string())

    # Generate Pareto trade-off charts
    plot_accuracy_vs_latency(benchmark_records, os.path.join(figures_dir, "pareto_accuracy_vs_latency.png"))
    plot_accuracy_vs_parameters(benchmark_records, os.path.join(figures_dir, "pareto_accuracy_vs_parameters.png"))

    print(f"\nAll artifacts, tables, and figures successfully saved to {reports_dir}")


if __name__ == "__main__":
    main()
