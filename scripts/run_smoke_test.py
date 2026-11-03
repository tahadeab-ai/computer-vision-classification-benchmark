import os
import sys
import torch
import torch.nn as nn

from cv_benchmark.utils import set_seed, get_device, count_parameters
from cv_benchmark.data import get_smoke_test_dataloaders
from cv_benchmark.models import create_model
from cv_benchmark.training import Trainer
from cv_benchmark.evaluation import evaluate_model, profile_model_efficiency


def main():
    print("=" * 60)
    print("CV Benchmark: Remote End-to-End Smoke Test")
    print("=" * 60)

    set_seed(42)
    device = get_device("cuda")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Custom CNN Smoke Test
    print("\n--- 1. Testing Custom CNN (Scratch) ---")
    custom_loaders = get_smoke_test_dataloaders(
        model_name="custom_cnn",
        data_dir="./data",
        batch_size=16,
        train_size=32,
        val_size=16,
        test_size=16,
        image_size=(32, 32),
    )
    custom_model, _ = create_model("custom_cnn", num_classes=100)
    total_p, trainable_p = count_parameters(custom_model)
    print(f"Custom CNN: Total Params = {total_p:,}, Trainable = {trainable_p:,}")

    custom_trainer = Trainer(
        model=custom_model,
        train_loader=custom_loaders["train"],
        val_loader=custom_loaders["val"],
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(custom_model.parameters(), lr=1e-3),
        device=device,
        experiment_name="smoke_custom_cnn",
        output_dir="./reports",
        checkpoint_dir="./checkpoints",
        mixed_precision=(device.type == "cuda"),
    )
    custom_trainer.fit(epochs=1)

    eval_res = evaluate_model(custom_model, custom_loaders["test"], device=device, num_classes=100)
    print(f"Custom CNN Smoke Test Top-1 Acc: {eval_res['top1_accuracy']}%")

    # 2. Pretrained Backbone Smoke Test (MobileNetV2)
    print("\n--- 2. Testing Pretrained Backbone (MobileNetV2) ---")
    mobilenet_loaders = get_smoke_test_dataloaders(
        model_name="mobilenet_v2",
        data_dir="./data",
        batch_size=8,
        train_size=16,
        val_size=8,
        test_size=8,
        image_size=(224, 224),
    )
    mb_model, _ = create_model("mobilenet_v2", num_classes=100, pretrained=True, mode="feature_extraction")
    mb_total, mb_trainable = count_parameters(mb_model)
    print(f"MobileNetV2: Total = {mb_total:,}, Trainable (head only) = {mb_trainable:,}")

    mb_trainer = Trainer(
        model=mb_model,
        train_loader=mobilenet_loaders["train"],
        val_loader=mobilenet_loaders["val"],
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(mb_model.parameters(), lr=1e-3),
        device=device,
        experiment_name="smoke_mobilenet_v2",
        output_dir="./reports",
        checkpoint_dir="./checkpoints",
        mixed_precision=(device.type == "cuda"),
    )
    mb_trainer.fit(epochs=1)

    # 3. Efficiency Profiler Smoke Test
    print("\n--- 3. Testing Efficiency Profiler ---")
    profile_res = profile_model_efficiency(
        custom_model,
        input_shape=(1, 3, 32, 32),
        device=device,
        warmup_runs=5,
        benchmark_runs=10,
    )
    print(f"Inference Latency: {profile_res['latency_mean_ms']} ms | Throughput: {profile_res['throughput_fps']} fps")

    print("\n" + "=" * 60)
    print("Smoke Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
