import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def profile_model_efficiency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device = torch.device("cpu"),
    warmup_runs: int = 15,
    benchmark_runs: int = 50,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """Profile model computational efficiency, parameter footprint, and latency."""
    model.eval()
    model.to(device)

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # 2. Checkpoint size
    checkpoint_size_mb = 0.0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    else:
        # Theoretical parameter footprint in float32 (4 bytes per param)
        checkpoint_size_mb = (total_params * 4) / (1024 * 1024)

    # 3. Peak GPU memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Latency benchmarking
    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    latencies_arr = np.array(latencies)
    batch_size = input_shape[0]
    avg_latency = float(np.mean(latencies_arr))
    throughput = float((batch_size / (avg_latency / 1000.0))) if avg_latency > 0 else 0.0

    return {
        "total_parameters_m": round(total_params / 1e6, 3),
        "trainable_parameters_m": round(trainable_params / 1e6, 3),
        "frozen_parameters_m": round(frozen_params / 1e6, 3),
        "checkpoint_size_mb": round(checkpoint_size_mb, 2),
        "latency_mean_ms": round(avg_latency, 2),
        "latency_std_ms": round(float(np.std(latencies_arr)), 2),
        "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        "throughput_fps": round(throughput, 1),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
    }
