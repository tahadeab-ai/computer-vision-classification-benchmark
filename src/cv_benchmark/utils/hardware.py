import time
from typing import Dict, Tuple
import torch
import torch.nn as nn


def get_device(preference: str = "cuda") -> torch.device:
    """Return the specified device if available, otherwise fall back to CPU."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_info() -> Dict[str, str]:
    """Retrieve GPU information for logging and verification."""
    if not torch.cuda.is_available():
        return {"available": "False", "device": "CPU"}
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {
        "available": "True",
        "device": device_name,
        "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
        "total_memory_gb": f"{total_memory_gb:.2f} GB",
    }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch model.

    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device = torch.device("cpu"),
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark inference latency (ms) and throughput (samples/sec)."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    batch_size = input_shape[0]

    for _ in range(benchmark_runs):
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    latencies.sort()
    avg_latency = float(sum(latencies) / len(latencies))
    p50 = float(latencies[int(0.50 * len(latencies))])
    p95 = float(latencies[int(0.95 * len(latencies))])
    p99 = float(latencies[int(0.99 * len(latencies))])
    throughput = float((batch_size / (avg_latency / 1000.0))) if avg_latency > 0 else 0.0

    return {
        "mean_latency_ms": round(avg_latency, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "p99_latency_ms": round(p99, 3),
        "throughput_fps": round(throughput, 2),
    }
