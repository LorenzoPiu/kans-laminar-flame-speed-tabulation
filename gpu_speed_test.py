#!/usr/bin/env python3
"""
PyTorch neural-network training speed benchmark.

Supports:
- CUDA GPUs
- Apple Silicon GPUs via MPS, including M1/M2/M3/M4 Macs
- CPU fallback

Example usage:

    python benchmark_pytorch_training.py

    python benchmark_pytorch_training.py --epochs 5 --batch-size 128

    python benchmark_pytorch_training.py --device mps

    python benchmark_pytorch_training.py --model cnn --image-size 224
"""

import argparse
import platform
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class BenchmarkResult:
    device: str
    model: str
    batch_size: int
    epochs: int
    samples_per_epoch: int
    avg_epoch_time_sec: float
    avg_samples_per_sec: float
    avg_batch_time_ms: float
    peak_memory_mb: float | None


class SmallCNN(nn.Module):
    """
    A simple CNN benchmark model.

    This is large enough to exercise the GPU, but small enough to run on laptops.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class MLP(nn.Module):
    """
    A dense neural network benchmark model.

    Useful for testing general matrix-multiplication speed.
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 4096, num_classes: int = 10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def choose_device(requested_device: str | None) -> torch.device:
    """
    Selects CUDA, MPS, or CPU.

    MPS is the Apple Metal backend used on Apple Silicon GPUs,
    including M1, M2, M3, and M4.
    """

    if requested_device is not None:
        requested_device = requested_device.lower()

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
            return torch.device("cuda")

        if requested_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested, but torch.backends.mps.is_available() is False.")
            return torch.device("mps")

        if requested_device == "cpu":
            return torch.device("cpu")

        raise ValueError("Unsupported device. Use one of: cuda, mps, cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sync_device(device: torch.device):
    """
    Synchronize GPU work before timing.

    CUDA and MPS operations are asynchronous, so timing without synchronization
    can under-report training time.
    """

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        # Available in modern PyTorch versions.
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def make_dataset(
    model_type: str,
    num_samples: int,
    image_size: int,
    input_dim: int,
    num_classes: int,
    dtype: torch.dtype,
):
    """
    Creates a synthetic dataset.

    Synthetic data avoids disk I/O and data augmentation bottlenecks, making this
    mostly a compute benchmark.
    """

    if model_type == "cnn":
        x = torch.randn(num_samples, 3, image_size, image_size, dtype=dtype)
    elif model_type == "mlp":
        x = torch.randn(num_samples, input_dim, dtype=dtype)
    else:
        raise ValueError("model_type must be either 'cnn' or 'mlp'")

    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


def make_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
):
    if model_type == "cnn":
        return SmallCNN(num_classes=num_classes)

    if model_type == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    raise ValueError("model_type must be either 'cnn' or 'mlp'")


def benchmark(args) -> BenchmarkResult:
    device = choose_device(args.device)

    print("=" * 80)
    print("PyTorch training benchmark")
    print("=" * 80)
    print(f"Python version:        {platform.python_version()}")
    print(f"Platform:              {platform.platform()}")
    print(f"PyTorch version:       {torch.__version__}")
    print(f"Selected device:       {device}")

    if device.type == "cuda":
        print(f"CUDA device name:      {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:          {torch.version.cuda}")

    if device.type == "mps":
        print("Apple Metal MPS:       enabled")
        print("Likely Apple Silicon:  yes, if running on M1/M2/M3/M4 hardware")

    print(f"Model:                 {args.model}")
    print(f"Epochs:                {args.epochs}")
    print(f"Batch size:            {args.batch_size}")
    print(f"Samples per epoch:     {args.num_samples}")
    print(f"Precision:             {args.precision}")
    print("=" * 80)

    if args.precision == "float32":
        dtype = torch.float32
    elif args.precision == "float16":
        dtype = torch.float16
    else:
        raise ValueError("precision must be float32 or float16")

    # On MPS, float32 is usually the safest default.
    # float16 may work for many models, but some operations may be unsupported
    # or behave differently depending on your PyTorch version.
    if device.type == "mps" and dtype == torch.float16:
        print("Warning: float16 on MPS may not support every operation. float32 is safer.")

    dataset = make_dataset(
        model_type=args.model,
        num_samples=args.num_samples,
        image_size=args.image_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        dtype=dtype,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = make_model(
        model_type=args.model,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device=device, dtype=dtype)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup avoids measuring first-time setup overhead.
    model.train()
    print(f"\nRunning {args.warmup_steps} warmup steps...")

    loader_iter = iter(loader)
    for _ in range(args.warmup_steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    sync_device(device)

    epoch_times = []
    batch_times = []

    print("\nBenchmarking...")
    for epoch in range(args.epochs):
        sync_device(device)
        epoch_start = time.perf_counter()

        running_loss = 0.0
        num_batches = 0

        for x, y in loader:
            sync_device(device)
            batch_start = time.perf_counter()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            sync_device(device)
            batch_end = time.perf_counter()

            running_loss += loss.item()
            num_batches += 1
            batch_times.append(batch_end - batch_start)

        sync_device(device)
        epoch_end = time.perf_counter()

        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        samples_per_sec = args.num_samples / epoch_time
        avg_loss = running_loss / max(num_batches, 1)

        print(
            f"Epoch {epoch + 1:>3}/{args.epochs} | "
            f"time: {epoch_time:.4f} sec | "
            f"samples/sec: {samples_per_sec:.2f} | "
            f"loss: {avg_loss:.4f}"
        )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_samples_per_sec = args.num_samples / avg_epoch_time
    avg_batch_time_ms = 1000.0 * sum(batch_times) / len(batch_times)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Device:                {device}")
    print(f"Model:                 {args.model}")
    print(f"Average epoch time:    {avg_epoch_time:.4f} sec")
    print(f"Average samples/sec:   {avg_samples_per_sec:.2f}")
    print(f"Average batch time:    {avg_batch_time_ms:.4f} ms")

    if peak_memory_mb is not None:
        print(f"Peak CUDA memory:      {peak_memory_mb:.2f} MB")
    else:
        print("Peak memory:           not reported for this backend")

    print("=" * 80)

    return BenchmarkResult(
        device=str(device),
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        samples_per_epoch=args.num_samples,
        avg_epoch_time_sec=avg_epoch_time,
        avg_samples_per_sec=avg_samples_per_sec,
        avg_batch_time_ms=avg_batch_time_ms,
        peak_memory_mb=peak_memory_mb,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use. Default: CUDA if available, else MPS if available, else CPU.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "mlp"],
        help="Benchmark model type.",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--num-classes", type=int, default=10)

    # CNN options
    parser.add_argument("--image-size", type=int, default=128)

    # MLP options
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Training precision. float32 is safest for MPS.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. Keep 0 for synthetic data and MPS benchmarking. "
            "For real datasets on CUDA, you may try 2, 4, or 8."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args)