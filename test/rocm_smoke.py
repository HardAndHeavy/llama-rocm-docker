"""Re-run the base image's ROCm check inside the derived image.

`make smoke-test` covers what this repository adds (llama.cpp on ROCm).
This script covers what it inherits: the pinned PyTorch stack still imports
after the image has been layered on top of the base, the three torch packages
still come from one coherent ROCm build, and a real GPU kernel still executes.
"""

import platform

import torch
import torchaudio
import torchvision


def main() -> None:
    modules = (torch, torchaudio, torchvision)

    for module in modules:
        print(f"{module.__name__}: {module.__version__}")

    # A base image rebuild that swaps one component (or a pip install here that
    # pulls a plain CUDA/CPU wheel over the pinned one) leaves torch, torchvision
    # and torchaudio on different local version labels. They must agree.
    suffixes = {}
    for module in modules:
        version = module.__version__
        suffixes[module.__name__] = version.split("+", 1)[1] if "+" in version else None
    if len(set(suffixes.values())) != 1:
        raise RuntimeError(f"torch packages come from different builds: {suffixes}")
    suffix = next(iter(suffixes.values()))
    if suffix is None:
        raise RuntimeError(
            "The PyTorch packages have no local version label; a CPU/CUDA wheel "
            f"has replaced the ROCm build: {suffixes}"
        )

    if torch.version.hip is None:
        raise RuntimeError("The installed PyTorch build does not include ROCm/HIP")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm GPU is visible. Pass /dev/kfd and /dev/dri to the container "
            "and use a compatible AMDGPU kernel driver."
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    print(f"Python: {platform.python_version()}")
    print(f"HIP: {torch.version.hip}")
    print(f"ROCm build: {suffix}")
    print(f"GPU: {properties.name}; GCN arch: {properties.gcnArchName}")
    print(f"GPU memory: {properties.total_memory / 1024**3:.1f} GiB")

    # Force kernel initialization and execution instead of stopping at device
    # discovery. This catches missing architecture kernels and broken mappings.
    torch.manual_seed(0)
    left = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    right = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    result = (left @ right).float()
    torch.cuda.synchronize(device)

    if result.shape != (1024, 1024) or not torch.isfinite(result).all().item():
        raise RuntimeError("The GPU matrix multiplication produced invalid output")
    if not torch.any(result != 0).item():
        raise RuntimeError("The GPU matrix multiplication produced only zeros")

    print(
        "BF16 matrix multiplication passed; "
        f"allocated GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MiB"
    )
    print("Base ROCm runtime smoke test passed.")


if __name__ == "__main__":
    main()
