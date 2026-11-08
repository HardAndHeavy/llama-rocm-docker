"""Verify that torch and llama_cpp share one process, the way ComfyUI runs them.

`llama_smoke.py` checks llama.cpp on its own and `rocm_smoke.py` checks the
inherited PyTorch stack on its own, each in its own interpreter. Their consumers
do not: `comfyui-rocm` derives from this image precisely so that ComfyUI and its
LLM nodes run `transformers` and `llama_cpp` from one `/opt/venv`. This script
covers that combination, because a pair of green single-purpose tests says
nothing about whether the two ROCm stacks can be loaded together.

Both import orders are checked and both must pass. They used to differ: the
llama.cpp wheel carried a DT_RPATH pointing at the deb ROCm under /opt/rocm, so
importing `llama_cpp` before `torch` pulled in a second libamd_comgr.so, and the
two LLVMs aborted the process with "Option 'spirv-expand-step' registered more
than once". The Dockerfile now rewrites that RPATH to the pip ROCm that PyTorch
uses, which is why a fresh interpreter can import either one first. The reverse
order is still checked here, so a rebuild that loses the rewrite is caught.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch


def matmul(device: torch.device, label: str) -> None:
    """Run a real kernel; a broken ROCm mapping yields zeros, not an exception."""
    torch.manual_seed(0)
    left = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    right = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    result = (left @ right).float()
    torch.cuda.synchronize(device)
    if result.shape != (1024, 1024) or not torch.isfinite(result).all().item():
        raise RuntimeError(f"{label}: the GPU matrix multiplication produced invalid output")
    if not torch.any(result != 0).item():
        raise RuntimeError(f"{label}: the GPU matrix multiplication produced only zeros")
    print(f"ok   {label} — {result.device}, {torch.cuda.memory_allocated(device) / 1024**2:.0f} MiB")


def generate(llama_cpp) -> None:
    """Offload a real model while torch holds the device, if one is on disk.

    Backend initialization proves the library loads; only a generation proves
    llama.cpp kernels run next to an active torch context. The model is not
    downloaded here: `make coexist-test` stays cheap, and `make test` already
    owns the download. Run `make download-model` first to include this step.
    """
    model_path = Path(os.getenv("MODEL_PATH", ""))
    if not model_path.is_file():
        print(f"skip GGUF generation — no model at {model_path or '$MODEL_PATH'}")
        return

    model = llama_cpp.Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=512,
        seed=0,
        verbose=False,
    )
    completion = model.create_completion(
        "The capital of France is", max_tokens=8, temperature=0.0, seed=0
    )
    text = completion["choices"][0]["text"].strip()
    if not text:
        raise RuntimeError("llama.cpp produced an empty completion next to torch")
    print(f"ok   GGUF generation next to torch — {model_path.name}: {text!r}")
    del model


def torch_then_llama(device: torch.device) -> None:
    """The order ComfyUI uses: torch kernels, then llama.cpp, then torch again."""
    matmul(device, "torch before llama_cpp")

    import llama_cpp

    info = llama_cpp.llama_print_system_info().decode().strip()
    if "ROCm" not in info:
        raise RuntimeError(f"llama.cpp has no ROCm backend compiled in:\n{info}")
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError(
            "llama.cpp reports no GPU offload support with torch already loaded"
        )
    llama_cpp.llama_backend_init()
    print(f"ok   llama_cpp after torch — {llama_cpp.__version__}, GPU offload available")

    generate(llama_cpp)

    # Loading the second ROCm stack must not disturb the first: this is the
    # state ComfyUI is in when an LLM node runs mid-generation.
    matmul(device, "torch after llama_cpp")


def llama_then_torch() -> None:
    """The reverse order in a fresh interpreter, which must work as well.

    A script, or a node that shells out to one, may well import `llama_cpp`
    first. That used to abort the process before any Python code could handle
    it; the RPATH rewrite in the Dockerfile is what makes it work, so it is
    worth its own process rather than a second import in this one.
    """
    program = textwrap.dedent(
        """
        import llama_cpp
        import torch

        assert llama_cpp.llama_supports_gpu_offload(), "no GPU offload"
        assert torch.cuda.is_available(), "torch cannot see the GPU"
        print("reverse order ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        tail = "\n      ".join(detail[-4:]) if detail else f"exit code {result.returncode}"
        raise RuntimeError(
            "importing llama_cpp before torch fails; the RPATH rewrite in the "
            "Dockerfile is not in effect, so the wheel resolves libamd_comgr "
            "from the deb ROCm instead of the pip one:\n"
            f"      {tail}"
        )
    print("ok   llama_cpp before torch — reverse order works in a fresh interpreter")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No ROCm GPU is visible. Pass --device=/dev/kfd and --device=/dev/dri "
            "to the container and use a compatible AMDGPU kernel driver."
        )

    device = torch.device("cuda:0")
    print(f"torch: {torch.__version__}; HIP: {torch.version.hip}")
    print(f"GPU: {torch.cuda.get_device_properties(device).name}")

    torch_then_llama(device)
    llama_then_torch()

    print("torch and llama_cpp coexist smoke test passed.")


if __name__ == "__main__":
    main()
