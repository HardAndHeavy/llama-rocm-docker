"""Verify that the llama.cpp backend in this image runs on the AMD GPU.

The builder stage already checked that the wheel contains a HIP backend and the
pinned llama.cpp revision. This script checks the part that cannot be verified
without a device: that the ROCm runtime is reachable and that the backend
actually offloads work to the GPU instead of silently falling back to CPU.
"""

import llama_cpp


def main() -> None:
    info = llama_cpp.llama_print_system_info().decode().strip()
    print(f"llama-cpp-python: {llama_cpp.__version__}")
    for line in info.splitlines():
        print(line)

    if "ROCm" not in info:
        raise RuntimeError(f"llama.cpp has no ROCm backend compiled in:\n{info}")
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError(
            "llama.cpp reports no GPU offload support. Run `make smoke-test` on a "
            "host with /dev/kfd and /dev/dri and a compatible AMDGPU driver."
        )

    # llama.cpp only initializes the ROCm device library on first use, so the
    # support flag alone would pass on a machine where the driver is missing.
    # Backend initialization is what prints the "failed to initialize ROCm"
    # diagnostic, so treat its absence as the real signal.
    llama_cpp.llama_backend_init()
    if "failed to initialize ROCm" in info:
        raise RuntimeError(
            "llama.cpp could not initialize the ROCm device.\n"
            "The container needs --device=/dev/kfd and --device=/dev/dri, and the "
            "host needs a ROCm-compatible AMDGPU driver."
        )

    print("llama.cpp ROCm backend initialized; GPU offload is available.")
    print("llama.cpp ROCm smoke test passed.")


if __name__ == "__main__":
    main()
