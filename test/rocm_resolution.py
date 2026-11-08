"""Check that llama.cpp resolves the same ROCm libraries as PyTorch.

PyTorch loads ROCm from the pip rocm-sdk packages in /opt/venv. If any library
in the llama_cpp package resolves libamd_comgr or libamdhip64 from the deb ROCm
under /opt/rocm instead, a process that imports llama_cpp before torch ends up
with two LLVMs and aborts. The Dockerfile rewrites the RPATH to prevent this;
this script verifies the result statically, with `ldd`, so it needs no GPU and
runs during the image build as well as on demand:

    docker run --rm -v "$PWD/test:/app/test:ro" IMAGE python /app/test/rocm_resolution.py
"""

from __future__ import annotations

import pathlib
import site
import subprocess
import sys

# The two libraries that carry an LLVM. Resolving either from a second location
# is what causes the duplicate command-line option registration.
WATCHED = ("libamd_comgr", "libamdhip64")


def main() -> None:
    packages = pathlib.Path(site.getsitepackages()[0])
    lib_dir = packages / "llama_cpp" / "lib"
    pip_rocm = (packages / "_rocm_sdk_core" / "lib").resolve()

    libraries = sorted(p for p in lib_dir.glob("*.so*") if not p.is_symlink())
    if not libraries:
        sys.exit(f"no shared libraries under {lib_dir}")

    resolved: dict[str, set[str]] = {}
    for library in libraries:
        out = subprocess.run(
            ["ldd", str(library)], capture_output=True, text=True, check=True
        ).stdout
        for line in out.splitlines():
            if "=>" not in line or not any(name in line for name in WATCHED):
                continue
            target = line.split("=>", 1)[1].split("(", 1)[0].strip()
            if target == "not found":
                sys.exit(f"{library.name}: {line.strip()}")
            resolved.setdefault(str(pathlib.Path(target).resolve()), set()).add(library.name)

    if not resolved:
        sys.exit("no library in llama_cpp links libamd_comgr or libamdhip64")

    foreign = {t: users for t, users in resolved.items() if not t.startswith(str(pip_rocm))}
    for target in sorted(resolved):
        print(f"{'ok  ' if target not in foreign else 'FAIL'} {target}")
    if foreign:
        sys.exit(
            "llama_cpp resolves ROCm outside the pip rocm-sdk packages PyTorch uses "
            f"({pip_rocm}); importing llama_cpp before torch will abort. "
            "The RPATH rewrite in the Dockerfile builder stage did not take effect."
        )
    print(f"llama_cpp resolves ROCm from {pip_rocm}, the same copy as PyTorch.")


if __name__ == "__main__":
    main()
