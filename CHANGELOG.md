# Changelog

Image versions are independent of the ROCm, PyTorch and llama.cpp versions.
Only changes that affect the published image or the repository commands are
listed.

## 3.0.2

Fix-up release for 3.0.1. No change to the pinned stack: same base image, same
llama.cpp and llama-cpp-python revisions. The change is to how the llama.cpp
libraries find ROCm.

### Fixed: importing `llama_cpp` before `torch` aborted the process

PyTorch loads ROCm from the pip `rocm-sdk-*` packages in `/opt/venv`, while
llama.cpp is compiled against the deb ROCm under `/opt/rocm`. Until this
release, `llama-cpp-python`'s CMakeLists turned on
`CMAKE_INSTALL_RPATH_USE_LINK_PATH`, which recorded that deb link directory in
the `DT_RPATH` of `libllama.so` and `libggml.so`. The two stacks then loaded
different copies of `libamd_comgr.so`, each containing an LLVM, and the second
registration aborted the process:

```
: CommandLine Error: Option 'spirv-expand-step' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```

Importing `torch` first happened to hide it, because the pip copy was already
resident. `comfyui-rocm` imports `torch` during startup and was therefore
unaffected, but any script or custom-node subprocess that imported `llama_cpp`
first would die before Python could report anything.

- The builder stage now repacks the wheel with the RPATH of every compiled
  library pointing at the pip ROCm first and the deb install as a fallback
  (`libomp` is only shipped by the deb LLVM). The variables could not be set
  from `CMAKE_ARGS`: llama-cpp-python sets them in its CMakeLists without
  `CACHE`, so they win over anything on the command line.
- `patchelf` is added to the build lock and used only in the builder stage; the
  shipped image does not contain it.
- Add `test/rocm_resolution.py`, run during the build and on demand: it fails
  if any library in the package resolves `libamd_comgr` or `libamdhip64`
  outside the pip ROCm. This catches a base image that moves the pip packages.
- Add `test/coexist_smoke.py` (`make coexist-test`). The existing tests load
  PyTorch and llama.cpp in separate processes, but `comfyui-rocm` runs both in
  one. It runs a PyTorch kernel, initializes llama.cpp, generates on the GPU
  when the test model is present, runs a PyTorch kernel again, and then checks
  the reverse import order in a fresh interpreter. Both orders must pass.
- README: show the image chain and say exactly what each image is a base for.
  The old wording called `transformers-rocm` "the base for every GPU project
  here", which is not true of every repository.

## Unreleased

- `make bash` now opens a plain shell and no longer downloads the test model
  first; `make chat` is the interactive LFM3 chat. The old split (`bash` for
  chat, `bash-dev` for a shell) made the same target mean different things here
  and in `transformers-rocm-docker`.
- Run targets (`bash`, `chat`, `download-model`, `smoke-test`, `test`,
  `base-smoke-test`, `coexist-test`) use the published image by default, so a
  fresh clone works without compiling llama.cpp. Pass `LOCAL=1` to target the
  image `make build` produced; `build` and `publish` are unchanged and still
  operate on the local image only.

## 3.0.1

Rebuilt on `transformers-rocm:3.0.1` (ROCm 10.1.0, Ubuntu 26.04, Python 3.14,
PyTorch 2.13.0), with llama.cpp v0.9.0 and llama-cpp-python 0.3.36 compiled from
source.

- Fix the build against the new base image. The base ships the `rocm` PyPI
  meta-package into `/opt/venv`, whose `hipcc`/`hipconfig` wrappers shadow the
  deb-installed ones on `PATH`. `enable_language(HIP)` followed that wrapper to
  `_rocm_sdk_core`, which has no `hip-lang` CMake package, and the build failed
  with "does not contain the HIP runtime CMake package". `CMAKE_HIP_COMPILER`
  now names `/opt/rocm/llvm/bin/clang++`, so CMake derives the ROCm root from
  the deb installation.
- Fix the base-image pin, which recorded the digest of
  `rocm/dev-ubuntu-26.04:10.1.0-full` instead of
  `hardandheavy/transformers-rocm:3.0.1`; every build failed with "not found".
  `tools/pin_base.py` (`make pin-base`) now resolves the pin from the registry
  and refuses to write it when the versioned tag and `latest` disagree.
- Install the runtime lock in the builder stage too, so the wheel verification
  can import `llama_cpp` instead of failing on a missing `diskcache`.
- Add `test/llama_smoke.py`: `make smoke-test` previously ran the inherited
  PyTorch check and never exercised llama.cpp. It now asserts the ROCm backend
  is compiled in and initializes a device.
- Add `test/rocm_smoke.py` for `make base-smoke-test`, which referenced a
  nonexistent `test/base_smoke.py`, and make it detect a non-ROCm PyTorch wheel
  by comparing the local version labels of torch, torchvision and torchaudio.
- Rename the test flag to `--test` in `test/lfm3.py`, matching what `make test`
  has always passed; the script only knew `--auto-test`, so the target failed.
- Honour `GPU_LAYERS`/`--gpu-layers` instead of hard-coding full offload, which
  the Makefile was already setting and the script was ignoring.
- Add `make pin-base-check` and `make base-smoke-test` to `.PHONY`, and include
  the helper scripts in the `make check` syntax check.
- Write a real README; it contained a `# placeholder` line.

## 3.0.0

- ROCm 10.1.0 base, llama.cpp v0.9.0 with LFM3 support.
