# Changelog

Image versions are independent of the ROCm, PyTorch and llama.cpp versions.
Only changes that affect the published image or the repository commands are
listed.

## Unreleased

- `make bash` now opens a plain shell and no longer downloads the test model
  first; `make chat` is the interactive LFM3 chat. The old split (`bash` for
  chat, `bash-dev` for a shell) made the same target mean different things here
  and in `transformers-rocm-docker`.
- Run targets (`bash`, `chat`, `download-model`, `smoke-test`, `test`,
  `base-smoke-test`) use the published image by default, so a fresh clone works
  without compiling llama.cpp. Pass `LOCAL=1` to target the image `make build`
  produced; `build` and `publish` are unchanged and still operate on the local
  image only.

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
