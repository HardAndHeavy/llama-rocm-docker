# syntax=docker/dockerfile:1.7
#
# llama-cpp-python with a HIP-enabled llama.cpp backend, on top of the published
# transformers-rocm image. The base image provides ROCm, the pinned PyTorch
# stack and a CPython 3.14 virtualenv at /opt/venv; nothing ROCm-specific is
# installed here.
#
# The build is split in two stages on purpose. Compiling llama.cpp needs CMake,
# Ninja and a C++ toolchain; installing them in the final stage would leave them
# in a lower layer even after a later `pip uninstall`. The builder stage produces
# a wheel and is thrown away, so the shipped image carries the compiled binaries
# and nothing else.

# BEGIN GENERATED: tools/versions.py args
ARG IMAGE_VERSION=3.0.1
ARG BASE_IMAGE=hardandheavy/transformers-rocm:3.0.1@sha256:a46fe7a3efd7c7f87a20d63f58bed76c499604d959597584ffb117c2b336c74c
ARG LLAMA_CPP_PYTHON_VERSION=0.3.36
ARG LLAMA_CPP_PYTHON_REVISION=1652066e0af45f2313b339670ef9555e8a54e545
ARG LLAMA_CPP_VERSION=v0.9.0
ARG LLAMA_CPP_REVISION=cd959bae99ff4bed22bef2604f45241c09bbb8c8
ARG AMDGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1200;gfx1201"
# END GENERATED

# ---------------------------------------------------------------------------
# Stage 1: compile llama-cpp-python against the pinned llama.cpp revision.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ARG LLAMA_CPP_PYTHON_VERSION
ARG LLAMA_CPP_PYTHON_REVISION
ARG LLAMA_CPP_VERSION
ARG LLAMA_CPP_REVISION
ARG AMDGPU_TARGETS

# The toolchain comes from a hash-pinned lock so that rebuilding the wheel does
# not depend on whatever versions PyPI serves that day. `--no-build-isolation`
# below makes scikit-build-core, CMake and Ninja resolve from this exact set
# instead of pip downloading a build backend mid-build.
COPY requirements-build.lock /opt/build/requirements-build.lock
RUN python -m pip install --no-cache-dir --require-hashes \
        --requirement /opt/build/requirements-build.lock

# The wheel is imported below to prove the HIP backend was actually produced, and
# importing llama_cpp pulls in its runtime dependencies. Installing the runtime
# lock here as well keeps that check honest instead of stubbing the import.
COPY requirements-runtime.lock /opt/build/requirements-runtime.lock
RUN python -m pip install --no-cache-dir --require-hashes \
        --requirement /opt/build/requirements-runtime.lock

# Both repositories are fetched by object id, not by tag: a tag can be moved, a
# commit cannot. Fetching the exact object also avoids cloning the full
# multi-hundred-megabyte history of either project.
RUN set -eux; \
    git init /tmp/llama-cpp-python; \
    git -C /tmp/llama-cpp-python remote add origin https://github.com/abetlen/llama-cpp-python.git; \
    git -C /tmp/llama-cpp-python fetch --depth 1 origin "${LLAMA_CPP_PYTHON_REVISION}"; \
    git -C /tmp/llama-cpp-python checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/llama-cpp-python rev-parse HEAD)" = "${LLAMA_CPP_PYTHON_REVISION}"; \
    rm -rf /tmp/llama-cpp-python/vendor/llama.cpp; \
    git init /tmp/llama-cpp-python/vendor/llama.cpp; \
    git -C /tmp/llama-cpp-python/vendor/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git; \
    git -C /tmp/llama-cpp-python/vendor/llama.cpp fetch --depth 1 origin "${LLAMA_CPP_REVISION}"; \
    git -C /tmp/llama-cpp-python/vendor/llama.cpp checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/llama-cpp-python/vendor/llama.cpp rev-parse HEAD)" = "${LLAMA_CPP_REVISION}"

# The upstream sdist vendors an older llama.cpp without LFM3 support, which is
# why the Python binding is compiled from source against a newer backend.
# LLAVA_BUILD is off: the multimodal library is another link step and none of
# the text-only tests use it.
#
# CMAKE_HIP_COMPILER is set explicitly and is not optional. The base image
# installs the `rocm` PyPI meta-package into /opt/venv, and it ships hipcc and
# hipconfig wrappers whose ROCm root is the pip package directory
# (`_rocm_sdk_core`), not the deb-installed /opt/rocm. Because
# /opt/venv/bin precedes /opt/rocm/bin on PATH, `enable_language(HIP)` picks up
# that wrapper and then fails with "does not contain the HIP runtime CMake
# package". Naming the deb-installed compiler makes CMake derive the ROCm root
# from it (/opt/rocm/core-10.1) instead.
ENV CMAKE_ARGS="-DGGML_HIP=ON -DGGML_NATIVE=OFF -DLLAVA_BUILD=OFF -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DAMDGPU_TARGETS=${AMDGPU_TARGETS}"

RUN set -eux; \
    python -m pip wheel --no-cache-dir --no-build-isolation --no-deps \
        --wheel-dir /opt/wheels /tmp/llama-cpp-python; \
    ls -la /opt/wheels

# Verify the wheel before it leaves the builder: the version must be the pinned
# one, the vendored backend must be the pinned llama.cpp release, and the HIP
# backend must actually have been produced. A silent CPU-only fallback is the
# classic failure mode of a HIP build.
RUN set -eux; \
    python -m pip install --no-cache-dir --no-deps /opt/wheels/llama_cpp_python-*.whl; \
    python - <<'PY'
import os
import pathlib
import sys

import llama_cpp

expected = os.environ["LLAMA_CPP_PYTHON_VERSION"]
assert llama_cpp.__version__ == expected, (llama_cpp.__version__, expected)

lib_dir = pathlib.Path(llama_cpp.__file__).parent / "lib"
hip = lib_dir / "libggml-hip.so"
assert hip.is_file(), f"HIP backend missing: {sorted(p.name for p in lib_dir.iterdir())}"

# libllama.so.<llama.cpp release> proves which backend version was linked in.
backend = os.environ["LLAMA_CPP_VERSION"].removeprefix("v")
assert (lib_dir / f"libllama.so.{backend}").is_file(), sorted(
    p.name for p in lib_dir.iterdir()
)

# LFM3 is a llama.cpp architecture added after the revision the PyPI sdist
# vendors, so its presence is direct evidence that our checkout was compiled.
if b"lfm3" not in (lib_dir / "libllama.so").read_bytes():
    sys.exit("libllama.so has no LFM3 support; the wrong llama.cpp was built")

print(f"llama-cpp-python: {llama_cpp.__version__}")
print(f"llama.cpp: {os.environ['LLAMA_CPP_VERSION']}")
print(f"HIP backend: {hip.name} ({hip.stat().st_size / 1024**2:.0f} MiB)")
PY

# ---------------------------------------------------------------------------
# Stage 2: the shipped image.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

ARG IMAGE_VERSION
ARG IMAGE_REVISION=unknown
ARG IMAGE_CREATED=1970-01-01T00:00:00Z
ARG LLAMA_CPP_PYTHON_VERSION
ARG LLAMA_CPP_PYTHON_REVISION
ARG LLAMA_CPP_VERSION
ARG LLAMA_CPP_REVISION
ARG AMDGPU_TARGETS

# BEGIN GENERATED: tools/versions.py labels
LABEL org.opencontainers.image.title="llama-rocm" \
      org.opencontainers.image.description="llama.cpp with ROCm support for AMD Radeon GPUs" \
      org.opencontainers.image.source="https://github.com/HardAndHeavy/llama-rocm-docker" \
      org.opencontainers.image.url="https://hub.docker.com/r/hardandheavy/llama-rocm" \
      org.opencontainers.image.documentation="https://github.com/HardAndHeavy/llama-rocm-docker#readme" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.base.name="hardandheavy/transformers-rocm:3.0.1" \
      org.opencontainers.image.base.digest="sha256:a46fe7a3efd7c7f87a20d63f58bed76c499604d959597584ffb117c2b336c74c" \
      ru.hardandheavy.llama-cpp-python.version="0.3.36" \
      ru.hardandheavy.llama-cpp-python.revision="1652066e0af45f2313b339670ef9555e8a54e545" \
      ru.hardandheavy.llama-cpp.version="v0.9.0" \
      ru.hardandheavy.llama-cpp.revision="cd959bae99ff4bed22bef2604f45241c09bbb8c8" \
      ru.hardandheavy.amdgpu-targets="gfx1030;gfx1100;gfx1101;gfx1200;gfx1201"
# END GENERATED

# Recorded for operators and for the assertions below. `llama_cpp.__version__`
# reads the package metadata, so these are documentation, not the source of
# truth for the check.
ENV LLAMA_CPP_PYTHON_VERSION=${LLAMA_CPP_PYTHON_VERSION} \
    LLAMA_CPP_PYTHON_REVISION=${LLAMA_CPP_PYTHON_REVISION} \
    LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION} \
    LLAMA_CPP_REVISION=${LLAMA_CPP_REVISION} \
    AMDGPU_TARGETS=${AMDGPU_TARGETS}

# Only what llama-cpp-python imports but the base image does not already ship
# (diskcache), installed from a hash-pinned lock. numpy, jinja2 and
# typing-extensions are already present with versions the base image guarantees,
# and the wheel is installed with --no-deps so it cannot pull in its own copies
# and silently replace the pinned stack.
COPY requirements-runtime.lock /opt/requirements/requirements-runtime.lock
COPY --from=builder /opt/wheels/llama_cpp_python-*.whl /opt/wheels/
RUN set -eux; \
    python -m pip install --no-cache-dir --require-hashes \
        --requirement /opt/requirements/requirements-runtime.lock; \
    python -m pip install --no-cache-dir --no-deps /opt/wheels/llama_cpp_python-*.whl; \
    rm -rf /opt/wheels /opt/requirements; \
    python -m pip check

# Fail the build, not the first request, if an import or the backend is missing.
# This needs no GPU: llama.cpp loads HIP kernels lazily, so the check runs on any
# builder host. GPU access is verified by the healthcheck and by `make test`.
RUN python - <<'PY'
import llama_cpp

print(f"llama-cpp-python: {llama_cpp.__version__}")
print(llama_cpp.llama_print_system_info().decode().strip().splitlines()[0])
PY

# The base image's healthcheck only proves that PyTorch can see the GPU. This
# one fails if the llama.cpp backend could not be initialized on the device,
# which is the failure mode specific to this image. It requires the container to
# be started with --device=/dev/kfd and --device=/dev/dri, like every run
# command in the README.
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'ROCm GPU is not available'; llama_cpp.llama_backend_init()"

WORKDIR /app
CMD ["bash"]
