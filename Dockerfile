FROM hardandheavy/transformers-rocm:3.0.0

ARG IMAGE_VERSION=3.0.0
ARG IMAGE_REVISION=unknown
ARG IMAGE_CREATED=1970-01-01T00:00:00Z
ARG LLAMA_CPP_PYTHON_VERSION=0.3.36
ARG LLAMA_CPP_PYTHON_REVISION=1652066e0af45f2313b339670ef9555e8a54e545
# Pin llama.cpp v0.9.0; its tag dereferences to this commit.
ARG LLAMA_CPP_REVISION=cd959bae99ff4bed22bef2604f45241c09bbb8c8
ARG AMDGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1200;gfx1201"

LABEL org.opencontainers.image.title="llama-rocm" \
      org.opencontainers.image.description="llama-cpp-python with ROCm support for AMD Radeon GPUs" \
      org.opencontainers.image.source="https://github.com/HardAndHeavy/llama-rocm-docker" \
      org.opencontainers.image.url="https://hub.docker.com/r/hardandheavy/llama-rocm" \
      org.opencontainers.image.documentation="https://github.com/HardAndHeavy/llama-rocm-docker#readme" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.base.name="hardandheavy/transformers-rocm:3.0.0"

ENV LLAMA_CPP_PYTHON_VERSION=${LLAMA_CPP_PYTHON_VERSION} \
    LLAMA_CPP_PYTHON_REVISION=${LLAMA_CPP_PYTHON_REVISION} \
    LLAMA_CPP_REVISION=${LLAMA_CPP_REVISION}

RUN python -m pip install --no-cache-dir cmake==4.4.4 ninja==1.13.2 \
    && git init /tmp/llama-cpp-python \
    && git -C /tmp/llama-cpp-python remote add origin https://github.com/abetlen/llama-cpp-python.git \
    && git -C /tmp/llama-cpp-python fetch --depth 1 origin ${LLAMA_CPP_PYTHON_REVISION} \
    && git -C /tmp/llama-cpp-python checkout --detach FETCH_HEAD \
    && git init /tmp/llama-cpp-python/vendor/llama.cpp \
    && git -C /tmp/llama-cpp-python/vendor/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git -C /tmp/llama-cpp-python/vendor/llama.cpp fetch --depth 1 origin ${LLAMA_CPP_REVISION} \
    && git -C /tmp/llama-cpp-python/vendor/llama.cpp checkout --detach FETCH_HEAD \
    && test -f /tmp/llama-cpp-python/vendor/llama.cpp/src/models/lfm3.cpp \
    && CMAKE_ARGS="-DGGML_HIP=ON -DGGML_NATIVE=OFF -DLLAVA_BUILD=OFF -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DAMDGPU_TARGETS=${AMDGPU_TARGETS}" \
        python -m pip install --no-cache-dir /tmp/llama-cpp-python \
    && python -m pip uninstall -y cmake ninja \
    && rm -rf /tmp/llama-cpp-python \
    && python -m pip check \
    && python -c 'import os, llama_cpp; assert llama_cpp.__version__ == os.environ["LLAMA_CPP_PYTHON_VERSION"]; print("llama-cpp-python:", llama_cpp.__version__)'
# GPU access requires /dev/kfd and /dev/dri and is checked by `make smoke-test`.

WORKDIR /app
CMD ["bash"]
