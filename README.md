# LFM3 + llama.cpp + ROCm in Docker

A Linux/amd64 image for running GGUF models with
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) on AMD Radeon
GPUs. Published as
[`hardandheavy/llama-rocm`](https://hub.docker.com/r/hardandheavy/llama-rocm).

The repository includes an interactive LFM3 chat, a non-interactive model
generation test, and a fast ROCm backend check. The default model is
[`LiquidAI/LFM3-1.7B-Q4_K_M.gguf`](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/main/LFM3-1.7B-Q4_K_M.gguf).

## Software stack

| Component | Version |
| --- | --- |
| Base image | `hardandheavy/transformers-rocm:3.0.0` |
| ROCm | 10.1.0 |
| llama-cpp-python | 0.3.36, pinned source revision |
| llama.cpp | v0.9.0, pinned source revision |
| Default model | LFM3 1.7B, Q4_K_M GGUF |

The released ROCm wheel of `llama-cpp-python` vendors an older llama.cpp without
LFM3 support. The image therefore builds the pinned Python binding with the
pinned llama.cpp `v0.9.0` backend and HIP enabled. Prebuilt GPU targets are
`gfx1030`, `gfx1100`, `gfx1101`, `gfx1200`, and `gfx1201`.

## Requirements

- Linux x86_64 with an AMD GPU supported by the image's ROCm backend.
- A compatible AMDGPU kernel driver exposing `/dev/kfd` and `/dev/dri`.
- Docker Engine, Git, and `make`.

The ROCm user-space stack is installed inside the image; installing it on the
host is not required. For host driver installation, see the
[AMD ROCm Linux installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

## Build and test

```bash
git clone https://github.com/HardAndHeavy/llama-rocm-docker.git
cd llama-rocm-docker

make check                 # Dockerfile static checks and Python syntax
make build tag=3.0.0       # creates llama-rocm:3.0.0 and :latest
make smoke-test tag=3.0.0  # checks ROCm initialization and GPU offload support
make test tag=3.0.0        # downloads LFM3, loads it on the GPU, and generates text
```

`3.0.0` is the default `tag`, so it can be omitted. The source build compiles
llama.cpp and can take some time. Both runtime test targets require a real AMD
GPU; `docker build` verifies imports but cannot verify host GPU access.

`make test` runs `python test/lfm3.py --auto-test` inside the container. It
validates a complete, non-empty model response rather than comparing generated
text with a fixed answer.

## Model download

`make test` and `make bash` automatically run the download target when needed.
To download explicitly:

```bash
make download-model
```

The pinned model revision is saved as
`./huggingface/LFM3-1.7B-Q4_K_M.gguf` (about 1.10 GB). Interrupted downloads are
resumed, and a completed download is verified against its SHA256 checksum before
being renamed. The `huggingface` directory is ignored by Git and is also mounted
as the Hugging Face cache inside the container.

## Interactive chat

After building the image:

```bash
make bash
```

This starts `test/lfm3.py` directly with a terminal attached. The model is loaded
once and conversation history is preserved between messages. Use `/clear` to
reset history, `/help` to list commands, and `/exit`, `/quit`, or `Ctrl+D` to
leave the chat.

For a normal container shell instead of the chat:

```bash
make bash-dev
```

The repository is mounted at `/app`. To override generation limits through the
Make targets:

```bash
make test MAX_TOKENS=256 N_CTX=8192
make bash MAX_TOKENS=256
```

Or run the script with explicit arguments inside `make bash-dev`:

```bash
python test/lfm3.py --auto-test --max-tokens 256 --n-ctx 8192
python test/lfm3.py --model-path /app/huggingface/LFM3-1.7B-Q4_K_M.gguf
```

The sampling defaults follow the LFM3 model card: temperature `0.1`, top-k `50`,
and repeat penalty `1.1`. All model layers are placed on the GPU by default.

## Publishing a release

After logging in with `docker login` and passing all checks on an AMD GPU host:

```bash
make publish tag=3.0.0
```

This pushes both `hardandheavy/llama-rocm:3.0.0` and `:latest`.
`make publish tag=latest` is rejected to avoid an unversioned release.

## License

The original code and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). Third-party software and models remain subject to
their respective licenses. In particular, the downloaded LFM3 model is provided
under the [Liquid AI LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/main/LICENSE).
