# llama.cpp + ROCm in Docker

A reproducible Linux/amd64 image for running [llama.cpp](https://github.com/ggml-org/llama.cpp)
on AMD Radeon GPUs, published as
[`hardandheavy/llama-rocm`](https://hub.docker.com/r/hardandheavy/llama-rocm).

It is built on top of
[`transformers-rocm`](https://github.com/HardAndHeavy/transformers-rocm-docker),
which supplies ROCm, the pinned PyTorch stack and a CPython virtualenv. This
repository adds only the HIP-enabled `llama-cpp-python` binding, compiled from
source against a pinned llama.cpp revision.

The images form one chain:

```
rocm/dev-ubuntu-26.04      ROCm, Ubuntu
  └─ transformers-rocm     pinned PyTorch, Transformers, venv at /opt/venv
       └─ llama-rocm       this image: llama-cpp-python with a HIP backend
            └─ comfyui-rocm   ComfyUI, on the same /opt/venv
```

The split is deliberate.
[`comfyui-rocm-docker`](https://github.com/HardAndHeavy/comfyui-rocm-docker)
derives from this image so that ComfyUI and its LLM nodes find a working
`llama-cpp-python` in the same Python environment as PyTorch, and pins it so a
custom node cannot replace it. Keeping the compile in its own image means
ComfyUI inherits a binary that was built and tested here, including in the same
process as PyTorch (`make coexist-test`), instead of building llama.cpp on every
user's machine. Both stacks resolve one copy of ROCm, so a consumer may import
either one first; see
[Using this image from another project](#using-this-image-from-another-project).

## Version 3.0.2 software stack

See the [changelog](CHANGELOG.md) for release highlights. The image version is
independent of the ROCm, PyTorch and llama.cpp versions.

<!-- versions:begin -->
| Component | Version |
| --- | --- |
| Base image | hardandheavy/transformers-rocm:3.0.1@sha256:a46fe7a3efd7c7f87a20d63f58bed76c499604d959597584ffb117c2b336c74c |
| llama-cpp-python | 0.3.36 (source build) |
| llama.cpp | v0.9.0 (source build) |
| Python | 3.14 |
| HIP targets | gfx1030;gfx1100;gfx1101;gfx1200;gfx1201 |
| Default test model | LFM3-1.7B-Q4_K_M.gguf (LFM3-1.7B-GGUF) |
<!-- versions:end -->

`versions.json` is the single source of truth for these pins. The Dockerfile
build arguments, the OCI labels and the table above are generated from it by
`tools/versions.py`, so they cannot drift apart. Edit `versions.json`, then run
`make versions`. `make check` fails if a generated file is out of date.

## Requirements

- Linux x86_64 with an AMD GPU supported by ROCm 10.1.
- A compatible AMDGPU kernel driver exposing `/dev/kfd` and `/dev/dri`.
- Docker Engine with BuildKit.
- Git and `make` for the repository commands.
- [`uv`](https://docs.astral.sh/uv/) 0.13.7 only when intentionally regenerating
  the dependency locks; the version is pinned in the `Makefile` and checked by
  `make lock`.

The ROCm user-space stack ships inside the image; installing it on the host is
not required. For the host driver, follow the
[AMD ROCm Linux installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

## Quick start with Docker Hub

```bash
docker run --rm -it \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  hardandheavy/llama-rocm:3.0.2
```

`latest` points to the same release after publication, but the numbered tag is
recommended for reproducible deployments.

## Build and test from source

```bash
git clone https://github.com/HardAndHeavy/llama-rocm-docker.git
cd llama-rocm-docker

make check                 # Dockerfile lint, version sync, Python syntax
make build                 # creates llama-rocm:3.0.2
make base-smoke-test       # the inherited PyTorch/ROCm stack still works
make smoke-test            # llama.cpp reports GPU offload
make coexist-test          # torch and llama.cpp work together in one process
make test                  # loads LFM3 into the GPU and generates a response
```

`3.0.2` is the default `tag`, so explicit `tag=` arguments can be omitted.

Run targets use the published image by default, so a fresh clone can skip the
build entirely and start with `make bash`. After `make build`, pass `LOCAL=1` to
run against what you just built — that is the image the release is verified
against:

```bash
make build
make LOCAL=1 smoke-test
make LOCAL=1 test
```

Compiling llama.cpp is the expensive part of the build and it happens in a
throwaway builder stage: CMake, Ninja and the C++ toolchain never reach the
shipped image. The wheel is verified before it leaves the builder — the version
must be the pinned one, `libggml-hip.so` must exist, and `libllama.so` must
contain the LFM3 architecture. A silent CPU-only fallback is the classic
failure mode of a HIP build, and it is caught at build time instead of at the
first request.

`make smoke-test` and `make base-smoke-test` each load only one of the two ROCm
stacks. `make coexist-test` loads both in one interpreter, as ComfyUI does: it
runs a PyTorch kernel, initializes llama.cpp, generates a few tokens on the GPU
if the test model is already downloaded, runs a PyTorch kernel again, and then
checks the reverse import order in a fresh interpreter. Both orders must pass;
before 3.0.2 the reverse one aborted the process (see below).

The smoke and coexist tests require `/dev/kfd` and `/dev/dri` on the host.
`make test` additionally downloads the test model (about 1 GB) on its first
run.

## Shell and interactive LFM3 chat

```bash
make bash     # plain shell in the image, repository mounted at /app
make chat     # interactive LFM3 chat
```

`make bash` starts a shell and needs nothing but Docker and an AMD GPU; it does
not download a model. `make chat` downloads the test model if it is missing
(about 1 GB) and starts an interactive chat. Use `/clear` to reset the history,
`/help` to list commands, and `/exit`, `/quit`, or `Ctrl+D` to leave.

Both run the published image unless `LOCAL=1` is passed:

```bash
make LOCAL=1 bash
make LOCAL=1 chat
```

Override the model, generation limit, context size or offload depth on the
command line or through the environment:

```bash
make test MAX_TOKENS=256 N_CTX=4096 GPU_LAYERS=20
# Or inside `make chat`:
python test/lfm3.py --gpu-layers 20 --max-tokens 256
```

`GPU_LAYERS=-1` offloads every layer. Lower it if the model does not fit in
VRAM.

## Reproducible builds

Every external input is pinned:

- The base image is pinned by manifest digest, not by tag.
- Both source repositories are fetched **by commit object**, not by tag: a tag
  can be moved, a commit cannot. Fetching the exact object also avoids cloning
  the full multi-hundred-megabyte history of either project, and the build
  asserts that the checked-out tree matches the requested revision.
- Build and runtime Python dependencies are locked with SHA256 hashes and
  installed with `pip --require-hashes`. `pip check` runs after installation.
- OCI labels record the image version, Git revision, build timestamp, source,
  license, base digest, and both llama.cpp revisions.

The base image is the one input this repository does not build. After a new
`transformers-rocm` release is published, re-resolve the pin and propagate it:

```bash
make pin-base        # asks the registry for the digest behind the tag
make versions        # writes the digest into the Dockerfile and README
make build
```

`make pin-base` refuses to act if the versioned tag and `latest` point at
different manifests, which means the base repository is mid-release.
`make pin-base-check` verifies the recorded pin without changing anything; run
it before publishing.

Run `make lock` **only for an intentional dependency update**, review both lock
files, and then rebuild and run all five test targets.

## Health checks

The base image's healthcheck only proves that PyTorch can see the GPU. This
image replaces it with a check that fails when the llama.cpp HIP backend cannot
initialize on the device, which is the failure mode specific to this image. It
requires `/dev/kfd` and `/dev/dri`, as in every run command above.

Kubernetes ignores the Dockerfile `HEALTHCHECK`; configure an equivalent probe:

```yaml
readinessProbe:
  exec:
    command:
      - python
      - -c
      - "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload()"
  periodSeconds: 30
  timeoutSeconds: 30
  failureThreshold: 3
```

## Publishing a release

After `docker login` and a green `make check` on an AMD GPU host:

```bash
make publish
```

This pushes both `hardandheavy/llama-rocm:3.0.2` and `:latest`.
`make publish tag=latest` is deliberately rejected to avoid an unversioned
release. Both pushes retry independently (`PUSH_RETRIES`, 8 attempts by
default); every attempt reuses the layers the registry already accepted, so
rerunning the target continues where the previous attempt stopped. Verify with:

```bash
docker buildx imagetools inspect hardandheavy/llama-rocm:3.0.2
```

A multi-architecture build is available through `make build tags=a,b`, which
applies several local aliases to one build without recompiling.

## Using this image from another project

The intended downstream is a Dockerfile that derives from this image and runs
`llama_cpp` from the same `/opt/venv` interpreter ComfyUI uses:

```dockerfile
FROM hardandheavy/llama-rocm:3.0.2
RUN python -c "import torch, llama_cpp; assert llama_cpp.llama_supports_gpu_offload()"
```

Because `llama-cpp-python` is installed with `--no-deps`, it never overwrites
the pinned `numpy`, `jinja2` or `typing-extensions` from the base image. Do not
`pip install llama-cpp-python` again downstream — rebuild this image instead,
so the HIP binary and its llama.cpp revision stay verified.

### One ROCm for both stacks

A process may import `torch` and `llama_cpp` in either order. Both work, and
`make coexist-test` checks both: the reverse order in a fresh interpreter, the
torch-first order in one process with a real GPU generation in between.

That was not always true. PyTorch loads ROCm from the pip `rocm-sdk-*` packages
in `/opt/venv`, while llama.cpp is compiled against the deb ROCm under
`/opt/rocm` (see the `CMAKE_HIP_COMPILER` note in the Dockerfile). Until 3.0.2,
`llama-cpp-python`'s own CMakeLists recorded that deb path in the `DT_RPATH` of
`libllama.so` and `libggml.so`, so importing `llama_cpp` first loaded a second
`libamd_comgr.so` — a second LLVM, which registers the same command-line
options as the first and aborted the process:

```
: CommandLine Error: Option 'spirv-expand-step' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```

The builder stage now rewrites that RPATH in the wheel, pointing it at the pip
ROCm before the deb directories, so both stacks resolve one copy of the
libraries. `patchelf` is a builder-stage tool and is not in the shipped image,
and `test/rocm_resolution.py` fails the build if the rewrite ever stops
matching what the base image ships.

This matters for downstream users because `comfyui-rocm` runs both stacks in one
interpreter, and its custom nodes are free to spawn scripts that import either
one first.

## License

The original code and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). Third-party software and models remain subject
to their respective licenses.
