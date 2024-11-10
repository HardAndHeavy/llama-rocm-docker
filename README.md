# LLaMA on GPU AMD Radeon in Docker
[LLaMA](https://github.com/abetlen/llama-cpp-python) in a [Docker container with GPU Radeon support](https://hub.docker.com/repository/docker/hardandheavy/llama-rocm/general) for working with [llama models](https://huggingface.co/models?library=gguf). Tested on AMD Radeon RX 7900 XTX.

### Requirements
* Ubuntu
* make
* Docker
* ROCm (see the installation in [transformers-rocm-docker](https://github.com/HardAndHeavy/transformers-rocm-docker?tab=readme-ov-file#install-rocm))

### Testing
```
make bash
make test
```
