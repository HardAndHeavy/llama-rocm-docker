# Transformers on GPU AMD Radeon in Docker
Basic configuration of the docker container [hardandheavy/llama-rocm](https://hub.docker.com/repository/docker/hardandheavy/llama-rocm/general) for working with [llama models](https://huggingface.co) on GPU from Radeon. Based on [transformers-rocm-docker](https://github.com/HardAndHeavy/transformers-rocm-docker).

### Requirements
* Ubuntu
* make
* Docker
* ROCm (see the installation in [transformers-rocm-docker](https://github.com/HardAndHeavy/transformers-rocm-docker?tab=readme-ov-file#install-rocm))

### Testing
Tested on AMD Radeon RX 7900 XTX.
```
make bash
make test
```
