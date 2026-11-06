DOCKER ?= docker
IMAGE ?= llama-rocm
PUBLISH_IMAGE ?= hardandheavy/llama-rocm
tag ?= 3.0.0
LLAMA_CPP_PYTHON_VERSION ?= 0.3.36
MAX_TOKENS ?= 512
N_CTX ?= 8192
LOCAL_IMAGE = $(IMAGE):$(tag)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_REVISION ?= $(shell git rev-parse HEAD)

MODEL_REPOSITORY ?= LiquidAI/LFM3-1.7B-GGUF
MODEL_REVISION ?= cd0c74c575b5c416f3011b5b4c298077f26ca8c2
MODEL_FILE ?= LFM3-1.7B-Q4_K_M.gguf
MODEL_SHA256 ?= 3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4
MODEL_URL = https://huggingface.co/$(MODEL_REPOSITORY)/resolve/$(MODEL_REVISION)/$(MODEL_FILE)

DOCKER_RUN_FLAGS = --rm \
	--ipc=host \
	--device=/dev/kfd \
	--device=/dev/dri \
	-w /app \
	-e MODEL_PATH=/app/huggingface/$(MODEL_FILE) \
	-e MAX_TOKENS=$(MAX_TOKENS) \
	-e N_CTX=$(N_CTX) \
	-v $(CURDIR):/app \
	-v $(CURDIR)/huggingface:/root/.cache/huggingface

build:
	$(DOCKER) build \
		--build-arg IMAGE_VERSION=$(tag) \
		--build-arg IMAGE_REVISION=$(GIT_REVISION) \
		--build-arg IMAGE_CREATED=$(BUILD_DATE) \
		--build-arg LLAMA_CPP_PYTHON_VERSION=$(LLAMA_CPP_PYTHON_VERSION) \
		-t $(LOCAL_IMAGE) \
		-t $(IMAGE):latest \
		-f Dockerfile .

check:
	$(DOCKER) build --check -f Dockerfile .
	python3 -c 'import ast, pathlib; ast.parse(pathlib.Path("test/lfm3.py").read_text())'

# Run curl as root in the image so Docker-owned cache directories also work.
download-model:
	$(DOCKER) run --rm \
		-v $(CURDIR)/huggingface:/model \
		--entrypoint bash $(LOCAL_IMAGE) -c '\
		set -e; \
		if [ ! -f /model/$(MODEL_FILE) ]; then \
			curl -fL --retry 5 --retry-delay 2 -C - \
				-o /model/$(MODEL_FILE).part "$(MODEL_URL)"; \
			echo "$(MODEL_SHA256)  /model/$(MODEL_FILE).part" | sha256sum -c -; \
			mv /model/$(MODEL_FILE).part /model/$(MODEL_FILE); \
		else \
			echo "Model already downloaded: /model/$(MODEL_FILE)"; \
		fi'

# The default interactive command is the LFM3 chat, not a shell.
bash: download-model
	$(DOCKER) run -it $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python test/lfm3.py

bash-dev:
	$(DOCKER) run -it $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) bash

smoke-test:
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python -c 'import llama_cpp; assert llama_cpp.__version__ == "$(LLAMA_CPP_PYTHON_VERSION)"; assert llama_cpp.llama_supports_gpu_offload(); print("llama-cpp-python:", llama_cpp.__version__); print(llama_cpp.llama_print_system_info().decode()); llama_cpp.llama_backend_init(); print("ROCm backend: OK")'

test: download-model
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(LOCAL_IMAGE) python test/lfm3.py --auto-test

publish:
	@if [ "$(tag)" = "latest" ]; then \
		echo "Refusing to publish without an explicit release tag (for example, tag=3.0.0)."; \
		exit 1; \
	fi
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):$(tag)
	$(DOCKER) push $(PUBLISH_IMAGE):$(tag)
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):latest
	$(DOCKER) push $(PUBLISH_IMAGE):latest

.PHONY: test
