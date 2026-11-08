DOCKER ?= docker
# Comma-separated tags to apply to the locally built image. `make build tags=a,b`
# publishes several aliases without rebuilding.
tags ?= 3.0.2
IMAGE ?= llama-rocm
PUBLISH_IMAGE ?= hardandheavy/llama-rocm
tag ?= 3.0.2

LOCAL_IMAGE = $(IMAGE):$(tag)

# Run targets default to the published release, so a fresh clone can start a
# shell or a chat without compiling llama.cpp first. `make build` still tags the
# local image only; passing LOCAL=1 to a run target switches it to that build,
# which is what release verification needs.
RUN_IMAGE ?= $(PUBLISH_IMAGE):$(tag)
ifeq ($(LOCAL),1)
RUN_IMAGE := $(LOCAL_IMAGE)
endif

BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_REVISION ?= $(shell git rev-parse HEAD)
# One image is built and then aliased, so a second name never costs a rebuild.
_ = $(foreach t,$(tags),$(eval TAG_ARGS += -t $(IMAGE):$(t)))

# Lock generation must be reproducible too, so the generator is pinned and
# verified instead of silently using whatever uv is on PATH.
UV ?= uv
UV_VERSION ?= 0.13.7
UV_LOCK_FLAGS = --python-version 3.14 \
	--python-platform x86_64-unknown-linux-gnu \
	--prerelease disallow \
	--generate-hashes \
	--no-annotate \
	--emit-index-url \
	--custom-compile-command 'make lock'

# A multi-gigabyte layer over a home connection gets reset mid-upload often
# enough that a single push attempt is not a reliable release step. Each rerun
# reuses the layers the registry already accepted, so plain retries converge.
PUSH_RETRIES ?= 8
PUSH_RETRY_DELAY ?= 20

# $(call push_with_retries,<image reference>)
define push_with_retries
	@attempt=1; \
	while [ $$attempt -le $(PUSH_RETRIES) ]; do \
		echo "Push attempt $$attempt/$(PUSH_RETRIES): $(1)"; \
		if $(DOCKER) push $(1); then \
			break; \
		fi; \
		if [ $$attempt -eq $(PUSH_RETRIES) ]; then \
			echo "Giving up after $(PUSH_RETRIES) attempts; already-uploaded layers are kept, rerun \`make publish\` to continue."; \
			exit 1; \
		fi; \
		echo "Interrupted; retrying in $(PUSH_RETRY_DELAY)s (completed layers are reused)."; \
		sleep $(PUSH_RETRY_DELAY); \
		attempt=$$((attempt + 1)); \
	done
endef

MODEL_REPOSITORY ?= LiquidAI/LFM3-1.7B-GGUF
MODEL_REVISION ?= cd0c74c575b5c416f3011b5b4c298077f26ca8c2
MODEL_FILE ?= LFM3-1.7B-Q4_K_M.gguf
MODEL_SHA256 ?= 3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4
MODEL_URL = https://huggingface.co/$(MODEL_REPOSITORY)/resolve/$(MODEL_REVISION)/$(MODEL_FILE)

MAX_TOKENS ?= 512
N_CTX ?= 8192
GPU_LAYERS ?= -1

DOCKER_RUN_FLAGS = --rm \
	--ipc=host \
	--device=/dev/kfd \
	--device=/dev/dri \
	-w /app \
	-e MODEL_PATH=/app/huggingface/$(MODEL_FILE) \
	-e MAX_TOKENS=$(MAX_TOKENS) \
	-e N_CTX=$(N_CTX) \
	-e GPU_LAYERS=$(GPU_LAYERS) \
	-v $(CURDIR):/app \
	-v $(CURDIR)/huggingface:/root/.cache/huggingface

build:
	$(DOCKER) build \
		--build-arg IMAGE_VERSION=$(tag) \
		--build-arg IMAGE_REVISION=$(GIT_REVISION) \
		--build-arg IMAGE_CREATED=$(BUILD_DATE) \
		$(TAG_ARGS) \
		-f Dockerfile .

check:
	$(DOCKER) build --check -f Dockerfile .
	python3 -B tools/versions.py check
	python3 -B -m py_compile test/lfm3.py test/llama_smoke.py test/rocm_smoke.py \
		test/coexist_smoke.py test/rocm_resolution.py \
		tools/versions.py tools/pin_base.py

# Propagate versions.json into the Dockerfile blocks and the README table.
versions:
	python3 -B tools/versions.py sync

# Resolve the digest of the published base image and write it into
# versions.json. Run after publishing a new transformers-rocm release, then
# `make versions` and rebuild: without this the base tag could be re-pointed
# under the build.
pin-base:
	python3 -B tools/pin_base.py

# `make check` stays offline; this target asks the registry whether the
# recorded pin still matches the tag. Run it before a release.
pin-base-check:
	python3 -B tools/pin_base.py --check

# Run only for an intentional dependency update; normal builds use the locks.
lock:
	@test "$$($(UV) --version | cut -d' ' -f2)" = "$(UV_VERSION)" \
		|| { echo "uv $(UV_VERSION) is required, found $$($(UV) --version)"; exit 1; }
	$(UV) pip compile requirements-build.in \
		--output-file requirements-build.lock \
		$(UV_LOCK_FLAGS)
	$(UV) pip compile requirements-runtime.in \
		--output-file requirements-runtime.lock \
		$(UV_LOCK_FLAGS)

bash:
	$(DOCKER) run -it $(DOCKER_RUN_FLAGS) $(RUN_IMAGE) bash

chat: download-model
	$(DOCKER) run -it $(DOCKER_RUN_FLAGS) $(RUN_IMAGE) python test/lfm3.py

# Run curl as root in the image so Docker-owned cache directories also work.
download-model:
	$(DOCKER) run --rm \
		-v $(CURDIR)/huggingface:/model \
		--entrypoint bash $(RUN_IMAGE) -c '\
		set -e; \
		if [ ! -f /model/$(MODEL_FILE) ]; then \
			curl -fL --retry 5 --retry-delay 2 -C - \
				-o /model/$(MODEL_FILE).part "$(MODEL_URL)"; \
			echo "$(MODEL_SHA256)  /model/$(MODEL_FILE).part" | sha256sum -c -; \
			mv /model/$(MODEL_FILE).part /model/$(MODEL_FILE); \
		else \
			echo "Model already downloaded: /model/$(MODEL_FILE)"; \
		fi'

smoke-test:
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(RUN_IMAGE) python test/llama_smoke.py

test: download-model
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(RUN_IMAGE) python test/lfm3.py --test

# `smoke-test` checks this image's addition (the llama.cpp HIP backend).
# `base-smoke-test` re-checks what it inherits from the base image.
base-smoke-test:
	$(DOCKER) run --rm --ipc=host --device=/dev/kfd --device=/dev/dri \
		-w /app -v $(CURDIR):/app $(RUN_IMAGE) python test/rocm_smoke.py

# `smoke-test` and `base-smoke-test` each load one of the two ROCm stacks.
# This one loads both in one process, which is what a consumer such as
# comfyui-rocm does, and is the only check of the pair together.
coexist-test:
	$(DOCKER) run $(DOCKER_RUN_FLAGS) $(RUN_IMAGE) python test/coexist_smoke.py

publish:
	@if [ "$(tag)" = "latest" ]; then \
		echo "Refusing to publish without an explicit release tag (for example, tag=3.0.2)."; \
		exit 1; \
	fi
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):$(tag)
	$(call push_with_retries,$(PUBLISH_IMAGE):$(tag))
	$(DOCKER) image tag $(LOCAL_IMAGE) $(PUBLISH_IMAGE):latest
	$(call push_with_retries,$(PUBLISH_IMAGE):latest)

.PHONY: build check versions pin-base pin-base-check lock bash chat download-model smoke-test test base-smoke-test coexist-test publish
