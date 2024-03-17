build:
	docker build \
		-t llama-rocm-docker:$(tag) \
		-f Dockerfile .

publish:
	docker image tag llama-rocm-docker:$(tag) hardandheavy/llama-rocm-docker:$(tag)
	docker push hardandheavy/llama-rocm-docker:$(tag)
	docker image tag llama-rocm-docker:$(tag) hardandheavy/llama-rocm-docker:latest
	docker push hardandheavy/llama-rocm-docker:latest

bash-dev:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		llama-rocm-docker:$(tag) bash

bash:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		hardandheavy/llama-rocm-docker:latest bash

test:
	python test/base.py

.PHONY: test
