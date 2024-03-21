build:
	docker build \
		-t llama-rocm:$(tag) \
		-f Dockerfile .

publish:
	docker image tag llama-rocm:$(tag) hardandheavy/llama-rocm:$(tag)
	docker push hardandheavy/llama-rocm:$(tag)
	docker image tag llama-rocm:$(tag) hardandheavy/llama-rocm:latest
	docker push hardandheavy/llama-rocm:latest

bash-dev:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		llama-rocm:$(tag) bash

bash:
	docker run -it --rm \
		-w /app \
		-v .:/app \
		-v ./huggingface:/root/.cache/huggingface \
		--device=/dev/kfd \
		--device=/dev/dri \
		hardandheavy/llama-rocm:latest bash

test:
	python test/base.py

.PHONY: test
