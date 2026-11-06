"""Chat with LFM3 GGUF or run a non-interactive llama.cpp/ROCm test."""

import argparse
import os
from pathlib import Path
from typing import Any

import llama_cpp
from llama_cpp import Llama


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "huggingface" / "LFM3-1.7B-Q4_K_M.gguf"
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
DEFAULT_N_CTX = int(os.getenv("N_CTX", "8192"))
SYSTEM_PROMPT = "Ты — полезный русскоязычный ассистент. Отвечай кратко."
AUTO_TEST_PROMPT = "Назови столицу Франции. Ответь одним словом."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an interactive LFM3 chat by default, or generate and validate "
            "one response with --auto-test."
        )
    )
    parser.add_argument(
        "--auto-test",
        action="store_true",
        help="run one fixed prompt, validate model generation, and exit",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="local GGUF path (default: MODEL_PATH or %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="generation limit per response (default: MAX_TOKENS or %(default)s)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=DEFAULT_N_CTX,
        help="context size (default: N_CTX or %(default)s)",
    )
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be greater than zero")
    if args.n_ctx < 1:
        parser.error("--n-ctx must be greater than zero")
    return args


def initial_messages() -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def check_rocm() -> None:
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError(
            "ROCm GPU offload is unavailable. Run `make test` on a host with "
            "/dev/kfd and /dev/dri and a compatible AMDGPU driver."
        )
    print(f"llama-cpp-python: {llama_cpp.__version__}")
    print(llama_cpp.llama_print_system_info().decode().strip())


def load_model(model_path: Path, n_ctx: int) -> Llama:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"GGUF model not found: {model_path}. Run `make download-model` first."
        )

    print(f"Model: {model_path}")
    print(f"Context: {n_ctx} tokens; GPU layers: all")
    model = Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        seed=0,
        verbose=False,
    )
    architecture = model.metadata.get("general.architecture")
    if architecture != "lfm3":
        raise RuntimeError(f"Expected an LFM3 GGUF, got architecture {architecture!r}")
    return model


def generate_response(
    model: Llama, messages: list[dict[str, str]], max_tokens: int
) -> tuple[str, int]:
    """Generate one response without modifying the conversation history."""
    completion: Any = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        top_k=50,
        repeat_penalty=1.1,
        seed=0,
        stream=False,
    )
    choice = completion["choices"][0]
    content = choice["message"].get("content") or ""
    generated_tokens = completion.get("usage", {}).get("completion_tokens", 0)

    if choice.get("finish_reason") == "length":
        raise RuntimeError(
            "The generation limit was reached before the response finished; "
            "increase --max-tokens."
        )

    # The embedded LFM3 template enables reasoning. Keep only the final answer
    # in chat history so old reasoning does not consume subsequent contexts.
    response = content.rsplit("</think>", 1)[-1].strip()
    if generated_tokens == 0 or not response:
        raise RuntimeError("The model returned an empty response")
    return response, generated_tokens


def run_auto_test(model: Llama, max_tokens: int) -> None:
    messages = initial_messages()
    messages.append({"role": "user", "content": AUTO_TEST_PROMPT})
    response, generated_tokens = generate_response(model, messages, max_tokens)

    print(f"\nPrompt: {AUTO_TEST_PROMPT}")
    print(f"Generated response: {response}")
    print(f"Generated tokens: {generated_tokens}")
    print("ROCm llama.cpp auto-test passed.")


def print_chat_help() -> None:
    print("Команды: /clear — очистить историю; /exit или /quit — выйти.")


def run_chat(model: Llama, max_tokens: int) -> None:
    messages = initial_messages()
    print("\nДиалог с LFM3. Модель загружена, история сохраняется между репликами.")
    print_chat_help()

    while True:
        try:
            prompt = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДиалог завершён.")
            return

        if not prompt:
            continue

        command = prompt.lower()
        if command in {"/exit", "/quit", "/q", "exit", "quit", "выход"}:
            print("Диалог завершён.")
            return
        if command == "/clear":
            messages = initial_messages()
            print("История очищена.")
            continue
        if command == "/help":
            print_chat_help()
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            response, _ = generate_response(model, messages, max_tokens)
        except KeyboardInterrupt:
            messages.pop()
            print("\nГенерация прервана. Можно продолжить диалог или ввести /exit.")
            continue
        except Exception:
            # Do not leave an unanswered user message in the next request.
            messages.pop()
            raise

        messages.append({"role": "assistant", "content": response})
        print(f"\nLFM3: {response}")


def main() -> None:
    args = parse_args()
    check_rocm()
    model = load_model(args.model_path, args.n_ctx)
    if args.auto_test:
        run_auto_test(model, args.max_tokens)
    else:
        run_chat(model, args.max_tokens)


if __name__ == "__main__":
    main()
