from llama_cpp import Llama
import warnings

warnings.filterwarnings("ignore", message="The `local_dir_use_symlinks` argument is deprecated")

SYSTEM_PROMPT = "Ты — русскоязычный автоматический ассистент. Ты помогаешь структурировать ответы."

def safe_decode(text_bytes):
    """Безопасное декодирование с обработкой surrogate symbols"""
    try:
        return text_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return text_bytes.decode('utf-8', errors='replace')

def interact(
    n_ctx=8192,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repeat_penalty=1.1,
    max_tokens=500
):
    print("=" * 60)
    print("  Начинается загрузка модели Saiga Llama 3 8B...")
    print("=" * 60)
    
    model = Llama.from_pretrained(
        repo_id="IlyaGusev/saiga_llama3_8b_gguf",
        filename="model-q8_0.gguf",
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        verbose=True,
        chat_format="llama-3"
    )
    
    print("=" * 60)
    print("  ✓ Модель успешно загружена!")
    print("  Для выхода введите: выход, exit или quit")
    print("=" * 60)
    
    while True:
        try:
            user_message = input("\nВы: ").strip()
            
            if user_message.lower() in ['выход', 'exit', 'quit']:
                print("\nЗавершение работы. До свидания!")
                break
                
            if not user_message:
                print("Пожалуйста, введите сообщение")
                continue
                
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            print("\nАссистент: ", end="", flush=True)
            
            response = model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        try:
                            print(content, end="", flush=True)
                            full_response += content
                        except UnicodeEncodeError:
                            safe_content = content.encode('utf-8', errors='replace').decode('utf-8')
                            print(safe_content, end="", flush=True)
                            full_response += safe_content
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\n[Ошибка: {e}]")
            print("Попробуйте еще раз или переформулируйте запрос")
            continue

if __name__ == "__main__":
    interact()
