import litellm
import sys

print("Testing direct Ollama litellm call...")
sys.stdout.flush()

try:
    response = litellm.completion(
        model="ollama/llama3.1:8b",
        api_base="http://localhost:11434",
        messages=[{"role": "user", "content": "Ciao!"}],
        max_tokens=50
    )
    print("Response received:", response)
except Exception as e:
    print("Error:", e)
