from crewai import LLM
import litellm

print("Original:", LLM.supports_function_calling)
LLM.supports_function_calling = lambda self: False
print("Patched:", LLM.supports_function_calling)

llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
print("Instance supports:", llm.supports_function_calling())
