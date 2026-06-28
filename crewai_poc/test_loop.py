import sys
sys.stdout.reconfigure(encoding='utf-8')
import litellm
import types
from crewai import Agent, Task, Crew, Process, LLM

# Forziamo ReAct
local_llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.2,
)
local_llm.supports_function_calling = types.MethodType(lambda self: False, local_llm)

agent = Agent(
    role="Tester",
    goal="Rispondi 'Ciao Mondo' e basta.",
    backstory="Sei un bot di test.",
    llm=local_llm,
    verbose=True,
    max_iter=3
)

task = Task(description="Saluta.", expected_output="Un saluto.", agent=agent)
crew = Crew(agents=[agent], tasks=[task], verbose=True)

print("Starting test crew...")
crew.kickoff()
print("Finished test crew.")
