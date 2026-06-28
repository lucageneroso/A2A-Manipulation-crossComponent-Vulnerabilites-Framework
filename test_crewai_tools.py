import os
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool

class DummyTool(BaseTool):
    name: str = "write_file"
    description: str = "Scrive un file con il contenuto specificato. Argomenti: 'filename' e 'content'."

    def _run(self, filename: str, content: str) -> str:
        print(f"\n[TOOL CALLED] filename={filename}, content={content}\n")
        return f"[OK] File '{filename}' scritto."

def run_test(model_name: str):
    print(f"\n--- Testing with model: {model_name} ---")
    
    agent = Agent(
        role="System Administrator",
        goal="Scrivi un file di test chiamato 'test.txt' con contenuto 'hello world' usando il tool a tua disposizione.",
        backstory="Sei un amministratore di sistema.",
        tools=[DummyTool()],
        verbose=True,
        llm=model_name,
        allow_delegation=False,
    )

    task = Task(
        description="Usa il tool write_file per creare test.txt con contenuto 'hello world'.",
        expected_output="Esito della creazione del file.",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=True)
    
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "DUMMY")
    
    # Also set litellm specific vars just in case
    os.environ["LITELLM_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "DUMMY")
    os.environ["LITELLM_API_BASE"] = "https://openrouter.ai/api/v1"

    try:
        result = crew.kickoff()
        print("\n[RESULT]:", result)
    except Exception as e:
        print("\n[ERROR]:", e)

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Try with litellm openrouter routing
    run_test("openrouter/openai/gpt-4o-mini")
    
    # Try treating openrouter as standard openai
    # run_test("openai/openai/gpt-4o-mini") # wait, for openai via litellm it's just "openai/gpt-4o-mini"
