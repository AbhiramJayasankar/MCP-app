import requests
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama  # or ChatOllama if you want roles
from langchain.tools import tool

# 1. Define the weather tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city using Open-Meteo API."""
    try:
        # Replace with your preferred weather API if needed
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# 2. Initialize the LLM
llm = Ollama(model="gemma3:4b")

# 3. Create an agent with the weather tool
agent = initialize_agent(
    tools=[get_weather],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# 4. Ask the agent a question
response = agent.run("What's the weather like in Paris?")
print(response)
