from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # NEW package!
from langchain_core.runnables import RunnableSequence

# 1. Create the LLM using the new langchain_ollama
llm = OllamaLLM(model="gemma3:4b")

# 2. Define the prompt using PromptTemplate
prompt = PromptTemplate.from_template("Hello {name}, who r u ")

# 3. Create a RunnableSequence (modern alternative to LLMChain)
chain = prompt | llm

# 4. Run the chain using .invoke()
output = chain.invoke({"name": "hi"})
print(output)
