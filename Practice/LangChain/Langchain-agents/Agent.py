import os
from dotenv import load_dotenv
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

load_dotenv()

ddg_search = DuckDuckGoSearchResults()

prompt_template = "Write a Python function that {task_description}."

llm = Ollama(model="llama2")

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

def write_and_save_code(code: str, filename: str) -> None:
    with open(filename, 'w') as file:
        file.write(code)
    print(f"Code has been written to {filename}")

code_editor_tool = Tool.from_function(
    func=write_and_save_code,
    name="CodeEditor",
    description="Write code in a text editor and save it"
)


Model_tool = Tool.from_function(
    func=llm_chain.run,
    name="Modelling",
    description="Provide Assistance on building AI deep learning models"
)

code_generation_tool = Tool.from_function(
    func=llm_chain.run,
    name="CodeGeneration",
    description="Generate Python code"
)

tools = [ddg_search, code_generation_tool, Model_tool, code_editor_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

prompt = "Build a simple Transformer LLM architecture using pytorch."

response = agent.run(prompt)
print(response)