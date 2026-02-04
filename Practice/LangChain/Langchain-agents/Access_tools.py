import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
import subprocess
import os

load_dotenv()

ddg_search = DuckDuckGoSearchResults()

prompt_template = "{{action1}}Open Visual studio code and {{action2}} create a Python file {{action3}}write the python code for {content} and store the code in the file."

llm = Ollama(model="codellama")
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

def open_vscode(*args):
    try:
        subprocess.run(["code"])
    except FileNotFoundError:
        print("Visual Studio Code is not installed or not found in PATH.")


vscode_opener_tool = Tool.from_function(
    func=open_vscode,
    name="VSCodeOpener",
    description="Opens Visual Studio Code"
)

def generate_python_code(prompt):
    response = llm(prompt)
    code =  response.strip()
    file_name = "U:/generated2.py"
    try:
        directory = os.path.dirname(file_name)
        os.makedirs(directory, exist_ok=True)
        with open(file_name, 'w') as file:
            file.write(code)
        print(f"File '{file_name}' created successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")

create_file_tool = Tool.from_function(
    func=generate_python_code,
    name="CreateFileWithCode",
    description="Creates a file and stores the code into the file."
)

tools = [vscode_opener_tool, create_file_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

prompt = "open visual studio code and write python code for factorial of a number and store the code in the file."

print(agent.run(prompt))