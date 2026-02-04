import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
import subprocess

load_dotenv()

ddg_search = DuckDuckGoSearchResults()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

# def parse_html(content) -> str:
#     soup = BeautifulSoup(content, 'html.parser')
#     text_content_with_links = soup.get_text()
#     return text_content_with_links

# def fetch_web_page(url: str) -> str:
#     response = requests.get(url, headers=HEADERS)
#     return parse_html(response.content)

# web_fetch_tool = Tool.from_function(
#     func=fetch_web_page,
#     name="WebFetcher",
#     description="Fetches the content of a web page"
# )

# prompt_template = "Research how to use the requests library in Python and open Visual Studio Code. Then, create a python.py file and store the following code in it:\n\n{code}"
prompt_template = "{{action1}} how to use the requests library in Python and {{action2}} Visual Studio Code, create a Python.py file, and store the code in it."

llm = Ollama(model="codellama")
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# def open_vscode():
#     subprocess.run(["code"])

# def create_file_with_code(file_name, code):
#     with open(file_name, 'w') as file:
#         file.write(code)

# open_vscode_tool = Tool.from_function(
#     func=open_vscode,
#     name="OpenVSCode",
#     description="Opens Visual Studio Code"
# )

# create_file_tool = Tool.from_function(
#     func=create_file_with_code,
#     name="CreateFileWithCode",
#     description="Creates a file with code"
# )

# summarize_tool = Tool.from_function(
#     func=llm_chain.run,
#     name="Summarizer",
#     description="Summarizes a web page"
# )

# tools = [ddg_search, web_fetch_tool, summarize_tool, open_vscode_tool, create_file_tool]

# agent = initialize_agent(
#     tools=tools,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     llm=llm,
#     handle_parsing_errors=True
# )

# prompt = "Research how to use the requests library in Python and open visual studio code and create a python.py file and store the code in it."

# print(agent.run(prompt))
import subprocess

def open_vscode():
    subprocess.run(["code"])

vscode_opener_tool = Tool.from_function(
    func=open_vscode,
    name="VSCodeOpener",
    description="Opens Visual Studio Code"
)

tools = [vscode_opener_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

def create_file_with_code(file_name, code=""):
    try:
        abs_file_path = os.path.abspath(file_name)
        os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        with open(abs_file_path, 'w') as file:
            file.write(code)
        print(f"File '{abs_file_path}' created successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")

def interactive_agent(agent, prompt):
    response = agent.run(prompt)

    while True:
        if response.actions:
            for action in response.actions:
                if action.type == "ask":
                    user_input = input(f"{action.prompt}: ")
                    response = agent.continue_with(user_input)
                elif action.type == "inform":
                    print(action.description)
                elif action.type == "execute":
                    action.execute()
                elif action.type == "complete":
                    return
        else:
            return

# Example usage:
prompt = "open visual studio code and create a python.py file and store factorial of a number code in it."
interactive_agent(agent, prompt)

