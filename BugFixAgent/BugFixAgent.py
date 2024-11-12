import os,re
import subprocess
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_tool_calling_agent

from langchain.agents import AgentExecutor
from langchain import hub
import requests

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("ENDPOINT")

def get_repo_name(api_url, token):
    headers = {"Authorization": f"token {token}"}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        repo_data = response.json()
        return repo_data.get("full_name", "Unknown Repository")
    else:
        return f"Error: {response.status_code} - {response.json().get('message', 'Unknown error')}"
class GitHubPusher:
    def __init__(self, repo_name, token, commit_message="Updated code after error fixes"):
        self.repo_name = repo_name
        self.token = token
        self.commit_message = commit_message

    def push_code(self, code: str, file_name: str = "modified_code.py"):
        """
        Push the provided code (as a string) to the GitHub repository.

        Args:
        - code (str): The code to be pushed to GitHub.
        - file_name (str): The name of the file to save the code in.

        Returns:
        - str: Success or error message.
        """
        try:
            with open(file_name, "w") as f:
                f.write(code)

            repo_url = f"https://{self.token}@github.com/{self.repo_name}.git"
            if not os.path.exists("./repo"):
                subprocess.run(["git", "clone", repo_url, "./repo"], check=True)

            repo_file_path = os.path.join("./repo", file_name)
            os.replace(file_name, repo_file_path)

            os.chdir("./repo")
            subprocess.run(["git", "add", file_name], check=True)
            subprocess.run(["git", "commit", "-m", self.commit_message], check=True)
            subprocess.run(["git", "push"], check=True)

            os.chdir("..")
            return "Code successfully pushed to GitHub."

        except subprocess.CalledProcessError as e:
            os.chdir("..")
            return f"Error during GitHub push: {e}"

        except Exception as e:
            return f"Unexpected error: {e}"

class CodeAnalyzer:
    def analyze_code(self, code):
        try:
            exec(code)
        except SyntaxError as e:
            error_message = str(e)
            return f"Syntax error: {error_message}"
        except Exception as e:
            error_message = str(e)
            return f"Error: {error_message}"
        return "No errors found!"

class CodeSuggestor:
    def __init__(self):
        self.prompt_template = """You are a Code Helper for Python codes. Analyse the code :{code} and suggest fixes to solve the errors. Provide the corrected code only, without any additional comments or explanations."""

    def suggest_fixes(self, code):
        prompt = self.prompt_template.replace("{code}", code)
        response = llm.invoke(prompt, max_tokens=1024)
        corrected_code = response.content
        return {"output": corrected_code}

model = "gpt-4o-mini"

llm = AzureChatOpenAI(
    azure_deployment=model,
    api_version="2023-06-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=15,
)

analyze_code_tool = Tool.from_function(
    func=CodeAnalyzer().analyze_code,
    name="CodeAnalyzer",
    description="Analyze the given code and generate a detailed analysis of any errors or issues."
)

suggest_fix_tool = Tool.from_function(
    func=CodeSuggestor().suggest_fixes,
    name="CodeSuggestor",
    description="Generate suggestions for fixing errors or issues in the given code and modify the code without any errors."
)
github_token = os.getenv("GITHUB_TOKEN")

repo_name = "Babi-01/AutonomousAgent"
print(repo_name)

github_tool = Tool.from_function(
    func=GitHubPusher(repo_name=repo_name, token=github_token).push_code,
    name="GitHubPusher",
    description="Push the modified code to the specified GitHub repository."
)

tools = [analyze_code_tool, suggest_fix_tool, github_tool]
code = """
#include <iostream>
using namespace std

int main() {

  int first_number, second_number, sum
    
  cout << "Enter two integers: ";
  cin >> first_number >> second_number;
  sum = first_number , second_number;

  cout << first_number << " + " <<  second_number << " = " << sum;     

  return 0;
}

"""
prompt_template = PromptTemplate(
    input_variables=["code", "agent_scratchpad"],
    template="""
You are a code helper for all programming languages. Analyze and fix the provided code strictly using the available tools: {{tools}}. 
Push the Modified Code to GitHub**: After fixing the code, invoke the GitHub push tool to upload the changes to the specified repository.
You must only invoke the tools when analyzing or suggesting fixes for the code or pushing the code to the github.

Here is the code: {code}

{agent_scratchpad}
"""
)
# prompt_template = PromptTemplate(
#     input_variables=["code", "agent_scratchpad"],
#     template="""
# You are a highly skilled code assistant specializing in all programming languages. Your task is to analyze, fix, and finalize the provided code. 

# Follow these steps strictly:
# 1. **Analyze** the code using the available tools to identify errors or issues.
# 2. **Suggest Fixes** and modify the code to resolve all identified errors, ensuring the final code is error-free.
# 3. **Push the Modified Code to GitHub**: After fixing the code, invoke the GitHub push tool to upload the changes to the specified repository.

# Rules:
# - Use only the available tools: {{tools}}.
# - Ensure every step is executed in sequence.
# - If any step fails, do not proceed to the next until the issue is resolved.
# - Provide clear and concise updates for each step in your response.

# Here is the code: {code}

# {agent_scratchpad}
# """
# )


agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

text = {"code": code, "agent_scratchpad": ""}

result = agent_executor.invoke(text)

print(result['output'])