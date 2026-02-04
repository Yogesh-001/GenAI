import os
import subprocess
from dotenv import load_dotenv
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

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
    def suggest_fixes(self, code):
        try:
            output = subprocess.check_output(["autopep8", "--in-place", "--aggressive", os.path.abspath("temp.py")],
                                             input=code.encode(),
                                             stderr=subprocess.STDOUT,
                                             shell=True)
            return output.decode()
        except subprocess.CalledProcessError as e:
            return e.output.decode()
        return "No more suggestions."
    
class FileCodeAnalyzer:
    def analyze_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                code = file.read()
            return analyze_code_tool.run(code)
        except Exception as e:
            return f"Error: {e}"

class FileCodeModifier:
    def modify_file(self, file_path, agent, tools):
        try:
            # Call the agent to analyze and potentially modify the code
            modified_code = agent.analyze_and_modify_code(file_path, tools)

            # Check if modified_code is available (returned by agent)
            if modified_code:
                with open(file_path, "w") as file:
                    file.write(modified_code)
                return "File modified successfully."
            else:
                return "No modifications required for the code."  # Or a custom message

        except Exception as e:
            return f"Error: {e}"

class GitPusher:
    def push_changes(self, repo_dir, commit_message):
        try:
            subprocess.run(["cd", repo_dir], check=True)

            subprocess.run(["git", "add", "."], check=True)

            subprocess.run(["git", "commit", "-m", commit_message], check=True)

            subprocess.run(["git", "push"], check=True)

        except subprocess.CalledProcessError as e:
            return f"Error: {e}"

        return "Changes pushed to GitHub successfully."

prompt_template = """You are a Code Helper for Python. Your job is to analyze and review the Python code in the file at {file_path} to identify any errors or inefficiencies. Once you have identified 
                    any issues, you will suggest fixes and modifications to improve the code. Your analysis should be detailed and thorough, taking into account the code's functionality, efficiency, and readability. You should also provide explanations for any suggested changes,
                    **commented in Python format**, so that the person reviewing the code can understand why the changes are necessary. The changes should be commented in python.
                    After modifying the code, you will push the changes back to the remote GitHub repository using the `push_to_github_tool` tool.
                    Above all, your goal is to help create clean, efficient, and error-free Python code"""

llm = Ollama(model="mistral",temperature=0.3)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

analyze_code_tool = Tool.from_function(
    func=CodeAnalyzer().analyze_code,
    name="CodeAnalyzer",
    description="Analyze the given code and generate a detailed analysis of any errors or issues"
)

suggest_fix_tool = Tool.from_function(
    func=CodeSuggestor().suggest_fixes,
    name="CodeSuggestor",
    description="Generate suggestions for fixing errors or issues in the given code and modify the code without any errors."
)

analyze_file_tool = Tool.from_function(
    func=FileCodeAnalyzer().analyze_file,
    name="FileCodeAnalyzer",
    description="Analyzes the code in a file for errors or issues."
)

modify_file_tool = Tool.from_function(
    func=FileCodeModifier().modify_file,
    name="FileCodeModifier",
    description="Writes the modified code back to the file."
)

push_to_github_tool = Tool.from_function(
    func=GitPusher().push_changes,
    name="GitPusher",
    description="Pushes the modified code to the remote GitHub repository."
)

tools = [analyze_file_tool, suggest_fix_tool, modify_file_tool, push_to_github_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

file_path = "C:/Users/UYW1KOR/React/Sample.py"
input = {"file_path": file_path}
result = agent.run(input="C:/Users/UYW1KOR/React/Sample.py")
print(result)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         code = request.form["code"]
#         response = agent.run(f"check if there are any errors in the code : {code} and suggest fixes and modify the code to solve the errors.")
#         return render_template("index.html", code=code, response=response)
#     return render_template("index.html")


