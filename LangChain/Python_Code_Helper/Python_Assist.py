import os
import subprocess
from dotenv import load_dotenv
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

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


prompt_template = "You are a Code Helper for Python codes Analyse the code : {code} and suggest fixes to solve the errors and modify the code."

llm = Ollama(model="mistral")

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

tools = [analyze_code_tool, suggest_fix_tool]

agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        code = request.form["code"]
        response = agent.run(f"check if there are any errors in the code : {code} and suggest fixes and modify the code to solve the errors.")
        return render_template("index.html", code=code, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
