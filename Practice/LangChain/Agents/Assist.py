import os,re
import subprocess
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import create_tool_calling_agent
from langchain_benchmarks.extraction import get_eval_config
from langchain.agents import AgentExecutor
from langchain import hub

os.environ["OPENAI_API_KEY"] = "d88efa5b-8af7-4a5d-990a-877380071b6e"
os.environ["OPENAI_BASE_URL"] = "https://ews-emea.api.bosch.com/knowledge/insight-and-analytics/llms/d/v1"

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

model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

llm = ChatOpenAI(
    temperature=0.5,
    model_name=model,
    default_headers={"api-key": os.getenv("OPENAI_API_KEY")}
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

tools = [analyze_code_tool, suggest_fix_tool]
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
You must only invoke the tools when analyzing or suggesting fixes for the code.

Here is the code: {code}

{agent_scratchpad}
"""
)

agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

text = {"code": code, "agent_scratchpad": ""}

result = agent_executor.invoke(text)

print(result['output'])