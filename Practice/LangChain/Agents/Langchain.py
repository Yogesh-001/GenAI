import os,re
import subprocess
from dotenv import load_dotenv
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

# code = """
# nterms = int(input("How many terms? "))

# n1 n2 = 0, 1
# count = 0

# if nterms <= 0:
#    print("Please enter a positive integer")

# elif nterms == 1:
   
#    print(n1

# else:
#    while count < nterms:
#        print(n1)
#        nth = n1 + n2
#        n1 = n2
#        n2 = nth
#        count += 1
#  """
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

prompt = hub.pull("hwchase17/openai-functions-agent")

model = "meta-llama/Meta-Llama-3.1-70B-Instruct"  

llm = ChatOpenAI(
    temperature=0.5,
    model_name=model,
    default_headers={"api-key": os.getenv("OPENAI_API_KEY")}
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

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# text = """you are autonomous agent for jenkins build where you will go 
#         through the jenkins error logs and solve the errors in 
#         the code based on the logs and push it back to github 
#         and start the jenkins build again. this process happens 
#         unless jenkins doesn't produce any error logs. Use your tools for connecting to jenkins and github to do the tasks."""


text = f"""You are a Code Helper for all programming languages Analyse the code {code} and suggest fixes to solve the errors.
            use your tools:{tools} for analyzing and suggesting fixes to the code and also specify which tool you are using for solving error. and the output should only be corrected code without any explanation.
            or additional comments."""

result = agent_executor.invoke({"input": text})
print(result['output'])