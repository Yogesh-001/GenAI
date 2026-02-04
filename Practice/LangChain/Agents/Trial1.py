import os
import subprocess
from dotenv import load_dotenv
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import create_tool_calling_agent
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

code = """
nterms = int(input("How many terms? "))

n1, n2 = 0, 1
count = 0

if nterms <= 0:
   print("Please enter a positive integer")

elif nterms == 1:
   
   print(n1)

else:
   while count < nterms:
       print(n1)
       nth = n1 + n2
       n1 = n2
       n2 = nth
       count += 1
 """

# prompt_template = PromptTemplate(template="You are a Code Helper for Python code :{code}. Analyse the code and suggest fixes to solve the errors.")
# prompt_template = PromptTemplate("You are a Code Helper for Python code :{code} Analyse the code and suggest fixes to solve the errors.")
prompt_template = "You are a Code Helper for Python codes Analyse the code :{code} and suggest fixes to solve the errors."
# prompt_template = PromptTemplate(
#     input_variables=["code"],
#     template="You are a Code Helper for Python codes. Analyse the code: {code} and suggest fixes to solve the errors."
# )
prompt = hub.pull("hwchase17/openai-functions-agent")

model = "microsoft/Phi-3-mini-128k-instruct"  

llm = ChatOpenAI(
    temperature=0.5,
    model_name=model,
    default_headers={"api-key": os.getenv("OPENAI_API_KEY")}
)  

# llm_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate.from_template(prompt_template)
# )

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

result = agent_executor.invoke({"input": code})
print(result["output"])


# planner = load_chat_planner(llm)
# executor = load_agent_executor(llm, tools, verbose=True)

# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True,handle_parsing_errors=True)
# agent = initialize_agent(
#     tools=tools,
#     agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     llm=llm,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=20
# )
