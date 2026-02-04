import os,re
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

code = """
nterms = int(input("How many terms? "))

n1 n2 = 0, 1
count = 0

if nterms <= 0:
   print("Please enter a positive integer")

elif nterms == 1:
   
   print(n1

else:
   while count < nterms:
       print(n1)
       nth = n1 + n2
       n1 = n2
       n2 = nth
       count += 1
 """

prompt_template = """You are a Code Helper for Python codes. Analyse the code :{code} and suggest fixes to solve the errors.
                    Provide the corrected code only, without any additional comments or explanations.
                    """

model = "meta-llama/Meta-Llama-3-70B-Instruct"  

llm = ChatOpenAI(
    temperature=0.5,
    model_name=model,
    default_headers={"api-key": os.getenv("OPENAI_API_KEY")}
)  

prompt = prompt_template.replace("{code}", code)
response = llm.invoke(prompt, max_tokens=1024)
print(response.content)