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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_benchmarks.extraction import get_eval_config
from langchain.agents import AgentExecutor
from langchain import hub
# from langchain_community.callbacks import ContextCallbackHandler
from langchain.callbacks import ContextCallbackHandler
import pypdf
import logging
import sys

os.environ["OPENAI_API_KEY"] = "d88efa5b-8af7-4a5d-990a-877380071b6e"
os.environ["OPENAI_BASE_URL"] = "https://ews-emea.api.bosch.com/knowledge/insight-and-analytics/llms/d/v/embeddings"



load_dotenv()

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

model = "HuggingFaceH4/zephyr-7b-beta"

embeddings_model = OpenAIEmbeddings(
    model=model,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

embeddings = embeddings_model.embed_query("Hello World!")
print(len(embeddings), len(embeddings[0]))