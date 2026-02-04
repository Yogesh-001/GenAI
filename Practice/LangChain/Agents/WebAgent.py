import requests,os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

os.environ["OPENAI_API_KEY"] = "d88efa5b-8af7-4a5d-990a-877380071b6e"
os.environ["OPENAI_BASE_URL"] = "https://ews-emea.api.bosch.com/knowledge/insight-and-analytics/llms/d/v1"

load_dotenv()

ddg_search = DuckDuckGoSearchResults()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

prompt_template = "Summarize the following content: {content}"
model = "microsoft/Phi-3-mini-128k-instruct"  

llm = ChatOpenAI(
    temperature=0.5,
    model_name=model,
    default_headers={"api-key": os.getenv("OPENAI_API_KEY")}
) 

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)

tools = [ddg_search, web_fetch_tool, summarize_tool]

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

result = agent.invoke("Research how to use the requests library in Python. Use your tools to search and summarize content into a guide of 10 points on how to use the requests library.")
print(result)
