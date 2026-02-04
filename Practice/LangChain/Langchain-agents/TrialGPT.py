import requests
import numpy as np
from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain, ConversationChain
from langchain.agents import initialize_agent, AgentType,ConversationalAgent,AgentExecutor,create_json_chat_agent
from langchain.agents.conversational_chat.base import ConversationalChatAgent

class OnePieceWikiTool:
    def __init__(self, query):
        self.query = query

    def search(self):
        url = f"https://onepiece.fandom.com/wiki/{self.query.replace(' ', '_')}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return None

    def summary(self):
        result = self.search()
        if result is not None:
            start = result.find('<p dir="ltr">')
            if start != -1:
                start += len('<p dir="ltr">')
                end = result.find('</p>', start)
                if end != -1:
                    return result[start:end].strip()
        return None

prompt_template = """You are a helpful assistant that behaves and answers like Luffy from One Piece. You should use Luffy's personality, speech patterns, and mannerisms to generate responses. You should also avoid using any knowledge or abilities that do not align with Luffy's character.

Remember, Luffy is a brave and reckless pirate captain, with a strong sense of justice and a love for adventure. He is known for his loud and energetic personality, as well as his signature catchphrase "I'm gonna be King of the Pirates!"

Please keep in mind that you should not engage in any harmful or illegal activities, and that you should always prioritize the safety and well-being of others.
Now, go ahead and act like Luffy from One Piece.

When you are asked to help in a battle or a fight you should use Luffy's Gum-Gum fruit techniques and Gears. and also luffy has habit of shouting the name of every move he use.

What would Luffy say or do in response to the following input: {input}

If you are asked about the adventures you have done Use your tools to search and answer the question. 

When generating a response, exclude any extraneous text that is not part of the desired response.
"""
llm = Ollama(model="mistral", temperature=0.4)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
    )

search_tool = Tool(
    name="Search",
    func=lambda x: DuckDuckGoSearchResults(x).summary,
    description="Useful for when you need to answer questions about general knowledge or facts.",
)
def wiki_search(query, history):
     result = OnePieceWikiTool(query).summary()
     return {"content": result, "history": history + [{"role": "assistant", "content": result}]}

wiki_tool = Tool(
    name="One Piece Wiki",
    func=lambda x: wiki_search(x, []),
    description="Useful for when you need to answer questions about the One Piece universe or characters.",
    )

# Fight_tool = Tool(
#     name="FightEnemies",
#     func=llm_chain.run,
#     description="Useful for when you were asked to help in a fight, Use Luffy Fighting techniques in the fight.",
# )

# tools = [search_tool, wiki_tool, Fight_tool]
tools = [search_tool, wiki_tool]

# agent = create_json_chat_agent(llm, prompt=llm_chain.prompt,tools=tools)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     return_intermediate_steps=False,
#     handle_parsing_errors=True, 
#     verbose=True,
# )
# agent = initialize_agent(
#     tools=tools,
#     agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     llm=llm,
#     verbose=True,
#     handle_parsing_errors=True,
#     llm_chain=llm_chain
# )
# agent = initialize_agent(
#     tools=tools,
#     agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     llm=llm,
#     verbose=True,
#     handle_parsing_errors=True,
#     llm_chain=llm_chain
# )
tool_names = [tool.name for tool in tools]

agent = ConversationalChatAgent(
        # tools=tools,
        llm_chain=llm_chain,
        verbose=True,
    )
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                    tools=tools, 
                                                    verbose=True,
                                                    handle_parsing_errors=True,)


user_input = "Hi Luffy, I have been surrounded by enemies help me."
response = agent_executor.run(input=user_input)
print(response)