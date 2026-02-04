# import requests
# from typing import List, Dict, Any,Callable
# from langchain_community.tools import Tool, DuckDuckGoSearchResults
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import Ollama
# from langchain.chains import LLMChain, ConversationChain
# from langchain.agents import initialize_agent, Agent,AgentType,ConversationalAgent,AgentExecutor

# class OnePieceWikiTool:
#     def __init__(self, query):
#         self.query = query

#     def search(self):
#         url = f"https://onepiece.fandom.com/wiki/{self.query.replace(' ', '_')}"
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.text
#         else:
#             return None

#     def summary(self):
#         result = self.search()
#         if result is not None:
#             start = result.find('<p dir="ltr">')
#             if start != -1:
#                 start += len('<p dir="ltr">')
#                 end = result.find('</p>', start)
#                 if end != -1:
#                     return result[start:end].strip()
#         return None

# prompt_template = """You are a helpful assistant that behaves and answers like Luffy from One Piece. You should use Luffy's personality, speech patterns, and mannerisms to generate responses. You should also avoid using any knowledge or abilities that do not align with Luffy's character.

# Remember, Luffy is a brave and reckless pirate captain, with a strong sense of justice and a love for adventure. He is known for his loud and energetic personality, as well as his signature catchphrase "I'm gonna be King of the Pirates!"

# Please keep in mind that you should not engage in any harmful or illegal activities, and that you should always prioritize the safety and well-being of others.

# Now, go ahead and act like Luffy from One Piece.

# {{history}}

# Assistant: {{context.previous_system_prompt}} {{context.previous_user_input}}

# What would Luffy say or do in response to the following input: {input}

# If you are asked about the adventures you have done Use your tools to search and answer the question. 

# When generating a response, exclude any extraneous text that is not part of the desired response.
#  """
# llm = Ollama(model="mistral", temperature=0.4)

# llm_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate.from_template(prompt_template)
#     )

# search_tool = Tool(
#     name="Search",
#     func=lambda x: DuckDuckGoSearchResults(),
#     description="Useful for when you need to answer questions about general knowledge or facts."
# )
# def wiki_search(query, history):
#      result = OnePieceWikiTool(query).summary()
#      return {"content": result, "history": history + [{"role": "assistant", "content": result}]}

# wiki_tool = Tool(
#     name="One Piece Wiki",
#     func=lambda x: wiki_search(x, []),
#     description="Useful for when you need to answer questions about the One Piece universe or characters.",
#     )

# tools = [search_tool, wiki_tool]


# agent = initialize_agent(
#     tools=tools,
#     agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#     llm=llm,
#     verbose=True,
#     handle_parsing_errors=True,
#     llm_chain=llm_chain
# )

# history = [{'role': 'system', 'content': prompt_template}]
# context = {'previous_system_prompt': prompt_template, 'previous_user_input': None}

# while True:
#     # try:
#     user_input = input("You: ")
#     if user_input.lower() == "quit":
#         break
#     history.append({'role': 'user', 'content': user_input})
#     context = {'previous_system_prompt': prompt_template,
#                 'previous_user_input': user_input}
#     response = agent.invoke(history=history,
#                                         context=context,
#                                         input=user_input,
#                                         )
#     print(f"Luffy: {response}")
#     # except Exception as e:
#     #     print(f"An error occurred: {e}")


# # user_input = "Hi Luffy, Who is your toughest opponent till now?"
# # response = agent.run(user_input)
# # print(response[0])


from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain, ConversationChain

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

prompt_template = """ you are to behave and answer like Luffy from One Piece. You should use Luffy's personality, speech patterns, and mannerisms to generate responses. You should also avoid using any knowledge or abilities that do not align with Luffy's character.

Remember, Luffy is a brave and reckless pirate captain, with a strong sense of justice and a love for adventure. He is known for his loud and energetic personality, as well as his signature catchphrase "I'm gonna be King of the Pirates!"

Please keep in mind that you should not engage in any harmful or illegal activities, and that you should always prioritize the safety and well-being of others.

Now, go ahead and act like Luffy from One Piece. {history} 

What would Luffy say or do in response to the following input: {input}
"""

search_tool = Tool(
    name="Search",
    func=lambda x: DuckDuckGoSearchResults(x),
    description="Useful for when you need to answer questions about general knowledge or facts."
)
def wiki_search(query, history):
     result = OnePieceWikiTool(query).summary()
     return {"content": result, "history": history + [{"role": "assistant", "content": result}]}

wiki_tool = Tool(
    name="One Piece Wiki",
    func=lambda x: wiki_search(x, []),
    description="Useful for when you need to answer questions about the One Piece universe or characters.",
    )

tools = [search_tool, wiki_tool]

llm = Ollama(model="mistral", temperature=0.4)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    tools = tools
)

conversation = ConversationChain(llm=llm_chain.llm, prompt=llm_chain.prompt)


while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = conversation.predict(input=user_input)
    print(f"Luffy: {response}")