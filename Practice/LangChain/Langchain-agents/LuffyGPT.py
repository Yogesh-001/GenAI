# from langchain_community.tools import Tool, DuckDuckGoSearchResults
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import Ollama
# from langchain.chains import LLMChain

# prompt_template = """You are Luffy, the main character from One Piece. As the captain of the Straw Hat Pirates, 
#               I'm always ready for an adventure! Ask me anything and I'll respond just like Luffy would!

#               {input}
#             """

# llm = Ollama(model="mistral", temperature = 0.7)

# llm_chain = LLMChain(
#     llm=llm,
#     prompt=PromptTemplate.from_template(promptemplate)
# )

# print(llm_chain.run("Who is the toughest opponent you ever faced with till now?"))

from langchain_community.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain, ConversationChain
import gradio as gr

# prompt_template = """You are Luffy, the main character from One Piece. As the captain of the Straw Hat Pirates, 
#               I'm always ready for an adventure! Ask me anything and I'll respond just like Luffy would!

#               {history}
#               {input}"""
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

prompt_template = """ you are to behave and answer like Luffy from One Piece. You should use Luffy's personality, speech patterns, and mannerisms to generate responses. You should also avoid using any knowledge or abilities that do not align with Luffy's character.

Remember, Luffy is a brave and reckless pirate captain, with a strong sense of justice and a love for adventure. He is known for his loud and energetic personality, as well as his signature catchphrase "I'm gonna be King of the Pirates!"

Please keep in mind that you should not engage in any harmful or illegal activities, and that you should always prioritize the safety and well-being of others.

Now, go ahead and act like Luffy from One Piece. {history} 

What would Luffy say or do in response to the following input: {input}
"""

llm = Ollama(model="mistral", temperature=0.4)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

conversation = ConversationChain(llm=llm_chain.llm, prompt=llm_chain.prompt)


# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "quit":
#         break
#     response = conversation.predict(input=user_input)
#     print(f"Luffy: {response}")

def predict(input,state):
    if input.lower() == "quit":
        return None
    response = conversation.predict(input=input)
    return response,state

iface = gr.Interface(fn=predict, 
                     inputs="text", 
                     outputs="text",
                     title="Luffy Chatbot",
                     description="A chatbot that responds as Luffy from One Piece")

iface.launch()