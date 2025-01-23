from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from functions import getHTMLdata, get_text_chunks, get_vector_store
import streamlit as st
import os

def get_conversational_chain():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    access_token = os.getenv("HF-ACCESS-TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map="auto")
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
    chat_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
  
    prompt_template = """
    You are a helpful assistant. Your role is to answer the user's question based on the provided context.

    - If the answer is in the context, provide it in detail.
    - If the answer is not in the context, respond with:
    'The answer is not available in the context.'
    - Avoid providing incorrect or speculative answers.

    Context:
    {context}

    Question:
    {input}
    """
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt_template,
    )

    llm_chain = LLMChain(
        llm=chat_pipeline,
        prompt=prompt,
        verbose=True,
    )

    return llm_chain

def user_input(user_question):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    result = " ".join([doc.page_content for doc in docs])
    print("Tool Output:", result)
    return result

def conversation():
    print("Chat with URL Content \n")
    url_link = "https://blog.spheron.network/how-to-build-an-llm-from-scratch-a-step-by-step-guide"
    raw_text = getHTMLdata(url_link)
    if raw_text:
        text_chunks = get_text_chunks(raw_text.get_text(separator="\n", strip=True))
        get_vector_store(text_chunks)

    conversation_chain = get_conversational_chain()

    while True:
        user_question = input("Ask a question based on the URL content (or type 'exit' to quit): \n")
        if user_question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
          
        if user_question.strip():
            print("\nSearching for relevant context...\n")
            context = user_input(user_question)
            print("\nGenerating an answer......\n")

            response = conversation_chain.run(input=user_question,context=context)
            print(f"Agent: {response}")

if __name__ == "__main__":
    conversation()
