from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import Ollama
from functions import getHTMLdata, get_text_chunks, get_vector_store
import streamlit as st
import os

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "The answer is not available in the context."
    Do not provide incorrect answers.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model_name = "meta-llama/Llama-3.1-70B"
    access_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name,use_auth_token=access_token, device_map="auto")
    hf_pipeline = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
    chat_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)
    chat_pipeline = HuggingFacePipeline(model=model, tokenizer=tokenizer)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(chat_pipeline, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embedding_model,allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print("Reply:", response["output_text"])
    st.write("Reply:", response["output_text"])

def conersation():
    url_link = "https://blog.spheron.network/how-to-build-an-llm-from-scratch-a-step-by-step-guide"
    raw_text = getHTMLdata(url_link)
    if raw_text:
        text_chunks = get_text_chunks(raw_text.get_text(separator="\n", strip=True))
        get_vector_store(text_chunks)
    user_question = input("Ask a question based on the URL content: \n")
    if user_question:
        print("\n Generating an answer......\n")
        user_input(user_question)

if __name__ == "__main__":
    conersation()
