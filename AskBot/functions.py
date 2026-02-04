from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import requests
import re

def getHTMLdata(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching in URL: {e}")
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def truncate_context(context, tokenizer, max_input_length=4096):
    # Tokenize and truncate the context to fit within max_input_length
    tokens = tokenizer.encode(context, truncation=True, max_length=max_input_length)
    truncated_context = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_context


def get_vector_store(text_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")