import os,re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from functions import getHTMLdata, get_text_chunks, get_vector_store
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

def user_input(user_question):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    result = " ".join([doc.page_content for doc in docs])
    return result


def get_conversational_chain(question, history):

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    access_token = os.getenv("HF_ACCESS_TOKEN")
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token,quantization_config = quantization_config, device_map="auto")
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt_template = """
    You are a helpful assistant. Answer the question as detailed as possible using the provided context.
    If the answer is not available in the context, respond with:
    'The answer is not available in the context.' Avoid guessing.

    Conversation history:
    {history}

    Context:
    {context}

    Question:
    {input}
    """
    
    prompt = PromptTemplate(
        input_variables=["history", "context", "input"],
        template=prompt_template,
    )
    context = user_input(question)

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )
    response = llm_chain.invoke({"input" : question, "context" : context, "history": history})

    match = re.search(r'Answer:\s*(.*)', response["text"], re.DOTALL)
    if match:
        answer = match.group(1).strip()
        print("Answer : \n")
        return answer
    else:
        return "Answer not found in the response."
    # return response

def conversation():
    print("Chat with URL Content \n")
    url_link = "https://blog.spheron.network/how-to-build-an-llm-from-scratch-a-step-by-step-guide"
    raw_text = getHTMLdata(url_link)
    if raw_text:
        text_chunks = get_text_chunks(raw_text.get_text(separator="\n", strip=True))
        get_vector_store(text_chunks)
    chat_history = []
    while True:
        user_question = input("Ask a question based on the URL content (or type 'exit' to quit): \n")        
        if user_question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        if user_question.strip():
            print("\nGenerating an answer......\n")
            response = get_conversational_chain(user_question,chat_history)
            print(f"Agent: {response}")
            chat_history.append({"input": user_question, "output": response})

if __name__ == "__main__":
    conversation()



