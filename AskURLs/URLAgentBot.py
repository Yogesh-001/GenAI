import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from functions import getHTMLdata, get_text_chunks, get_vector_store
from langchain.llms import HuggingFacePipeline

@st.cache_resource
def VectorStore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

@st.cache_resource
def get_chat_chain():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    access_token = os.getenv("HF_ACCESS_TOKEN")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=access_token, 
        quantization_config=quantization_config, 
        device_map="auto"
    )

    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    vectorstore = VectorStore()

    prompt_template = """
    You are a helpful assistant. Your role is to answer the user's question based on the provided context.

    - If the answer is in the context, provide it in detail.
    - If the answer is not in the context, respond with:
    'The answer is not available in the context.'
    - Avoid providing incorrect or speculative answers.

    **IMPORTANT** Provide the answer ONLY. Do not include the context, instructions, or any additional information in the response.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    chat_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False
    )

    return chat_chain

//without streamlit
def conversation():
    print("Chat with URL Content \n")
    url_link = input("Enter your URL link")
    raw_text = getHTMLdata(url_link)
    if raw_text:
        text_chunks = get_text_chunks(raw_text.get_text(separator="\n", strip=True))
        get_vector_store(text_chunks)

    conversation_chain = get_chat_chain()

    while True:
        question = input("Ask a question based on the URL content (or type 'exit' to quit): \n")
        if question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break

        if question.strip():
            print("\nGenerating an answer......\n")
            response = conversation_chain.invoke({"query" : question})
            answer = response["result"]
            sources = response["source_documents"]

            print(f"Agent : {answer}")
            print("\nSources:")
            for source in sources:
                print(f"-  {source.metadata.get('source', 'unknown')} :  {source.page_content[:200]}...")
def main():
    st.title("ChatBot for URL Contents")
    st.write("chat with URLs")
    url = st.text_input("Enter a URL to retrieve content:", "")
    if url:
        with st.spinner("Retrieving and processing the URL content..."):
            try:
                raw_text = getHTMLdata(url)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text.get_text(separator="\n", strip=True))
                    get_vector_store(text_chunks) 
                    st.success("Content from the URL has been successfully processed. Ask your questions below!")
                else:
                    st.error("Failed to retrieve content from the URL. Please check the URL and try again.")
            except Exception as e:
                st.error(f"An error occurred while processing the URL: {e}")
    question = st.text_input("Ask a question based on the content from the URL:", "")
    if url and question:
        with st.spinner("Generating an answer..."):
            chat_chain = get_chat_chain()
            response = chat_chain({"query": question})
            answer = response["result"]
            st.write(f"**Answer:** {answer}")
            sources = response["source_documents"]
            if sources:
                st.write("**Sources:**")
                for source in sources:
                    st.write(f"- {source.metadata.get('source', 'unknown')}: {source.page_content[:200]}...")



if __name__ == "__main__":
    main()
