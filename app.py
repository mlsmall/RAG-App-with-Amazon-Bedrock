import os
from dotenv import load_dotenv, find_dotenv
import boto3
import streamlit as st # to create the UI for the application

# Will be using Amazon Titan model to generate embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data Ingestion
# RecursiveCharacterTextSplitter splits the text into chunks at specific characters: spaces, newlines, etc.
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS # The database

# LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock clients
_ = load_dotenv(find_dotenv())
ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_KEY = os.environ['SECRET_KEY']
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock, credentials_profile_name = "default")

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    
    return docs

# Vector Embedding and Vector Store
def get_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(docs, embedding=bedrock_embeddings)
    vector_store_faiss.save_local("vector_store")

def get_llama2_llm():
    # create the Llama model
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock,
                  model_kwargs={"max_gen_len": 512})
    return llm

def get_titan_llm():
    # create the Titan model
    llm = Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock,
                  model_kwargs={"maxTokenCount": 512})
    return llm

prompt_template = """
Human: Use the following context to answer the question at the end. Summarize the answer with at least 250 words.
If you don't know the answer, just say that. Do not try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vector_store_faiss, query):
    # create the chain
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    # run the chain
    answer = chain({"query": query})
    
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF documents using Amazon Bedrock")
    
    user_query = st.text_input("Ask a question related to the PDF's")
    
    with st.sidebar:
        st.title("Update Vector Store")
        
        if st.button("Update Vector Store"):
            with st.spinner("Updating the vector store"):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated")
                
    if st.button("Titan Output"):
        with st.spinner("Generating response"):
            llm = get_titan_llm()
            vector_store_faiss = FAISS.load_local("vector_store", bedrock_embeddings, allow_dangerous_deserialization=True) 
            response = get_response_llm(llm, vector_store_faiss, user_query)
            st.success(response)
            
    if st.button("LLama2 Output"):
        with st.spinner("Generating response"):
            llm = get_llama2_llm()
            vector_store_faiss = FAISS.load_local("vector_store", bedrock_embeddings, allow_dangerous_deserialization=True) 
            response = get_response_llm(llm, vector_store_faiss, user_query)
            st.success(response)

if __name__ == "__main__":
    main()