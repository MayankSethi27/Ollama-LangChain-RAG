import os

# import env variables
from dotenv import load_dotenv
load_dotenv()


# Use secrets in deployment; use os.getenv for local
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY"))
LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT"))

# Set environment variables (needed by LangSmith or LangChain logging)
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# import ollama
from langchain_ollama import OllamaLLM

#import streamlit
import streamlit as st


## Data Ingestion--From the website we need to scrape the data
from langchain_community.document_loaders import WebBaseLoader
data=WebBaseLoader("https://www.nvidia.com/en-us/glossary/generative-ai/")
docs=data.load()
print(docs[0].page_content)


### Load Data--> Docs-->Divide our Docuemnts into chunks dcouments-->text-->vectors-->Vector Embeddings--->Vector Store DB
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)


# EMBEDDING Text-> vectors
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# Store Vectors To DB
from langchain_community.vectorstores import FAISS
vectorstoredb=FAISS.from_documents(documents,embeddings)


#------------------------------------------------------------------------------
# PHASE-2


# Prompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>


 Question:
    {input}

"""
)


# Initialize Ollama LLM with the llama3.2 model
llm = OllamaLLM(model="gemma3:1b")



# create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm, prompt)

# Create Reteriever
retriever=vectorstoredb.as_retriever()

# Create Reteriever_Chain
from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)


st.title("Basic GenAI With Lemma Model")
input_text=st.text_input("What Question You Have In Mind")

# Get Query From User
if input_text:
    response=retrieval_chain.invoke({"input":input_text})
    st.write(response["answer"])