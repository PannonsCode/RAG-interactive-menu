from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#Set embedding
MODEL_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {
    "normalize_embeddings": True 
    }
}

#Set EMBEDDING model and LLM model
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
model_name = "gpt-4.1-mini"
LLM = ChatOpenAI(model_name=model_name, temperature=0.7, openai_api_key=OPENAI_API_KEY, max_tokens=512)

#Function to load a vectorstore from a given path
def load_vectorstore(path: str):
    return FAISS.load_local(folder_path=path, embeddings=EMBEDDINGS, allow_dangerous_deserialization=True)

#Fuction to get retriever from vectorstore
def get_retriever(path: str, k: int = 5):
    vectorstore = load_vectorstore(path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever