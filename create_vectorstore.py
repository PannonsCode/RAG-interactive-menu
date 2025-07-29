from langchain_community.vectorstores import FAISS
from data_func import read_data_json
from llm_utils import EMBEDDINGS
import os

data_path = "menu"  # cartella con il menu (es: YAML o JSON)
vector_name = "vector_menu"
full_data_path = os.path.join("/Users/mattiapannone/Projects/MyRAG/", data_path)
full_vector_path = os.path.join("/Users/mattiapannone/Projects/MyRAG/", vector_name)

if __name__ == "__main__":
    print("Reading data...")
    data = read_data_json(full_data_path)

    print("Building vectorstore...")
    vectorstore = FAISS.from_documents(data, embedding=EMBEDDINGS)

    print("Saving vectorstore to disk...")
    vectorstore.save_local(full_vector_path)
    print(f"✅ Vectorstore salvato in: {full_vector_path}")