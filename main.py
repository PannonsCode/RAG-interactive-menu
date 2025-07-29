from rag_pipeline import MenuRAG

# Main script to run the RAG pipeline from the command line
if __name__ == "__main__":
    rag = MenuRAG(vectorstore_path="vector_menu")
    
    while True:
        user_query = input("Domanda: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = rag.answer(user_query)
        print(f"Risposta: {response}\n")
