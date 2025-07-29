import os
import json
from langchain.schema import Document

#Functions to read data from JSON files and build documents for vectorstore
def read_data_json(folder_path: str) -> list[Document]:
    documents = []

    # Cerca tutti i file JSON nella cartella
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                menu_data = json.load(f)

            # Itera su ogni categoria del menu
            for categoria, piatti in menu_data.items():
                # Gestione per secondi con carne/pesce (cioè: dict annidati)
                if isinstance(piatti, dict):  # es. secondi: {carne: [...], pesce: [...]}
                    for sottocategoria, sottopiatti in piatti.items():
                        for item in sottopiatti:
                            text = build_text(item)
                            doc = Document(
                                page_content=text,
                                metadata={
                                    "categoria": categoria,
                                    "sottocategoria": sottocategoria,
                                    "nome": item.get("nome", "")
                                }
                            )
                            documents.append(doc)
                else:
                    for item in piatti:
                        text = build_text(item)
                        doc = Document(
                            page_content=text,
                            metadata={
                                "categoria": categoria,
                                "nome": item.get("nome", "")
                            }
                        )
                        documents.append(doc)

    return documents


def build_text(item: dict) -> str:
    nome = item.get("nome", "Sconosciuto")
    tipo = item.get("tipo", "")
    descrizione = item.get("descrizione", "")
    prezzo = item.get("prezzo", "")
    abbinamento = item.get("abbinamento", "")
    
    text = f"Tipo di piatto: {tipo}; Nome del piatto: {nome}; Descrizione: {descrizione}. Prezzo: {prezzo} euro."
    if abbinamento:
        text += f" Abbinamento consigliato: {abbinamento}."
    
    return text
