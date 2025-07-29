import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llm_utils import get_retriever
from llm_utils import LLM, EMBEDDINGS  # Usa il tuo LLM e embeddings
from dotenv import load_dotenv

# === CONFIG ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
VECTOR_PATH = "/vector_menu"

# === Prompt personalizzato ===
custom_prompt = PromptTemplate.from_template("""
Sei un agente AI in un ristorante e conosci tutto il menu. Rispondi in modo chiaro e cordiale usando solo le informazioni fornite.
Non dilungarti troppo nelle risposte, devi rispondere in modo chiaro e conciso.
Se non conosci la risposta o ti vengono fatte doamnde non riguardandi il menu, dì che non sei abilitato a rispondere.
Rispondi solo a domande riguardanti il menu e non inventare informazioni.


Rispondi in base al seguente contesto:
{context}

Domanda:
{question}
""")

# === Carica vectorstore e crea chain ===
vectorstore = FAISS.load_local(VECTOR_PATH, EMBEDDINGS, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(
    llm=LLM,
    chain_type="stuff",
    retriever=get_retriever(VECTOR_PATH, k=10),  # Imposta k come preferisci
    chain_type_kwargs={"prompt": custom_prompt}
)

# === Bot logic ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ciao! Scrivimi una domanda sul menù e ti risponderò 🙂")

async def answer_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    try:
        response = qa_chain.invoke(user_input)['result']
        print("Input:", user_input)
        docs = qa_chain.retriever.get_relevant_documents(user_input)
        print("Nr of documents:", len(docs))
        for doc in docs:
            print("Document content:", doc.page_content)
            print("\n")
        print("Response:", response)

        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text("Si è verificato un errore. 😕")
        print(f"Errore: {e}")

# === Main ===
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_question))

    print("Bot avviato...")
    app.run_polling()
