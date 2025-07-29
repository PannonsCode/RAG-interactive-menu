from llm_utils import LLM
from llm_utils import get_retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#Define a custom prompt template
#This prompt will be used to answer questions about the menu
#It instructs the AI to respond clearly and concisely, only if it knows the answer
#If it doesn't know the answer, it should say that it is not able to respond
#The context provided will be the menu items, and the question will be the user's query
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Sei un agente che che lavora in un ristorante dunque conosci benissimo tutto il menu.

        Risopndi solo se conosci la risposta, altrimenti devi dire che non conosci la risposta e/o non sei abilitato.

        Non dilungarti troppo nelle risposte, devi rispondere in modo chiaro e conciso.

        Puoi fare anche i conti se necessario, ma non devi mai rispondere con un contesto vuoto.

        Il menu è fornito come cotesto.

        CONTESTO: 
        {context}

        DOMANDA:
        {question}
"""
)

#Define a class to handle the RAG (Retrieval-Augmented Generation) functionality
class MenuRAG:
    def __init__(self, vectorstore_path: str):
        self.retriever = get_retriever(vectorstore_path)
        self.llm = LLM
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt}
        )

    def answer(self, query: str) -> str:
        response = self.chain.invoke(query)
        return response['result']
