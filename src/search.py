import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, store, llm_model: str = "llama-3.1-8b-instant"):
        # Use shared FAISS store from api.py
        self.store = store

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.store.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
Summarize the following context for the query: '{query}'

Context:
{context}

Summary:
"""

        response = self.llm.invoke([prompt])
        return response.content