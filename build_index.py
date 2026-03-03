from src.data_loader import load_all_documents
from src.vectorStore import FaissVectorStore

print("Building FAISS index from documents...")
docs = load_all_documents("data")
print(f"Loaded {len(docs)} documents")
if docs:
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.save()
else:
    print("No documents found")
