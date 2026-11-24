hello
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Loading PDF...")

loader = PyPDFLoader("documents/hr policy.pdf")

docs = loader.load()

print("Chunking...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

print(f"Recurssive character splitter Chunks Created : {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print("Creating FAISS Index...")

vector_db = FAISS.from_documents(chunks,embedding_model)

vector_db.save_local("vectordb")

with open("vectordb/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index Saved Successfully")


with open("chunk_comparison.txt", "w", encoding="utf-8") as f:

    f.write("===== RECURSIVE CHUNKING =====\n\n")

    f.write(chunks[5].page_content)

    