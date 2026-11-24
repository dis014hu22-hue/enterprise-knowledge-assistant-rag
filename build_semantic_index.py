from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import pickle

loader = PyPDFLoader("documents/hr policy.pdf")
docs = loader.load()

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
semantic_splitter = SemanticChunker(embedding_model)
semantic_chunks = semantic_splitter.split_documents(docs)
print(f"Semantic Chunks: {len(semantic_chunks)}")

semantic_db = FAISS.from_documents(semantic_chunks, embedding_model)

semantic_db.save_local("semantic_vectordb")

with open(
    "semantic_vectordb/semantic_chunks.pkl",
    "wb"
) as f:

    pickle.dump(semantic_chunks,  f    )

with open(
    "Semantic_chunk_comparison.txt",
    "w",
    encoding="utf-8"
) as f:

   
    f.write("FIRST SEMANTIC CHUNK\n")
    f.write("-" * 50 + "\n")

    f.write(
        semantic_chunks[5].page_content
    )

print("Comparison Saved")