from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle
import gradio as gr
import os

with open("vectordb/chunks.pkl","rb") as f:
    chunks = pickle.load(f)
embedding_model= HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_db = FAISS.load_local("vectordb",embedding_model,
            allow_dangerous_deserialization=True)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
def rerank(query, docs, top_k=3):

    pairs = [(query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(
        zip(scores, docs)
    )

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in scored_docs[:top_k] ]

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

def hybrid_retrieve(query):
    results = vector_db.similarity_search(query,k=3)
    resultsbm = bm25_retriever.invoke(query)
    combined = results + resultsbm
    unique_docs={}
    for doc in combined:
        key = doc.page_content
        if key not in unique_docs:
            unique_docs[key] = doc
    return list(unique_docs.values())

prompt = PromptTemplate(input_variables=["query", "context"], 
    template = """
You are an Enterprise Knowledge Assistant.

Rules:
1. Answer ONLY from the provided context.
2. Do not use outside knowledge.
3. If the answer does not exist in the context, say:
   "Not found in documents."
4. Mention only information supported by the context.
5. Keep the answer under 200 words.

Question:
{query}

Context:
{context}

Answer:
""")
llm = ChatOllama(model="qwen2.5:7B")
parser = StrOutputParser()
rag_chain = prompt|llm|parser

def ask_question(query):
    hybrid_docs = hybrid_retrieve(query)
    top_docs = rerank(query,hybrid_docs)
    context = ""
    sources = set()
    for doc in top_docs:
        context += doc.page_content + "\n\n" 
        sources.add(f"{doc.metadata['source']}|"
                    f"page{doc.metadata['page']+1}")
    final_result = rag_chain.invoke({    "query": query, "context": context}) 
    source_text = "\n".join(sorted(sources))
    return final_result,source_text

demo = gr.Interface(
    fn=ask_question,

    inputs=gr.Textbox(
        label="Ask a Question",
        placeholder="What types of leave are available?"
    ),

    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources")
    ],

    title="Enterprise Knowledge Assistant",

    description=
    "Advanced RAG using "
    "FAISS + BM25 + Reranker + Qwen2.5",

    theme=gr.themes.Soft()
)

demo.launch(
    inbrowser=True
)