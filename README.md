Enterprise Knowledge Assistant (Advanced RAG)
Overview
A document question-answering system built using LangChain, FAISS, BM25 Hybrid Retrieval, Cross-Encoder Reranking, and Qwen2.5 running locally through Ollama.
The system answers questions from PDF documents and provides source citations for every answer.
________________________________________
Features
•	Recursive Chunking
•	Semantic Chunking
•	FAISS Vector Search
•	BM25 Keyword Search
•	Hybrid Retrieval
•	Cross Encoder Reranking
•	Local LLM (Qwen2.5 via Ollama)
•	Source Citations
•	Gradio Interface
________________________________________
Architecture
PDF Documents ↓ Chunking ↓ Embeddings ↓ FAISS + BM25 ↓ Hybrid Retrieval ↓ Cross Encoder Reranker ↓ Qwen2.5 ↓ Answer + Citations
________________________________________
Tech Stack
•	Python
•	LangChain
•	Ollama
•	Qwen2.5
•	FAISS
•	Sentence Transformers
•	BM25
•	Gradio
________________________________________
Setup
pip install -r requirements.txt
Run:
python Main.py
________________________________________
Example Query
Question:
What types of leave are available?
Answer:
The policy mentions Annual Leave, Statutory Holidays,
Special Leave, Maternity/Paternity Leave, and
Medical Leave.
________________________________________
Screenshots

________________________________________
Future Improvements
•	Parent Child Retrieval
•	Evaluation Framework
•	AWS Deployment
•	Docker Support

