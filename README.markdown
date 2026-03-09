# RAG Document Chatbot (LangChain + FAISS + Streamlit)

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload documents and ask questions about their content.

The system combines **vector search (FAISS)** with **large language models** to generate answers grounded in uploaded documents.

The application includes a **Streamlit web interface** and supports conversational memory for multi-turn interactions.

---

# Project Overview

Large language models can generate fluent responses but often lack access to specific private documents.

Retrieval-Augmented Generation (RAG) solves this problem by:

1. Converting documents into embeddings
2. Storing them in a vector database
3. Retrieving relevant chunks during a query
4. Providing them as context to the language model

This project demonstrates a full RAG pipeline for document question answering.

---

# Key Features

- Upload documents and query their content
- Conversational chatbot with memory
- Vector search using FAISS
- Document chunking and embeddings
- Retrieval-based question answering
- Streamlit interface for easy interaction
- Debug tools showing retrieved context

---

# Supported File Types

The system can process multiple document formats:

- PDF
- TXT
- CSV
- DOCX
- Markdown
- HTML
- JSON

Document loaders automatically select the correct parser based on file extension.

---

# Architecture

User Query
│
▼
Streamlit Interface
│
▼
LangChain Pipeline
│
▼
Document Retrieval (FAISS)
│
▼
Relevant Context Chunks
│
▼
LLM Response Generation
│
▼
Answer Returned to User

The retrieval step ensures the model answers using **document evidence rather than hallucinating information**.

---

# Technologies Used

- Python
- LangChain
- FAISS Vector Database
- Streamlit
- OpenAI API / LLMs
- RecursiveCharacterTextSplitter
- Document loaders (PyPDF, docx2txt, CSVLoader)

---

The system pipeline follows these steps:
