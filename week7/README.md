# 📄 Document Question Answering System (RAG)

A **Retrieval-Augmented Generation (RAG)** system that answers questions based on your own documents (PDFs or text files), instead of relying only on a language model's internal knowledge.

## 🚀 Live Demo

🔗 **Streamlit App:** https://document-answering.streamlit.app/

---

## 📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables users to upload custom documents and ask questions about them.

Instead of relying solely on a language model's internal knowledge, the system retrieves the most relevant information from uploaded documents and generates context-aware, grounded answers.

---

## 🏗️ How It Works

The RAG pipeline consists of the following stages:

### 1. Document Ingestion
- Upload PDF or text documents.
- Extract raw text from documents.

### 2. Text Chunking
- Split documents into overlapping chunks.
- Improves retrieval accuracy and preserves context.

### 3. Embedding Creation
- Convert each chunk into dense vector embeddings using **Sentence Transformers**.

### 4. Vector Database
- Store embeddings inside a **FAISS** vector index for efficient similarity search.

### 5. Query Processing
- Convert the user's question into an embedding.

### 6. Context Retrieval
- Retrieve the most relevant document chunks using vector similarity.

### 7. Answer Generation
- Use **Google FLAN-T5** to generate an answer grounded in the retrieved context.

---

# ✨ Features

- 📄 Upload PDF or Text documents
- 🔍 Semantic Search using FAISS
- 🤖 Answer Generation using FLAN-T5
- 📚 Context-aware responses
- ⚡ Streamlit Web Interface
- 🧠 Retrieval-Augmented Generation (RAG)

---

# 🛠️ Tech Stack

- Python
- Streamlit
- Sentence Transformers
- Hugging Face Transformers
- FAISS
- PyPDF
- NumPy

---

# 📂 Project Structure

```text
rag-document-qa/
│
├── app.py                        # Streamlit Web App
├── main.py                       # CLI entry point
├── requirements.txt
│
├── rag_pipeline/
│   ├── config.py
│   ├── document_loader.py
│   ├── text_chunker.py
│   ├── embedding_engine.py
│   ├── vector_store.py
│   └── answer_generator.py
│
└── sample_documents/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory

```bash
cd rag-document-qa
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

## Option 1: Streamlit (Recommended)

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

## Option 2: Command Line

```bash
python main.py
```

Type your question in the terminal.

Type **exit** to quit.

---

# 🌐 Live Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Demo

👉 https://document-answering.streamlit.app/

---

# 📖 Example

### Upload

```
Research_Paper.pdf
```

### Question

```
What is the main objective of the paper?
```

### Answer

```
The paper proposes a Retrieval-Augmented Generation (RAG) system that combines semantic search with language generation to answer questions based on custom documents.
```

---

# 🚀 Future Improvements

- Hybrid Search (Keyword + Vector Search)
- Cross-Encoder Re-ranking
- Support for Multiple File Formats
- Conversation History
- Multi-document Knowledge Base
- Persistent Vector Database
- Better Embedding Models
- LLM API Integration (Gemini/OpenAI)

---

# 📚 Key Learnings

- Retrieval-Augmented Generation (RAG)
- Document Processing
- Text Chunking
- Sentence Embeddings
- Vector Databases (FAISS)
- Semantic Search
- Hugging Face Transformers
- Streamlit Deployment

---

# 📜 License

This project is developed for educational purposes as part of the **Celebal Excellence Internship Program – Week 7 Assignment**.