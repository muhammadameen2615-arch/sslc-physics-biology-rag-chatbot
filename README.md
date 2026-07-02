# 📚 SSLC Physics and Biology RAG Chatbot

An AI-powered educational chatbot built using **Streamlit**, **Google Gemini**, **LlamaIndex**, and **ChromaDB**.

The chatbot answers SSLC Physics and Biology questions by retrieving relevant information from textbook PDFs using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📖 Physics & Biology Question Answering
- 🤖 Google Gemini Integration
- 🔍 Retrieval-Augmented Generation (RAG)
- 📚 PDF-based Knowledge Base
- ⚡ Fast Semantic Search
- 💻 Simple Streamlit Interface

---

## 🛠 Technologies Used

- Python
- Streamlit
- Google Gemini API
- LlamaIndex
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers

---

## 📂 Project Structure

```
Project/
│
├── app.py
├── create_vector_db.py
├── requirements.txt
├── books/
├── chroma_data/
├── README.md
└── .gitignore
```

---

## ⚙ Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create the vector database

```bash
python create_vector_db.py
```

Run the application

```bash
streamlit run app.py
```

---

## 📖 How it Works

1. Load PDF textbooks
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks
6. Generate answers using Google Gemini

---

## 👨‍💻 Author

**Muhammad Ameen**
