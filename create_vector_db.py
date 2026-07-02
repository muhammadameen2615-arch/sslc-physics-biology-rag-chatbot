# ============================================
# CREATE VECTOR DATABASE
# SSLC Physics & Biology RAG
# ============================================

import os

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

print("=" * 60)
print("SSLC RAG VECTOR DATABASE CREATOR")
print("=" * 60)

# ============================================
# STEP 1 : CHECK BOOKS FOLDER
# ============================================

if not os.path.exists("books"):
    raise Exception("books folder not found!")

print("Loading PDF files...")

documents = SimpleDirectoryReader(
    "books",
    recursive=True
).load_data()

print(f"Loaded {len(documents)} documents")

# ============================================
# STEP 2 : ADD SUBJECT METADATA
# ============================================

for doc in documents:

    path = doc.metadata["file_path"].lower()

    if "physics" in path:
        doc.metadata["subject"] = "physics"

    elif "biology" in path:
        doc.metadata["subject"] = "biology"

print("Subject labels added.")

# ============================================
# STEP 3 : CHUNKING
# ============================================

Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

print("Chunking configured.")

# ============================================
# STEP 4 : EMBEDDING MODEL
# ============================================

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# ============================================
# STEP 5 : CREATE CHROMADB
# ============================================

db = chromadb.PersistentClient(path="./chroma_data")

collection = db.get_or_create_collection(
    name="education"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

print("Creating vector database...")
print("Please wait...")

# ============================================
# STEP 6 : CREATE INDEX
# ============================================

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("=" * 60)
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Total Chunks : {collection.count()}")

print("\nSaved inside:")
print("./chroma_data")

print("\nNow run:")

print("streamlit run app.py")