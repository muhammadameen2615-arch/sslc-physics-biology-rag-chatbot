import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="SSLC Physics and Biology Chatbot",
    page_icon="📚",
    layout="wide"
)

# ============================================
# TITLE
# ============================================
st.title("📚 SSLC Physics and Biology Chatbot")
st.markdown("Ask questions about **Physics** and **Biology**!")

# ============================================
# SIDEBAR - SETTINGS
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Subject selection
    subject = st.selectbox(
        "Choose Subject:",
        ["General (Both)", "Physics", "Biology"],
        index=0
    )
    
    # Google API Key input (changed from HF token)
    api_key = st.text_input(
        "Google API Key:",
        type="password",
        help="Get your key from https://makersuite.google.com/app/apikey"
    )
    
    st.markdown("---")
    st.markdown("### 📖 How to use:")
    st.markdown("""
    1. Enter your Google API key
    2. Select a subject
    3. Ask your question!
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Stats:")
    

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ============================================
# LOAD RAG SYSTEM (CACHED)
# ============================================
@st.cache_resource
def load_vectorstore():
    """Load the vector store (runs once)"""
    import os
    
    # Check if chroma_data exists
    if not os.path.exists("./chroma_data"):
        st.error("❌ chroma_data folder not found!")
        st.info("💡 Please download chroma_data.zip from Colab and extract it to this folder.")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_data",
        collection_name="education",
        embedding_function=embeddings
    )
    
    return vectorstore

@st.cache_resource
def load_llm(_api_key):
    """Load the LLM (runs once per token)"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Updated to the model you have access to
        temperature=0,
        google_api_key=_api_key
    )
    
    return llm

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(vectorstore, llm, subject_filter):
    """Build RAG chain with subject filter"""
    
    # Create retriever based on subject
    if subject_filter == "Physics":
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "filter": {"subject": "physics"}
            }
        )
    elif subject_filter == "Biology":
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "filter": {"subject": "biology"}
            }
        )
    else:  # General
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful teacher for high school Physics and Biology.

Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer in simple, clear language:
"""
    )
    
    # Build chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ============================================
# MAIN APP LOGIC
# ============================================

# Check if API key is provided
if not api_key:
    st.warning("⚠️ Please enter your Google API key in the sidebar to continue.")
    st.info("💡 Get a free API key from: https://makersuite.google.com/app/apikey")
    st.stop()

# Load resources
try:
    with st.spinner("Loading vector store..."):
        vectorstore = load_vectorstore()
        st.session_state.vectorstore = vectorstore
    
    with st.spinner("Loading language model..."):
        llm = load_llm(api_key)
    
    # Show stats in sidebar
    with st.sidebar:
        st.info(f"✅ Loaded {vectorstore._collection.count()} document chunks")
    
except Exception as e:
    st.error(f"❌ Error loading resources: {str(e)}")
    st.info("💡 Make sure chroma_data folder exists in your project directory!")
    st.stop()

# ============================================
# CHAT INTERFACE
# ============================================

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build RAG chain
                rag_chain = get_rag_chain(vectorstore, llm, subject)
                
                # Get response
                response = rag_chain.invoke(prompt)
                
                # Display response
                st.markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================
# CLEAR CHAT BUTTON
# ============================================
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()