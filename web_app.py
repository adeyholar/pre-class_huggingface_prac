# web_app.py

import streamlit as st
import torch
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import our modular components
from llm_manager import LLMManager
from rag_system import RAGSystem
import config

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Local LLM RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6; /* Light gray background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #333; /* Darker headings */
    }
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f0f2f6; /* Match main background */
        padding: 10px 20px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
        z-index: 1000;
    }
    /* Adjust chat messages for better spacing and appearance */
    .stChatMessage {
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .stChatMessage.st-ai {
        background-color: #e0f7fa; /* Light cyan for bot */
        align-self: flex-start;
        border-bottom-left-radius: 0;
    }
    .stChatMessage.st-user {
        background-color: #e3f2fd; /* Light blue for user */
        align-self: flex-end;
        border-bottom-right-radius: 0;
        margin-left: auto; /* Push user messages to the right */
    }
    .stChatMessages {
        padding-bottom: 80px; /* Space for the fixed input bar */
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Local LLM RAG Chatbot")
st.markdown("Ask me anything about the documents in your `data` folder!")

# --- Session State Initialization ---
# Initialize session state variables with None or default values first
# This helps Pylance understand that these attributes will exist
if "llm_manager" not in st.session_state:
    st.session_state.llm_manager = None
if "transformers_pipeline" not in st.session_state:
    st.session_state.transformers_pipeline = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "device" not in st.session_state:
    st.session_state.device = None
if "llm_for_langchain" not in st.session_state:
    st.session_state.llm_for_langchain = None
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# Perform heavy initialization only once
if st.session_state.llm_manager is None:
    with st.spinner("Initializing LLM... This may take a moment."):
        st.session_state.llm_manager = LLMManager()
        st.session_state.transformers_pipeline = st.session_state.llm_manager.get_llm_pipeline()
        st.session_state.tokenizer = st.session_state.llm_manager.get_tokenizer()
        st.session_state.device = st.session_state.llm_manager.get_device()
        st.session_state.llm_for_langchain = HuggingFacePipeline(pipeline=st.session_state.transformers_pipeline)

if st.session_state.rag_system is None:
    with st.spinner("Setting up RAG system (loading documents & creating embeddings)..."):
        st.session_state.rag_system = RAGSystem(device=st.session_state.device)
        st.session_state.retriever = st.session_state.rag_system.get_retriever()
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

        if st.session_state.retriever:
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=st.session_state.llm_for_langchain,
                retriever=st.session_state.retriever,
                memory=st.session_state.memory,
                return_source_documents=True
            )
            st.success("RAG chain initialized. Ready to answer questions about your documents!")
        else:
            st.warning("No RAG documents loaded. Falling back to general chat mode (LLM only).")
            st.session_state.qa_chain = None


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Generation ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.qa_chain:
                    result = st.session_state.qa_chain.invoke({"question": prompt})
                    response_text = result["answer"]
                    source_documents = result.get("source_documents")

                    st.markdown(response_text)
                    if source_documents:
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        for i, doc in enumerate(source_documents):
                            source_info = f"**{i+1}.** {doc.metadata.get('source', 'N/A').split(os.sep)[-1]}"
                            page_info = f"Page: {doc.metadata.get('page', 'N/A')}"
                            st.markdown(f"- {source_info} ({page_info})")
                else:
                    # Ensure tokenizer and transformers_pipeline are not None before use
                    if st.session_state.tokenizer is None or st.session_state.transformers_pipeline is None:
                        response_text = "Error: LLM not fully initialized. Please refresh the page."
                        st.error(response_text)
                    else:
                        chat_history_for_llm = st.session_state.memory.chat_memory.messages + [{"role": "user", "content": prompt}]
                        # Pylance fix: Add explicit check for tokenizer before calling its method
                        formatted_input = st.session_state.tokenizer.apply_chat_template(
                            chat_history_for_llm,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        # Pylance fix: Add explicit check for transformers_pipeline before calling it
                        response_text = st.session_state.transformers_pipeline(formatted_input)[0]['generated_text']
                        st.markdown(response_text)
                        st.session_state.memory.chat_memory.add_user_message(prompt)
                        st.session_state.memory.chat_memory.add_ai_message(response_text)

            except Exception as e:
                response_text = f"An error occurred: {e}"
                st.error(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Sidebar for Status/Info (Optional) ---
with st.sidebar:
    st.header("Application Status")
    st.write(f"LLM Model: `{config.LLM_MODEL_NAME.split('/')[-1]}`")
    # Pylance fix: Check if device is not None before accessing it
    if st.session_state.device is not None:
        st.write(f"Device: `{st.session_state.device}`")
    else:
        st.write("Device: Not initialized")

    st.write(f"RAG Enabled: {'Yes' if st.session_state.qa_chain else 'No'}")
    # Pylance fix: Check if rag_system and vectorstore are not None before accessing
    if st.session_state.rag_system and st.session_state.rag_system.vectorstore:
        # Pylance fix: Add explicit check before calling .get()
        if hasattr(st.session_state.rag_system.vectorstore, 'get'):
            st.write(f"Documents Loaded: {len(st.session_state.rag_system.vectorstore.get()['documents'])} chunks")
        else:
            st.write("Documents Loaded: N/A (Vectorstore not fully initialized)")
    else:
        st.write("Documents Loaded: N/A (RAG not enabled)")

    st.markdown("---")
    st.markdown("Developed by Your Name/Company")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.memory: # Check if memory exists before clearing
            st.session_state.memory.clear()
        st.rerun()

# --- Footer (Optional) ---
st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 10px; font-size: 12px; color: #888; background-color: #f0f2f6;">
        Powered by Local LLMs & LangChain
    </div>
""", unsafe_allow_html=True)