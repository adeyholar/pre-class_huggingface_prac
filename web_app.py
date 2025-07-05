# web_app.py

import streamlit as st
import torch
import os
import uuid # To generate unique session IDs
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import our modular components
from llm_manager import LLMManager
from rag_system import RAGSystem
from history_manager import HistoryManager # NEW: Import HistoryManager
import config

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Local LLM RAG Chatbot",
    page_icon="🤖",
    layout="centered", # Can be "wide" for more space
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
if "history_manager" not in st.session_state: # NEW: HistoryManager instance
    st.session_state.history_manager = HistoryManager()
if "session_id" not in st.session_state: # NEW: Current chat session ID
    st.session_state.session_id = str(uuid.uuid4()) # Generate a new UUID for the session


# --- Initial Load of LLM and RAG System (only once) ---
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

# --- Function to load chat history for a session ---
def load_session_history(session_id_to_load):
    st.session_state.session_id = session_id_to_load
    loaded_messages = st.session_state.history_manager.load_history(session_id_to_load)
    st.session_state.messages = loaded_messages
    # Re-populate LangChain memory from loaded messages
    st.session_state.memory.clear()
    for msg in loaded_messages:
        if msg["role"] == "user":
            st.session_state.memory.chat_memory.add_user_message(msg["content"])
        else:
            st.session_state.memory.chat_memory.add_ai_message(msg["content"])
    st.rerun()

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response Generation ---
if prompt := st.chat_input("Ask a question..."):
    # Save user message to DB and session state
    st.session_state.history_manager.save_message(st.session_state.session_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text = ""
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
                    if st.session_state.tokenizer is None or st.session_state.transformers_pipeline is None:
                        response_text = "Error: LLM not fully initialized. Please refresh the page."
                        st.error(response_text)
                    else:
                        chat_history_for_llm = st.session_state.memory.chat_memory.messages + [{"role": "user", "content": prompt}]
                        formatted_input = st.session_state.tokenizer.apply_chat_template(
                            chat_history_for_llm,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        response_text = st.session_state.transformers_pipeline(formatted_input)[0]['generated_text']
                        st.markdown(response_text)
                        # Manually add to memory for LLM-only mode (already handled by LangChain memory for qa_chain)
                        st.session_state.memory.chat_memory.add_user_message(prompt)
                        st.session_state.memory.add_ai_message(response_text) # Use add_ai_message here

            except Exception as e:
                response_text = f"An error occurred: {e}"
                st.error(response_text)
            # Save bot message to DB and session state
            st.session_state.history_manager.save_message(st.session_state.session_id, "assistant", response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})


# --- Sidebar for Status/Info and Session Management ---
with st.sidebar:
    st.header("Application Status")
    st.write(f"LLM Model: `{config.LLM_MODEL_NAME.split('/')[-1]}`")
    if st.session_state.device is not None:
        st.write(f"Device: `{st.session_state.device}`")
    else:
        st.write("Device: Not initialized")

    st.write(f"RAG Enabled: {'Yes' if st.session_state.qa_chain else 'No'}")
    if st.session_state.rag_system and st.session_state.rag_system.vectorstore:
        if hasattr(st.session_state.rag_system.vectorstore, 'get'):
            st.write(f"Documents Loaded: {len(st.session_state.rag_system.vectorstore.get()['documents'])} chunks")
        else:
            st.write("Documents Loaded: N/A (Vectorstore not fully initialized)")
    else:
        st.write("Documents Loaded: N/A (RAG not enabled)")

    st.markdown("---")
    st.header("Chat Sessions")

    # Get all existing session IDs
    all_session_ids = st.session_state.history_manager.get_all_session_ids()
    # Add current session ID if it's not in the list (e.g., brand new session)
    if st.session_state.session_id not in all_session_ids:
        all_session_ids.insert(0, st.session_state.session_id) # Add to top

    # Display session selection
    selected_session_id = st.selectbox(
        "Select a chat session:",
        options=all_session_ids,
        index=all_session_ids.index(st.session_state.session_id) if st.session_state.session_id in all_session_ids else 0,
        format_func=lambda x: f"Session {x[:8]}..." if x != st.session_state.session_id else f"Current Session {x[:8]}..."
    )

    # Load selected session if different from current
    if selected_session_id != st.session_state.session_id:
        load_session_history(selected_session_id)

    # Button to start a new session
    if st.button("Start New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    # Button to clear current session history
    if st.button("Clear Current Session History"):
        st.session_state.history_manager.clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("Developed by Your Name/Company")
    st.markdown("---")


# --- Footer (Optional) ---
st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 10px; font-size: 12px; color: #888; background-color: #f0f2f6;">
        Powered by Local LLMs & LangChain
    </div>
""", unsafe_allow_html=True)