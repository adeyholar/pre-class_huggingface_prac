# rag_system.py

import os
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Explicitly import from langchain_community
from langchain_community.embeddings import HuggingFaceEmbeddings # ADDED this line
from langchain_community.vectorstores import Chroma # ADDED this line

import config

class RAGSystem:
    def __init__(self, device):
        self.device = device
        self.vectorstore = None
        self.retriever = None
        self._setup_rag()

    def _setup_rag(self):
        print("\n--- Setting up RAG (Retrieval Augmented Generation) ---")

        documents = []
        if not os.path.exists(config.DATA_DIR):
            print(f"Data directory '{config.DATA_DIR}' not found. Please create it and add PDF files.")
            return # Exit if data directory is missing, RAG won't work

        for file_name in os.listdir(config.DATA_DIR):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(config.DATA_DIR, file_name)
                print(f"Loading document: {file_path}")
                try:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        if not documents:
            print(f"No PDF documents found in '{config.DATA_DIR}'. RAG will not function.")
            print("Please place some PDF files in the 'data' directory.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(texts)} chunks.")

        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={'device': str(self.device)})

        print(f"Creating/Loading Chroma DB at {config.CHROMA_DB_DIR}...")
        if os.path.exists(config.CHROMA_DB_DIR) and len(os.listdir(config.CHROMA_DB_DIR)) > 0:
            self.vectorstore = Chroma(persist_directory=config.CHROMA_DB_DIR, embedding_function=embeddings)
            print("Loaded existing Chroma DB.")
        else:
            self.vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=config.CHROMA_DB_DIR)
            self.vectorstore.persist()
            print("Created and persisted new Chroma DB.")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
        print(f"RAG retriever initialized with top {config.RETRIEVER_K} results.")

    def get_retriever(self):
        return self.retriever