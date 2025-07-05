# config.py

import os
import torch

# --- Directory Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory for Hugging Face cached models and datasets
HF_CACHE_DIR = os.path.join("D:", "AI", "Models", "huggingface")

# Directory where your PDF documents are stored for RAG
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Directory where the Chroma vector database will be stored
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# SQLite Database Path for Chat History (NEW LINE)
CHAT_DB_PATH = os.path.join(PROJECT_ROOT, "chat_history.db")

# --- LLM Configuration ---
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_LOCAL_DIR = os.path.join("D:", "AI", "Models", "mistral-7b-instruct-v0.2")

LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

MAX_NEW_TOKENS = 512
DO_SAMPLE = True
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.7
NO_REPEAT_NGRAM_SIZE = 3

# --- Embedding Model Configuration (for RAG) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- RAG Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3

# --- General Application Settings ---
CHAT_HISTORY_MAX_LENGTH = 1000