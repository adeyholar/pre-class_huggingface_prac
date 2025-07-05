# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main.py'],
    pathex=['D:\\AI\\Gits\\pre-class_huggingface_prac'], # Ensure this path is correct for your project root
    binaries=[],
    datas=[
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\data', 'data'), # Include your data directory
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\chroma_db', 'chroma_db'), # Include your chroma_db
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\config.py', '.'), # Include config.py
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\llm_manager.py', '.'), # Include llm_manager.py
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\rag_system.py', '.'), # Include rag_system.py
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\web_app.py', '.'), # Include web_app.py
        # Add a placeholder for the SQLite DB if it exists, otherwise it will be created on first run
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\chat_history.db', '.'),
    ],
    hiddenimports=[
        'transformers.models.auto.tokenization_auto', # Common for AutoTokenizer
        'transformers.models.auto.modeling_auto',     # Common for AutoModelForCausalLM
        'transformers.pipelines',                     # For pipeline function
        'accelerate',                                 # Ensure accelerate is bundled
        'bitsandbytes',                               # Crucial for 4-bit quantization
        'bitsandbytes.cuda_setup.main',               # Specific bitsandbytes import
        'langchain_community.document_loaders',       # For PyPDFLoader, TextLoader
        'langchain_huggingface.llms',                 # For HuggingFacePipeline
        'langchain_huggingface.embeddings',           # For HuggingFaceEmbeddings
        'langchain_chroma',                           # For Chroma
        'langchain.chains',                           # For ConversationalRetrievalChain
        'langchain.memory',                           # For ConversationBufferMemory
        'pypdf',                                      # For PyPDFLoader dependency
        'sentence_transformers',                      # For embedding model
        'sqlite3',                                    # For HistoryManager
        'uuid',                                       # For session IDs
        'streamlit',                                  # Ensure Streamlit itself is bundled
        'asyncio', 'uvloop', 'httpcore', 'httpx', 'anyio', # Common async/networking deps for Streamlit/LangChain
        'pydantic', 'pydantic_core',                  # LangChain dependencies
        'numpy', 'scipy', 'sklearn',                  # Common scientific computing deps
        'tokenizers', 'regex', 'requests', 'tqdm', 'filelock', 'typing_extensions', # Common Hugging Face deps
        'networkx', 'jinja2', 'fsspec', 'sympy', 'mpmath', 'pillow', 'MarkupSafe', # Torch/other deps
        'charset_normalizer', 'idna', 'urllib3', 'certifi', # Requests deps
        'psutil', 'pyyaml',                           # Accelerate/other deps
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Add binaries for bitsandbytes (Crucial for GPU)
# You might need to adjust these paths based on your Conda environment's actual DLL locations
# Look in D:\ai\conda\envs\llm-local-chatbot3.11\Lib\site-packages\bitsandbytes\
# and D:\ai\conda\envs\llm-local-chatbot3.11\Library\bin\
# for .dll files related to CUDA and bitsandbytes.
# This is highly dependent on your specific bitsandbytes installation.
# If you get errors about missing DLLs, you'll need to find them and add them here.
# Example (adjust as needed):
a.binaries += [
    ('libbitsandbytes_cuda121.dll', 'D:\\ai\\conda\\envs\\llm-local-chatbot3.11\\Lib\\site-packages\\bitsandbytes\\libbitsandbytes_cuda121.dll', 'BINARY'),
    # You might also need CUDA-related DLLs if they are not picked up automatically
    # e.g., ('cublas64_12.dll', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin\\cublas64_12.dll', 'BINARY'),
    # You'll need to check your CUDA installation path for these.
]


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LLMChatbot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_info_in_exe=True,
    console=False, # Set to True for debugging console, False for no console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    cipher=block_cipher,
    # If you remove --windowed from the pyinstaller command, set console=True here
    # If you keep --windowed, set console=False here
)