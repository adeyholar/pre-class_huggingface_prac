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
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\web_app.py', '.'), # Include web_app.py - IMPORTANT: ensure this is copied to the root of the bundle
        ('D:\\AI\\Gits\\pre-class_huggingface_prac\\chat_history.db', '.'), # Include SQLite DB
    ],
    hiddenimports=[
        'transformers.models.auto.tokenization_auto',
        'transformers.models.auto.modeling_auto',
        'transformers.pipelines',
        'accelerate',
        'bitsandbytes',
        'bitsandbytes.cuda_setup.main',
        'langchain_community.document_loaders',
        'langchain_huggingface.llms',
        'langchain_huggingface.embeddings',
        'langchain_chroma',
        'langchain.chains',
        'langchain.memory',
        'pypdf',
        'sentence_transformers',
        'sqlite3',
        'uuid',
        'streamlit',
        'streamlit.web.cli', # IMPORTANT: Add this for Streamlit's internal CLI
        'streamlit.web.bootstrap', # IMPORTANT: Add this
        'streamlit.web.server', # IMPORTANT: Add this
        'asyncio', 'uvloop', 'httpcore', 'httpx', 'anyio',
        'pydantic', 'pydantic_core',
        'numpy', 'scipy', 'sklearn',
        'tokenizers', 'regex', 'requests', 'tqdm', 'filelock', 'typing_extensions',
        'networkx', 'jinja2', 'fsspec', 'sympy', 'mpmath', 'pillow', 'MarkupSafe',
        'charset_normalizer', 'idna', 'urllib3', 'certifi',
        'psutil', 'pyyaml',
        'click', # Streamlit/Flask dependency
        'werkzeug', # Streamlit/Flask dependency
        'watchdog', # Streamlit dependency for file watching
        'fastapi', 'uvicorn', 'starlette', # If you ever use FastAPI
        'pyarrow', # Often needed by pandas/data processing
        'pandas', # If you use pandas
        'matplotlib', # If you use matplotlib
        'PIL', # Pillow dependency
        'packaging', # Common dependency
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
# Verify this path: D:\ai\conda\envs\llm-local-chatbot3.11\Lib\site-packages\bitsandbytes\
a.binaries += [
    ('libbitsandbytes_cuda121.dll', 'D:\\ai\\conda\\envs\\llm-local-chatbot3.11\\Lib\\site-packages\\bitsandbytes\\libbitsandbytes_cuda121.dll', 'BINARY'),
    # Add any other missing CUDA DLLs if you encounter errors.
    # Example: ('cublas64_12.dll', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5\\bin\\cublas64_12.dll', 'BINARY'),
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
    console=True, # TEMPORARILY set to True for debugging! Change to False for final no-console app.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    cipher=block_cipher,
)