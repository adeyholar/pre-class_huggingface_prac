# main.py
# This file serves as the entry point for PyInstaller.
# It simply imports and runs the Streamlit web application.

import subprocess
import sys
import os

# Set the environment variable for Hugging Face cache before Streamlit/Transformers load
# This ensures models are downloaded/loaded to a predictable location relative to the app
# For PyInstaller, this might need to be adjusted to a path within the bundled app
# or a known external location.
# For now, we'll keep it as D:\AI\Models\huggingface as specified in config.py
# If you want it relative to the executable, you'd do:
# os.environ["HF_HOME"] = os.path.join(os.path.dirname(sys.executable), "huggingface_cache")
# But for large models, it's better to keep it outside the executable bundle.
# So, we'll rely on config.py setting the path.

# Ensure the project root is in sys.path for imports to work
# This is crucial when running as a bundled executable
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Run the Streamlit app using subprocess
# This is the standard way to run Streamlit from another Python script
try:
    print("Starting Streamlit application...")
    # Use the Python executable that PyInstaller bundles
    # This assumes Streamlit is available in the bundled environment
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(project_root, "web_app.py"),
        "--server.port", "8501", # Default Streamlit port
        "--server.address", "127.0.0.1", # Only accessible locally by default
        "--browser.gatherUsageStats", "false" # Disable usage stats
    ], check=True)
except Exception as e:
    print(f"Error starting Streamlit: {e}")
    input("Press Enter to exit...") # Keep console open on error