# main.py
# This file serves as the entry point for PyInstaller.
# It launches the Streamlit web application.

import subprocess
import sys
import os
import time

# Determine if running as a PyInstaller bundled app
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    # The base directory of the bundled executable
    bundle_dir = sys._MEIPASS
    # Adjust sys.path to include the root of the bundled app
    # This is where your config.py, llm_manager.py, etc., will be
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    # Set the current working directory to the bundle_dir
    # This helps relative paths for data/chroma_db resolve correctly
    os.chdir(bundle_dir)
else:
    # Running as a normal Python script
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)


# Path to the web_app.py within the bundle
web_app_path = os.path.join(bundle_dir, "web_app.py")

# Path to the Python executable bundled by PyInstaller
python_executable = sys.executable

# Ensure Streamlit is run using its module
streamlit_command = [
    python_executable,
    "-m", "streamlit", "run",
    web_app_path,
    "--server.port", "8501",
    "--server.address", "127.0.0.1",
    "--browser.gatherUsageStats", "false",
    "--browser.serverAddress", "127.0.0.1" # Explicitly set browser server address
]

print(f"Attempting to start Streamlit with command: {' '.join(streamlit_command)}")

try:
    # Use Popen to run Streamlit in a separate process
    # This allows the main.py process to exit without killing Streamlit
    # if Streamlit successfully detaches.
    # We still keep it in try-except for initial launch errors.
    process = subprocess.Popen(streamlit_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read output for a short period to catch immediate errors
    # This is a basic way to get some feedback without blocking indefinitely
    start_time = time.time()
    output_lines = []
    while time.time() - start_time < 10: # Read for up to 10 seconds
        line = process.stdout.readline()
        if line:
            print(f"Streamlit Output: {line.strip()}")
            output_lines.append(line)
        else:
            # If no more output for a while, Streamlit might be running or done.
            # Check if process has exited
            if process.poll() is not None:
                break
        time.sleep(0.1) # Small delay

    if process.poll() is not None:
        # Streamlit process has exited
        stdout, stderr = process.communicate()
        print(f"Streamlit process exited with code: {process.returncode}")
        if stdout:
            print(f"Streamlit STDOUT:\n{stdout}")
        if stderr:
            print(f"Streamlit STDERR:\n{stderr}")
        if process.returncode != 0:
            print("Streamlit failed to start. Please check the output above for errors.")
            # For a windowed app, you might want to show a message box here
            # For now, we'll just print and exit.
    else:
        print("Streamlit process seems to be running in the background.")
        # If running as --windowed, we want main.py to exit here
        # and let Streamlit run independently.
        # If console=True, main.py will stay open.
        pass

except Exception as e:
    print(f"An unexpected error occurred while trying to launch Streamlit: {e}")

# For --windowed apps, main.py should exit here.
# For console apps, you might want to add input() or time.sleep()
# to keep the console open, but not for --windowed.
print("Main application script finished.")
# The process will exit after this, unless Streamlit is still a child process
# and the OS keeps the parent alive.