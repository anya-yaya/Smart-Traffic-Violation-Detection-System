# src/utils/helpers.py

import os
import tempfile
from datetime import datetime

def save_uploaded_file(uploaded_file):
    """Saves an uploaded Streamlit file to a temporary location."""
    file_ext = uploaded_file.name.split('.')[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, f"uploaded_file_{timestamp}.{file_ext}")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def clear_temp_directory(temp_dir_path):
    """Clears files and the directory created by save_uploaded_file."""
    if os.path.exists(temp_dir_path) and os.path.isdir(temp_dir_path):
        for file_name in os.listdir(temp_dir_path):
            file_path = os.path.join(temp_dir_path, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        os.rmdir(temp_dir_path)
        print(f"Cleared temporary directory: {temp_dir_path}")
