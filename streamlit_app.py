import streamlit as st
import os
import requests
import zipfile
import shutil
from tensorflow.keras.models import load_model
import numpy as np

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "loaded_model.keras")
TEMP_PATH = os.path.join(MODEL_DIR, "temp_model.keras")

def delete_existing_files():
    """Remove any existing model files"""
    for path in [MODEL_PATH, TEMP_PATH]:
        if os.path.exists(path):
            os.remove(path)

def download_with_verification():
    """Download with integrity checks"""
    delete_existing_files()
    
    try:
        # Download to temporary location
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        with open(TEMP_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify file integrity
        if not os.path.exists(TEMP_PATH):
            return False, "Download failed - no file created"
            
        file_size = os.path.getsize(TEMP_PATH)
        if file_size < 1000000:  # 1MB minimum
            return False, f"File too small ({file_size/1024:.1f}KB)"
            
        # Verify ZIP structure
        if not zipfile.is_zipfile(TEMP_PATH):
            return False, "Downloaded file is not a valid ZIP archive"
            
        # Move to final location
        shutil.move(TEMP_PATH, MODEL_PATH)
        return True, f"Download successful! Size: {file_size/1024/1024:.1f}MB"
        
    except Exception as e:
        return False, f"Download error: {str(e)}"

def load_model_safely():
    """Attempt to load model with validation"""
    try:
        # Verify ZIP structure first
        if not zipfile.is_zipfile(MODEL_PATH):
            return None, "File is not a valid Keras ZIP archive"
            
        # Try loading
        model = load_model(MODEL_PATH)
        return model, "Model loaded successfully"
    except Exception as e:
        return None, f"Load failed: {str(e)}"

# Streamlit UI
st.title("🛠️ CNN Model Loader - Repair Mode")

# Current status
st.header("Current Status")
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    st.write(f"File exists at:\n`{MODEL_PATH}`")
    st.write(f"Size: {file_size/1024/1024:.1f}MB")
    
    # Verify file
    is_zip = zipfile.is_zipfile(MODEL_PATH)
    st.write(f"Valid ZIP file: {'✅' if is_zip else '❌'}")
    
    if not is_zip:
        st.error("Critical: File is not a valid .keras ZIP archive")
else:
    st.warning("No model file found")

# Repair actions
st.header("Repair Steps")

if st.button("🔄 Redownload Model"):
    st.warning("This will delete and re-download the model")
    success, message = download_with_verification()
    if success:
        st.success(message)
        st.balloons()
    else:
        st.error(message)

if st.button("🧪 Test Model Loading"):
    if os.path.exists(MODEL_PATH):
        model, message = load_model_safely()
        if model:
            st.success(message)
            
            # Test prediction
            try:
                input_shape = model.input_shape[1:]
                dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
                prediction = model.predict(dummy_input)
                st.success(f"Prediction successful! Output shape: {prediction.shape}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            st.error(message)
    else:
        st.error("Please download the model first")

# Debug information
st.header("Debug Info")
st.write(f"Working directory: `{os.getcwd()}`")
st.write(f"Files in directory: `{os.listdir()}`")
st.write(f"Python version: `{sys.version}`")
