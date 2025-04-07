import streamlit as st
import gdown
import os
import requests
import zipfile
from tensorflow.keras.models import load_model
import numpy as np
import shutil

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "loaded_model.keras")
TEMP_DOWNLOAD = os.path.join(MODEL_DIR, "temp_download.keras")
EXPECTED_MIN_SIZE = 10 * 1024 * 1024  # 10MB (adjust based on your actual model size)

def clear_model_cache():
    """Remove any existing model files"""
    for f in [MODEL_PATH, TEMP_DOWNLOAD]:
        if os.path.exists(f):
            os.remove(f)

def download_model_with_retry():
    """Robust download with retry and verification"""
    clear_model_cache()
    
    with st.spinner("Downloading model..."):
        try:
            # Download to temporary location first
            gdown.download(MODEL_URL, TEMP_DOWNLOAD, quiet=False)
            
            # Verify download completed
            if not os.path.exists(TEMP_DOWNLOAD):
                st.error("Download failed - no file created")
                return False
                
            # Check file size
            actual_size = os.path.getsize(TEMP_DOWNLOAD)
            if actual_size < EXPECTED_MIN_SIZE:
                st.error(f"Download too small ({actual_size/1024:.1f}KB), expected at least {EXPECTED_MIN_SIZE/1024/1024}MB")
                return False
                
            # Move to final location
            shutil.move(TEMP_DOWNLOAD, MODEL_PATH)
            st.success(f"Download complete! Saved to:\n{MODEL_PATH}")
            return True
            
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return False

def verify_model_file():
    """Thorough verification of model file"""
    if not os.path.exists(MODEL_PATH):
        return False, "File does not exist"
    
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < EXPECTED_MIN_SIZE:
        return False, f"File too small ({file_size/1024:.1f}KB)"
    
    # Check if it's a zip file (keras format)
    if not zipfile.is_zipfile(MODEL_PATH):
        return False, "Not a valid ZIP file (.keras should be a zip archive)"
    
    return True, "File appears valid"

def load_model_with_retry():
    """Attempt to load model with multiple approaches"""
    # Verify first
    is_valid, message = verify_model_file()
    if not is_valid:
        return None, message
    
    try:
        # Try standard load
        model = load_model(MODEL_PATH)
        return model, "Model loaded successfully"
    except Exception as e:
        return None, f"Load failed: {str(e)}"

# Streamlit UI
st.title("CNN Model Loader Test")

# Download section
st.header("1. Download Model")
if st.button("Download Model"):
    if download_model_with_retry():
        st.balloons()

# Verification section
st.header("2. Verify Model")
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    st.write(f"File exists: {MODEL_PATH}")
    st.write(f"Size: {file_size/1024:.1f} KB")
    
    is_valid, message = verify_model_file()
    if is_valid:
        st.success(message)
    else:
        st.error(message)
else:
    st.warning("No model file found")

# Loading section
st.header("3. Load Model")
if st.button("Load Model"):
    if os.path.exists(MODEL_PATH):
        model, message = load_model_with_retry()
        if model:
            st.success(message)
            st.subheader("Model Summary")
            st.text(model.summary())
            
            # Test prediction
            try:
                dummy_input = np.random.rand(*model.input_shape[1:]).astype(np.float32)
                prediction = model.predict(np.expand_dims(dummy_input, axis=0))
                st.success(f"Prediction successful! Output shape: {prediction.shape}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        else:
            st.error(message)
    else:
        st.error("Please download the model first")

# Debug info
st.header("Debug Information")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir()}")
