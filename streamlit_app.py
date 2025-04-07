import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import numpy as np
import os
import time

# App configuration
st.set_page_config(page_title="CNN Model Loader Test", layout="wide")
st.title("🔍 CNN Model Loader Test")
st.markdown("Testing model loading from Google Drive")

# Constants
FILE_ID = "13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "loaded_model.keras"
CHUNK_SIZE = 32768  # For download progress tracking

def format_size(size_bytes):
    """Convert file size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def download_with_progress(url, output):
    """Enhanced download function with better progress tracking"""
    try:
        import requests
        
        st.info(f"Starting download from:\n{url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        download_speed = st.empty()
        start_time = time.time()
        
        with open(output, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    elapsed = time.time() - start_time
                    progress = int(downloaded / total_size * 100)
                    speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    
                    progress_bar.progress(progress)
                    status_text.text(f"Downloaded: {format_size(downloaded)} / {format_size(total_size)}")
                    download_speed.text(f"Speed: {speed:.2f} MB/s")
        
        progress_bar.empty()
        status_text.empty()
        download_speed.empty()
        
        if os.path.exists(output):
            actual_size = os.path.getsize(output)
            if actual_size == total_size:
                st.success(f"Download complete! File saved as: {output}")
                return True
            else:
                st.error(f"Download incomplete! Expected {format_size(total_size)}, got {format_size(actual_size)}")
                os.remove(output)
                return False
        else:
            st.error("Download failed - no file was created")
            return False
            
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        if os.path.exists(output):
            os.remove(output)
        return False

def verify_model_integrity(filepath):
    """Check if the file appears to be a valid Keras model"""
    try:
        if not os.path.exists(filepath):
            return False, "File does not exist"
            
        min_keras_size = 1024  # Keras models are typically >1KB
        file_size = os.path.getsize(filepath)
        if file_size < min_keras_size:
            return False, f"File too small ({format_size(file_size)}) to be a valid Keras model"
            
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header == b'PK\x03\x04':  # ZIP signature (for .keras format)
                return True, "File appears valid (ZIP header found)"
            elif b'keras' in header:
                return True, "File appears valid (keras signature found)"
            
        return True, "File exists but format uncertain"
    except Exception as e:
        return False, f"Integrity check failed: {str(e)}"

def load_model_safely(filepath):
    """Attempt to load model with multiple fallbacks"""
    try:
        model = load_model(filepath)
        return model, "Standard load successful"
    except Exception as e:
        st.warning(f"Standard load failed: {str(e)}")
        
        try:
            model = load_model(filepath, compile=False)
            return model, "Load successful (compile=False)"
        except Exception as e:
            st.warning(f"Load with compile=False failed: {str(e)}")
            
            try:
                from tensorflow.keras.layers import InputLayer
                custom_objects = {'InputLayer': InputLayer}
                model = load_model(filepath, custom_objects=custom_objects)
                return model, "Load successful with custom objects"
            except Exception as e:
                return None, f"All load attempts failed: {str(e)}"

# Main test flow
with st.expander("⚙️ Test Configuration", expanded=True):
    st.write(f"**Model URL:** `{MODEL_URL}`")
    st.write(f"**Local path:** `{MODEL_PATH}`")
    
    if st.button("🔄 Reset Test (Delete Local Copy)"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.success("Local model deleted")
        else:
            st.info("No local model to delete")

st.header("1️⃣ Model Download")
if not os.path.exists(MODEL_PATH):
    if st.button("⬇️ Download Model"):
        if download_with_progress(MODEL_URL, MODEL_PATH):
            is_valid, message = verify_model_integrity(MODEL_PATH)
            if is_valid:
                st.success(f"✅ Model verification passed: {message}")
            else:
                st.error(f"❌ Model verification failed: {message}")
else:
    st.success("Model already downloaded")
    file_size = os.path.getsize(MODEL_PATH)
    st.write(f"**File size:** {format_size(file_size)}")
    is_valid, message = verify_model_integrity(MODEL_PATH)
    if is_valid:
        st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")

st.header("2️⃣ Model Loading")
if os.path.exists(MODEL_PATH):
    if st.button("🔧 Attempt Model Load"):
        model, message = load_model_safely(MODEL_PATH)
        if model is not None:
            st.success(message)
            
            st.subheader("Model Information")
            cols = st.columns(2)
            with cols[0]:
                st.write("**Input shape:**", model.input_shape)
            with cols[1]:
                st.write("**Output shape:**", model.output_shape)
            
            st.subheader("Layer Summary")
            with st.expander("Show layer details"):
                try:
                    from io import StringIO
                    import sys
                    
                    buffer = StringIO()
                    sys.stdout = buffer
                    model.summary(print_fn=lambda x: st.text(x) or x)
                    sys.stdout = sys.__stdout__
                except Exception as e:
                    st.warning(f"Couldn't show summary: {str(e)}")
            
            st.header("3️⃣ Prediction Test")
            input_shape = model.input_shape[1:]  # Get input shape (excluding batch)
            dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
            
            with st.spinner(f"Making prediction with input shape {input_shape}..."):
                try:
                    prediction = model.predict(dummy_input)
                    st.success(f"🎉 Prediction successful! Output shape: {prediction.shape}")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Input values sample:**")
                        st.code(dummy_input[0].flatten()[:5])
                    with cols[1]:
                        st.write("**Output values sample:**")
                        st.code(prediction[0].flatten()[:5])
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        else:
            st.error(message)
else:
    st.warning("No model file available - download first")

st.markdown("""
---
### Troubleshooting Guide

1. **Download Fails**
   - Verify the file is shared publicly (anyone with the link)
   - Check your internet connection
   - Try manual download: [Download Link](https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh)

2. **Load Fails**
   - Ensure TensorFlow version matches training environment
   - Try different loading methods (shown in code)
   - Check for custom layers that need registration

3. **Prediction Fails**
   - Verify input shape matches model expectations
   - Check if preprocessing is needed
   - Test with actual input data instead of random values
""")
