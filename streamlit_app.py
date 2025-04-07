import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import numpy as np
import os
import time
import requests
from io import StringIO
import sys
from tensorflow.keras.layers import InputLayer

# App configuration
st.set_page_config(page_title="CNN Model Loader Test", layout="wide")
st.title("🔍 CNN Model Loader Test")
st.markdown("Testing model loading from Google Drive")

# Constants
FILE_ID = "13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = os.path.abspath("loaded_model.keras")  # Absolute path
CHUNK_SIZE = 32768

def format_size(size_bytes):
    """Convert file size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def download_with_progress(url, output):
    """Enhanced download function with absolute paths"""
    output_path = os.path.abspath(output)
    try:
        st.info(f"Starting download from:\n{url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = int(downloaded / total_size * 100)
                    elapsed = time.time() - start_time
                    speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Downloaded: {format_size(downloaded)} / {format_size(total_size)}\n"
                        f"Speed: {speed:.2f} MB/s\n"
                        f"Saving to: {output_path}"
                    )
        
        progress_bar.empty()
        status_text.empty()
        
        if os.path.exists(output_path):
            actual_size = os.path.getsize(output_path)
            if actual_size == total_size:
                st.success(f"✅ Download complete! File saved to:\n{output_path}")
                return True
            else:
                st.error(f"❌ Download incomplete! Expected {format_size(total_size)}, got {format_size(actual_size)}")
                os.remove(output_path)
                return False
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def verify_model_integrity(filepath):
    """Enhanced verification with absolute paths"""
    abs_path = os.path.abspath(filepath)
    try:
        if not os.path.exists(abs_path):
            return False, f"❌ File not found at: {abs_path}"
            
        file_size = os.path.getsize(abs_path)
        if file_size < 1024:
            return False, f"❌ File too small ({format_size(file_size)})"
        
        # Try actual loading for best verification
        try:
            test_model = load_model(abs_path, compile=False)
            return True, f"✅ Valid Keras model at:\n{abs_path}"
        except Exception as e:
            st.warning(f"Initial load test failed: {str(e)}")
        
        # Check file signatures
        with open(abs_path, 'rb') as f:
            header = f.read(100)
            if header.startswith(b'\x89HDF'):
                return True, f"⚠️ HDF5 (.h5) format detected at:\n{abs_path}"
            if header.startswith(b'PK\x03\x04'):
                return True, f"⚠️ .keras zip format detected at:\n{abs_path}"
        
        return True, f"⚠️ File exists but format uncertain at:\n{abs_path}"
    except Exception as e:
        return False, f"❌ Verification error: {str(e)}\nPath: {abs_path}"

def load_model_safely(filepath):
    """Complete loading function with absolute paths"""
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        return None, f"❌ File not found at: {abs_path}"
    
    try:
        # Attempt 1: Standard load
        model = load_model(abs_path)
        return model, "✅ Model loaded successfully (standard)"
    except Exception as e1:
        # Attempt 2: Without compilation
        try:
            model = load_model(abs_path, compile=False)
            return model, "✅ Model loaded successfully (compile=False)"
        except Exception as e2:
            # Attempt 3: With custom objects
            try:
                custom_objects = {'InputLayer': InputLayer}
                model = load_model(abs_path, custom_objects=custom_objects)
                return model, "✅ Model loaded successfully (custom objects)"
            except Exception as e3:
                # Attempt 4: Try .h5 extension
                try:
                    model = load_model(abs_path + '.h5')
                    return model, "✅ Model loaded successfully (.h5 extension)"
                except:
                    return None, f"""❌ All load attempts failed:
                    - Standard: {str(e1)}
                    - compile=False: {str(e2)}
                    - Custom objects: {str(e3)}
                    File location: {abs_path}"""

# Main app flow
with st.expander("⚙️ Configuration", expanded=True):
    st.write(f"**Model URL:** `{MODEL_URL}`")
    st.write(f"**Target path:** `{MODEL_PATH}`")
    st.write(f"**Current working directory:** `{os.getcwd()}`")
    
    if st.button("🔄 Reset Test"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.success(f"Deleted: {MODEL_PATH}")
        else:
            st.info("No file to delete")

st.header("1️⃣ Model Download")
if not os.path.exists(MODEL_PATH):
    if st.button("⬇️ Download Model"):
        if download_with_progress(MODEL_URL, MODEL_PATH):
            is_valid, message = verify_model_integrity(MODEL_PATH)
            if is_valid:
                st.success(message)
            else:
                st.error(message)
else:
    st.success(f"Model already exists at:\n{MODEL_PATH}")
    file_size = os.path.getsize(MODEL_PATH)
    st.write(f"**Size:** {format_size(file_size)}")
    is_valid, message = verify_model_integrity(MODEL_PATH)
    if is_valid:
        st.success(message)
    else:
        st.error(message)

st.header("2️⃣ Model Loading")
if os.path.exists(MODEL_PATH):
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
        with st.expander("Show architecture"):
            buffer = StringIO()
            sys.stdout = buffer
            model.summary(print_fn=lambda x: st.text(x))
            sys.stdout = sys.__stdout__
        
        st.header("3️⃣ Prediction Test")
        try:
            input_shape = model.input_shape[1:]
            dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
            
            with st.spinner(f"Predicting with shape {input_shape}..."):
                prediction = model.predict(dummy_input)
                st.success(f"🎉 Prediction successful! Output shape: {prediction.shape}")
                
                cols = st.columns(2)
                with cols[0]:
                    st.write("**Input sample:**")
                    st.code(dummy_input[0].flatten()[:5])
                with cols[1]:
                    st.write("**Output sample:**")
                    st.code(prediction[0].flatten()[:5])
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    else:
        st.error(message)
else:
    st.error(f"File not found at:\n{MODEL_PATH}")

st.markdown("""
---
### Troubleshooting Guide

1. **File Not Found Errors**
   - Verify the exact path shown above
   - Check file permissions
   - Try the download again

2. **Load Failures**
   - Ensure TensorFlow version matches training environment
   - Try manual loading in Python shell:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model(r'{}', compile=False)
     ```
     
3. **Other Issues**
   - Check file integrity (should be {} bytes)
   - Try different model formats (.h5, .keras)
""".format(MODEL_PATH, os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else "unknown"))
