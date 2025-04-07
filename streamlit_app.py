import streamlit as st
import requests
import os
import time
from tensorflow.keras.models import load_model
import numpy as np
import shutil

# Configuration - ADJUST THESE FOR YOUR MODEL
MODEL_URL = "https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"  # Your Google Drive link
MODEL_FILENAME = "my_model.keras"  # Expected filename
MODEL_PATH = os.path.abspath(MODEL_FILENAME)
TEMP_PATH = os.path.abspath("temp_download.keras")
EXPECTED_SIZE = 310 * 1024 * 1024  # 310MB (adjust to your exact size)
MIN_ACCEPTABLE_SIZE = 300 * 1024 * 1024  # 300MB minimum

def clean_up():
    """Remove temporary files"""
    for path in [MODEL_PATH, TEMP_PATH]:
        if os.path.exists(path):
            os.remove(path)

def download_with_retry():
    """Robust download with size verification"""
    clean_up()
    
    try:
        with st.spinner(f"🚀 Downloading {MODEL_FILENAME} (310MB)..."):
            # Create session for better performance
            session = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Range": "bytes=0-"
            }
            
            # Initial request to check size
            response = session.head(MODEL_URL, headers=headers)
            remote_size = int(response.headers.get('content-length', 0))
            
            if remote_size < MIN_ACCEPTABLE_SIZE:
                st.error(f"❌ Remote file too small ({remote_size/1024/1024:.1f}MB)")
                return False
            
            # Download with progress
            response = session.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            speed_text = st.empty()
            start_time = time.time()
            
            downloaded = 0
            with open(TEMP_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):  # 32KB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = time.time() - start_time
                        
                        # Update UI every 100MB or 2 seconds
                        if downloaded % (100 * 1024 * 1024) == 0 or elapsed > 2:
                            progress = min(100, downloaded / EXPECTED_SIZE * 100)
                            speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                            remaining = (EXPECTED_SIZE - downloaded) / (1024 * 1024 * speed) if speed > 0 else 0
                            
                            progress_bar.progress(int(progress))
                            status_text.text(
                                f"📥 Downloaded: {downloaded/1024/1024:.1f}MB/{EXPECTED_SIZE/1024/1024:.1f}MB\n"
                                f"⏱️ Elapsed: {elapsed:.1f}s"
                            )
                            speed_text.text(
                                f"🚀 Speed: {speed:.2f} MB/s\n"
                                f"⏳ Remaining: {remaining:.1f}s" if remaining > 0 else ""
                            )
            
            # Final verification
            actual_size = os.path.getsize(TEMP_PATH)
            if abs(actual_size - EXPECTED_SIZE) > 5 * 1024 * 1024:  # Allow 5MB variance
                st.error(f"Size mismatch! Expected ~{EXPECTED_SIZE/1024/1024:.1f}MB, got {actual_size/1024/1024:.1f}MB")
                return False
                
            # Move to final location
            shutil.move(TEMP_PATH, MODEL_PATH)
            st.success(f"✅ Download complete! Size: {actual_size/1024/1024:.1f}MB")
            return True
            
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        clean_up()
        return False

def verify_model():
    """Thorough model verification"""
    if not os.path.exists(MODEL_PATH):
        return False, "File not found"
    
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < MIN_ACCEPTABLE_SIZE:
        return False, f"File too small ({file_size/1024/1024:.1f}MB)"
    
    # Quick check for Keras format
    try:
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(4)
            if header != b'PK\x03\x04':  # ZIP signature
                return False, "Not a valid .keras ZIP file"
    except:
        return False, "File read error"
    
    return True, f"✅ Valid .keras file ({file_size/1024/1024:.1f}MB)"

# Streamlit UI
st.set_page_config(layout="wide")
st.title(f"🧠 {MODEL_FILENAME} Loader (310MB)")

# Download section
st.header("1. Download Model")
if st.button("🔄 Download Model (310MB)"):
    if download_with_retry():
        st.balloons()

# Verification section
st.header("2. Verify Model")
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    st.write(f"📁 File location: `{MODEL_PATH}`")
    st.write(f"📏 File size: {file_size/1024/1024:.1f}MB")
    
    is_valid, message = verify_model()
    if is_valid:
        st.success(message)
        
        # Load test
        if st.button("🧪 Test Model Loading"):
            try:
                with st.spinner("Loading model..."):
                    model = load_model(MODEL_PATH)
                    st.success("✅ Model loaded successfully!")
                    
                    # Show model info
                    st.subheader("Model Information")
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Input shape:**", model.input_shape)
                    with cols[1]:
                        st.write("**Output shape:**", model.output_shape)
                    
                    # Test prediction
                    st.subheader("Prediction Test")
                    try:
                        dummy_input = np.random.rand(*model.input_shape[1:]).astype(np.float32)
                        prediction = model.predict(np.expand_dims(dummy_input, axis=0))
                        st.success(f"🎉 Prediction successful! Output shape: {prediction.shape}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        
            except Exception as e:
                st.error(f"❌ Load failed: {str(e)}")
    else:
        st.error(message)
else:
    st.warning("No model file found - please download first")

# Debug info
st.header("🛠️ Debug Information")
st.code(f"""
Working directory: {os.getcwd()}
Files present: {os.listdir()}
Expected size: {EXPECTED_SIZE/1024/1024:.1f}MB
Model path: {MODEL_PATH}
""")
