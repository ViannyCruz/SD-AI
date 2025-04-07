import streamlit as st
import os
import requests
import shutil
from tensorflow.keras.models import load_model
import numpy as np
import gdown

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_PATH = os.path.abspath("my_model.keras")
EXPECTED_SIZE = 310 * 1024 * 1024  # 310MB
MAX_RETRIES = 3

def reset_environment():
    """Clear any existing files"""
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

def download_with_gdown():
    """Alternative download using gdown"""
    try:
        reset_environment()
        with st.spinner(f"⬇️ Downloading model (310MB) using gdown..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            actual_size = os.path.getsize(MODEL_PATH)
            if actual_size >= EXPECTED_SIZE * 0.95:  # Allow 5% variance
                st.success(f"✅ Download complete! Size: {actual_size/1024/1024:.1f}MB")
                return True
        st.error("❌ Download incomplete")
        return False
    except Exception as e:
        st.error(f"❌ gdown failed: {str(e)}")
        return False

def download_with_requests():
    """Direct download with requests"""
    reset_environment()
    try:
        with st.spinner(f"⬇️ Downloading model (310MB) using requests..."):
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept-Encoding": "identity"
            }
            response = requests.get(MODEL_URL, headers=headers, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', EXPECTED_SIZE))
            if total_size < EXPECTED_SIZE * 0.5:
                st.error(f"❌ Reported size too small: {total_size/1024/1024:.1f}MB")
                return False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            downloaded = 0
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = min(100, downloaded / total_size * 100)
                        progress_bar.progress(int(progress))
                        status_text.text(
                            f"Downloaded: {downloaded/1024/1024:.1f}MB/"
                            f"{total_size/1024/1024:.1f}MB"
                        )
            
            actual_size = os.path.getsize(MODEL_PATH)
            if actual_size >= EXPECTED_SIZE * 0.95:
                st.success(f"✅ Download complete! Size: {actual_size/1024/1024:.1f}MB")
                return True
            else:
                st.error(f"❌ Incomplete download ({actual_size/1024/1024:.1f}MB)")
                return False
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return False

def verify_model():
    """Verify the downloaded model"""
    if not os.path.exists(MODEL_PATH):
        return False, "File not found"
    
    file_size = os.path.getsize(MODEL_PATH)
    if file_size < EXPECTED_SIZE * 0.95:
        return False, f"File too small ({file_size/1024/1024:.1f}MB)"
    
    try:
        # Quick check for Keras format
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(4)
            if header != b'PK\x03\x04':  # ZIP signature
                return False, "Not a valid .keras file"
        return True, f"✅ Valid model ({file_size/1024/1024:.1f}MB)"
    except Exception as e:
        return False, f"Verification failed: {str(e)}"

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🧠 Large Model Loader (310MB)")

# Download Section
st.header("1. Download Options")

col1, col2 = st.columns(2)
with col1:
    if st.button("Method 1: Download with gdown"):
        for attempt in range(MAX_RETRIES):
            if download_with_gdown():
                break
            if attempt < MAX_RETRIES - 1:
                st.warning(f"Retrying... ({attempt + 1}/{MAX_RETRIES})")

with col2:
    if st.button("Method 2: Download with requests"):
        for attempt in range(MAX_RETRIES):
            if download_with_requests():
                break
            if attempt < MAX_RETRIES - 1:
                st.warning(f"Retrying... ({attempt + 1}/{MAX_RETRIES})")

# Verification Section
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
                    
                    # Model info
                    st.subheader("Model Information")
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Input shape:**", model.input_shape)
                    with cols[1]:
                        st.write("**Output shape:**", model.output_shape)
                    
                    # Prediction test
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

# Debug Info
st.header("🛠️ Troubleshooting")
st.markdown("""
### If downloads fail:
1. **Check your Google Drive link**:
   - Ensure the file is shared with "Anyone with the link"
   - Try accessing it manually: [Download Link](https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh)

2. **Alternative solutions**:
   - Upload your model to another service (Dropbox, S3, etc.)
   - Split the model into smaller parts using:
     ```python
     split -b 100M my_model.keras my_model_part_
     ```
   - Convert to TensorFlow Lite for smaller size:
     ```python
     converter = tf.lite.TFLiteConverter.from_keras_model(model)
     tflite_model = converter.convert()
     ```

### Current Environment:
```python
Working directory: {cwd}
Files present: {files}
Python version: {py_version}
""".format(
    cwd=os.getcwd(),
    files=os.listdir(),
    py_version=sys.version
))
