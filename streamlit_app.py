import streamlit as st
import os
import sys  # Added missing import
import requests
import gdown
from tensorflow.keras.models import load_model
import numpy as np
import shutil

# Configuration - UPDATE THESE FOR YOUR DEPLOYMENT
MODEL_URL = "https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_FILENAME = "my_model.keras"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)  # Absolute path
EXPECTED_SIZE = 310 * 1024 * 1024  # 310MB
MAX_RETRIES = 3

# Initialize session state
if 'download_complete' not in st.session_state:
    st.session_state.download_complete = False

def reset_environment():
    """Clear any existing files"""
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.session_state.download_complete = False

def verify_file_location():
    """Check where files are actually being stored"""
    st.warning("Checking file locations...")
    st.write(f"Current working directory: `{os.getcwd()}`")
    st.write("Files in directory:", os.listdir())
    
    if os.path.exists(MODEL_PATH):
        st.success(f"Model found at: `{MODEL_PATH}`")
        st.write(f"File size: {os.path.getsize(MODEL_PATH)/1024/1024:.1f}MB")
    else:
        st.error(f"Model NOT found at: `{MODEL_PATH}`")

def download_model():
    """Robust download with verification"""
    reset_environment()
    
    try:
        with st.spinner(f"Downloading {MODEL_FILENAME} (310MB)..."):
            # Using gdown with forced download
            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False,
                fuzzy=True,
                resume=False
            )
            
        # Verify download
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            if file_size >= EXPECTED_SIZE * 0.95:  # Allow 5% variance
                st.session_state.download_complete = True
                st.success(f"✅ Download complete! Size: {file_size/1024/1024:.1f}MB")
                return True
            else:
                st.error(f"❌ Incomplete download ({file_size/1024/1024:.1f}MB)")
        else:
            st.error("❌ File was not created")
        return False
        
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        return False

def verify_model_file():
    """Thorough model verification"""
    if not os.path.exists(MODEL_PATH):
        return False, "File not found at the expected location"
    
    try:
        file_size = os.path.getsize(MODEL_PATH)
        if file_size < EXPECTED_SIZE * 0.95:
            return False, f"File too small ({file_size/1024/1024:.1f}MB)"
        
        # Check file signature
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(4)
            if header != b'PK\x03\x04':  # ZIP signature
                return False, "Not a valid .keras ZIP file"
        
        return True, f"✅ Valid model file ({file_size/1024/1024:.1f}MB)"
    except Exception as e:
        return False, f"Verification error: {str(e)}"

# Streamlit UI
st.set_page_config(layout="wide", page_title="Model Loader")
st.title("🧠 Large Model Deployment")

# Debug section
with st.expander("🔍 Debug Information", expanded=False):
    verify_file_location()
    st.code(f"""
    Python version: {sys.version}
    Working directory: {os.getcwd()}
    Model path: {MODEL_PATH}
    Files present: {os.listdir()}
    """)

# Download section
st.header("1. Download Model")
if st.button("⬇️ Download Model File"):
    for attempt in range(MAX_RETRIES):
        if download_model():
            break
        if attempt < MAX_RETRIES - 1:
            st.warning(f"Attempt {attempt + 1} failed, retrying...")

# Verification section
st.header("2. Verify Model")
if os.path.exists(MODEL_PATH):
    is_valid, message = verify_model_file()
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
                st.markdown("""
                **Common Solutions:**
                1. Try downloading the file again
                2. Verify the file is a valid Keras model
                3. Check TensorFlow version compatibility
                """)
    else:
        st.error(message)
else:
    st.warning("No model file found - please download first")

# Reset option
st.header("3. Troubleshooting")
if st.button("🔄 Reset Environment"):
    reset_environment()
    st.success("Environment reset - ready to try again")
