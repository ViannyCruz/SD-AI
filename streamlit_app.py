import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# App title
st.title("🔍 CNN Model Loader Test")
st.markdown("This app tests loading a Keras model from Google Drive")

# Configuration
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with your file ID
MODEL_PATH = "test_model.keras"

def download_model():
    """Download model from Google Drive with progress"""
    with st.spinner('Downloading model from Google Drive...'):
        progress_bar = st.progress(0)
        
        def update_progress(current, total, width=80):
            progress = int(current / total * 100)
            progress_bar.progress(progress)
        
        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=True,
            fuzzy=True,
            progress=update_progress
        )
        progress_bar.empty()
    st.success("Model downloaded successfully!")

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        with st.spinner('Loading model...'):
            model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

def test_model_prediction(model):
    """Test making a dummy prediction"""
    try:
        # Create a dummy input (adjust shape to match your model's expected input)
        dummy_input = np.random.rand(1, 224, 224, 3)  # Example for 224x224 RGB image
        
        with st.spinner('Testing prediction...'):
            prediction = model.predict(dummy_input)
        
        st.success(f"🎉 Prediction test successful! Output shape: {prediction.shape}")
        st.code(f"Sample output values:\n{prediction[0][:5]}")  # Show first 5 values
        return True
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return False

# Main app flow
if not os.path.exists(MODEL_PATH):
    download_model()

model = test_model_loading()

if model is not None:
    st.subheader("Model Summary")
    # Create expandable model summary
    with st.expander("Show model architecture"):
        try:
            from io import StringIO
            import sys
            
            # Capture model summary
            buffer = StringIO()
            sys.stdout = buffer
            model.summary()
            sys.stdout = sys.__stdout__
            
            st.text(buffer.getvalue())
        except Exception as e:
            st.warning(f"Couldn't show model summary: {str(e)}")
    
    # Test prediction
    st.subheader("Prediction Test")
    test_model_prediction(model)

# Add some debug info
st.subheader("Debug Information")
st.write(f"Model file exists: {os.path.exists(MODEL_PATH)}")
st.write(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")

# Cleanup instructions
st.markdown("""
---
### How to Use This Test:
1. Upload your `.keras` model to Google Drive
2. Right-click → Share → "Anyone with the link"
3. Get the file ID from the shareable link
4. Replace `YOUR_FILE_ID` in the code with your actual file ID
5. Run the app with `streamlit run app.py`
""")
