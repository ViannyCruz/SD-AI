import streamlit as st
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# App title
st.title("🔍 CNN Model Loader Test")
st.markdown("This app tests loading a Keras model from Google Drive")

# Configuration - Using your specific URL
FILE_ID = "13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "loaded_model.keras"

def download_model():
    """Download model from Google Drive with progress"""
    if os.path.exists(MODEL_PATH):
        st.info("Model file already exists, skipping download.")
        return
    
    with st.spinner('Downloading model from Google Drive...'):
        progress_bar = st.progress(0)
        
        def update_progress(current, total, width=80):
            progress = int(current / total * 100)
            progress_bar.progress(progress)
        
        try:
            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=True,
                fuzzy=True,
                progress=update_progress
            )
            progress_bar.empty()
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            progress_bar.empty()

def test_model_loading():
    """Test if the model loads correctly"""
    try:
        with st.spinner('Loading model...'):
            model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("Common fixes:")
        st.markdown("""
        - Make sure TensorFlow version matches the model's version
        - Check if the file downloaded completely (current size: {:.1f}MB)
        - The file might be corrupted during download - try downloading manually
        """.format(os.path.getsize(MODEL_PATH)/(1024*1024) if os.path.exists(MODEL_PATH) else 0))
        return None

def test_model_prediction(model):
    """Test making a dummy prediction"""
    try:
        # Try common image input shapes
        test_shapes = [
            (1, 224, 224, 3),  # Common image size
            (1, 128, 128, 3),  # Smaller image
            (1, 256, 256, 3)   # Larger image
        ]
        
        successful = False
        for shape in test_shapes:
            try:
                dummy_input = np.random.rand(*shape).astype(np.float32)
                with st.spinner(f'Testing prediction with shape {shape}...'):
                    prediction = model.predict(dummy_input)
                
                st.success(f"🎉 Prediction successful with shape {shape}!")
                st.code(f"Output shape: {prediction.shape}\nFirst values: {prediction[0].flatten()[:5]}")
                successful = True
                break
            except Exception as e:
                continue
                
        if not successful:
            st.error("❌ Couldn't find compatible input shape. Please specify your model's expected input shape.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# Main app flow
st.header("1. Model Download")
