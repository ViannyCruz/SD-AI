import streamlit as st
import os
import sys
import gdown
from tensorflow.keras.models import load_model
import numpy as np

# 1. Configuración de rutas compatible con Streamlit Cloud
MODEL_URL = "https://drive.google.com/uc?id=13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh"
MODEL_NAME = "my_model.keras"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_NAME)  # Ruta absoluta en el directorio de trabajo actual

# 2. Función de descarga robusta
def download_model():
    if os.path.exists(MODEL_PATH):
        st.warning("⚠️ El modelo ya existe. Borrando antes de redescargar...")
        os.remove(MODEL_PATH)
    
    try:
        with st.spinner(f"Descargando modelo (310MB) desde Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            if file_size > 300:  # Verifica que el tamaño sea >300MB
                st.success(f"✅ Descarga exitosa! Tamaño: {file_size:.1f}MB")
                return True
            else:
                st.error(f"❌ Archivo demasiado pequeño ({file_size:.1f}MB). ¿Descarga corrupta?")
        else:
            st.error("❌ El archivo no se creó correctamente")
        return False
    except Exception as e:
        st.error(f"❌ Error en descarga: {str(e)}")
        return False

# 3. Verificación de ubicación REAL del archivo
def debug_file_location():
    st.subheader("🔍 Debug: Ubicación Real")
    st.write(f"**Directorio actual:** `{os.getcwd()}`")
    st.write(f"**Archivos presentes:** `{os.listdir()}`")
    st.write(f"**Ruta esperada del modelo:** `{MODEL_PATH}`")
    if os.path.exists(MODEL_PATH):
        st.success(f"✔️ El modelo SÍ existe en la ruta esperada")
    else:
        st.error(f"✖️ El modelo NO está en la ruta esperada")

# Interfaz principal
st.title("🔧 Cargador de Modelos Grandes (310MB)")

# --- Sección de Descarga ---
st.header("1. Descargar Modelo")
if st.button("⬇️ Descargar Modelo"):
    if download_model():
        st.balloons()

# --- Sección de Debug --- 
debug_file_location()

# --- Sección de Carga ---
st.header("2. Cargar Modelo")
if os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Cargando modelo..."):
            model = load_model(MODEL_PATH)
            st.success("✅ ¡Modelo cargado correctamente!")
            
            # Test de predicción
            dummy_input = np.random.rand(*model.input_shape[1:]).astype(np.float32)
            prediction = model.predict(np.expand_dims(dummy_input, axis=0))
            st.success(f"🎉 ¡Predicción exitosa! Forma del output: {prediction.shape}")
            
    except Exception as e:
        st.error(f"❌ Error al cargar: {str(e)}")
        st.markdown("""
        **Soluciones comunes:**
        1. Verifica que el archivo sea un modelo Keras válido
        2. Revisa la compatibilidad de versiones de TensorFlow
        3. Usa el debug para confirmar la ubicación real
        """)
else:
    st.warning("⚠️ Primero descarga el modelo")

# --- Reset ---
if st.button("🔄 Reiniciar Todo"):
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.success("¡Entorno reiniciado!")
