import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download

# Configuración de la página
st.set_page_config(
    page_title="Detector de Retinopatía Diabética",
    page_icon="👁️",
    layout="centered"
)

# Título y descripción
st.title("Detector de Retinopatía Diabética")
st.markdown("""
Esta aplicación utiliza un modelo de aprendizaje profundo (CNN) para clasificar imágenes
de fondo de retina y detectar posibles casos de retinopatía diabética.
""")

# Función para descargar y cargar el modelo desde Hugging Face
@st.cache_resource
def cargar_modelo():
    """Descarga y carga el modelo CNN desde Hugging Face Hub"""
    try:
        # Crear directorio para el modelo si no existe
        os.makedirs("modelo", exist_ok=True)
        
        # Información de Hugging Face Hub
        # IMPORTANTE: Reemplaza estos valores con los de tu repositorio en Hugging Face
        repo_id = "tu-usuario/retinopatia-diabetica-cnn"  # Reemplaza con tu usuario y nombre de repo
        filename = "modelo_cnn_retina.h5"  # Nombre del archivo en Hugging Face
        
        with st.spinner("Descargando modelo desde Hugging Face Hub..."):
            # Descargar modelo
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="modelo"
            )
            
            # Cargar el modelo
            model = load_model(model_path)
            st.success("Modelo cargado correctamente!")
            return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    """Preprocesa la imagen para que sea compatible con el modelo CNN"""
    # Redimensionar la imagen al tamaño que espera el modelo (ajusta según tu modelo)
    imagen = imagen.resize((224, 224))
    
    # Convertir a array y normalizar
    img_array = img_to_array(imagen)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización
    
    return img_array

# Función para realizar la predicción
def predecir_retinopatia(modelo, imagen_preprocesada):
    """Realiza la predicción usando el modelo cargado"""
    prediccion = modelo.predict(imagen_preprocesada)
    return prediccion

# Interfaz para subir archivos
st.subheader("Subir imagen de fondo de retina")
imagen_subida = st.file_uploader("Selecciona una imagen de fondo de retina", type=["jpg", "jpeg", "png"])

# Si se ha subido una imagen
if imagen_subida is not None:
    # Mostrar la imagen subida
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption="Imagen subida", use_column_width=True)
    
    # Botón para analizar la imagen
    if st.button("Analizar imagen"):
        with st.spinner("Analizando imagen..."):
            # Cargar el modelo
            modelo = cargar_modelo()
            
            if modelo:
                # Preprocesar la imagen
                imagen_preprocesada = preprocesar_imagen(imagen)
                
                # Realizar predicción
                resultado = predecir_retinopatia(modelo, imagen_preprocesada)
                
                # Interpretar resultado (ajusta según tu modelo)
                probabilidad = resultado[0][0]  # Asumiendo que es un modelo binario
                
                # Mostrar resultado
                st.subheader("Resultado del análisis")
                
                if probabilidad > 0.5:
                    st.error(f"Retinopatía diabética detectada con {probabilidad:.2%} de probabilidad")
                else:
                    st.success(f"No se detecta retinopatía diabética ({1-probabilidad:.2%} de confianza)")
                
                # Visualización de la probabilidad
                st.progress(float(probabilidad))
                
                # Consejos adicionales
                st.info("Este análisis es preliminar y no sustituye el diagnóstico médico profesional. Consulte a un oftalmólogo para una evaluación completa.")
            else:
                st.error("No se pudo cargar el modelo. Por favor, verifica que el modelo esté correctamente configurado.")

# Información adicional
st.markdown("---")
st.subheader("Sobre esta aplicación")
st.markdown("""
Esta aplicación utiliza un modelo de Redes Neuronales Convolucionales (CNN) entrenado para 
detectar signos de retinopatía diabética en imágenes de fondo de retina.

La retinopatía diabética es una complicación de la diabetes que afecta a los ojos y puede 
llevar a la pérdida de visión si no se detecta y trata a tiempo.
""")
