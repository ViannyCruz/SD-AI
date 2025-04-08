import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download
import time

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

# Función para cargar modelo local
def cargar_modelo_local(ruta_modelo):
    try:
        modelo = load_model(ruta_modelo)
        return modelo, None
    except Exception as e:
        return None, str(e)

# Función para descargar y cargar el modelo desde Hugging Face
def cargar_modelo_huggingface():
    try:
        # Crear directorio para el modelo si no existe
        os.makedirs("modelo", exist_ok=True)
        
        # Información de Hugging Face Hub
        repo_id = "Ruthzen/RDCNN"
        filename = "best_model.h5"
        
        # Intentar descargar modelo
        with st.spinner(f"Descargando modelo desde {repo_id}..."):
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="modelo",
                force_download=True
            )
            
            # Cargar el modelo
            model = load_model(model_path)
            return model, None
    except Exception as e:
        return None, str(e)

# Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    """Preprocesa la imagen para que sea compatible con el modelo CNN"""
    # Redimensionar la imagen al tamaño que espera el modelo
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

# Sidebar para opciones
with st.sidebar:
    st.title("Configuración")
    
    # Opción para seleccionar fuente del modelo
    opcion_modelo = st.radio(
        "Fuente del modelo:",
        ["Hugging Face", "Archivo local"]
    )
    
    if opcion_modelo == "Archivo local":
        st.info("Selecciona tu archivo de modelo .h5 local.")
        archivo_modelo = st.file_uploader("Subir modelo", type=["h5"])
        
        if archivo_modelo:
            # Guardar el archivo temporalmente
            with open("modelo_temporal.h5", "wb") as f:
                f.write(archivo_modelo.getbuffer())
            st.success(f"Archivo '{archivo_modelo.name}' guardado temporalmente")
    
    st.markdown("---")
    
    # Información sobre retinopatía
    st.title("Información")
    st.info("""
    **¿Qué es la retinopatía diabética?**
    
    La retinopatía diabética es una complicación de la diabetes que daña los vasos sanguíneos en la retina.
    
    **Síntomas comunes:**
    - Visión borrosa o fluctuante
    - Áreas oscuras en el campo visual
    - Dificultad para percibir colores
    - Pérdida de visión
    """)
    
    st.warning("""
    **Aviso importante:**
    
    Esta herramienta es solo para fines educativos e informativos. No sustituye el diagnóstico profesional.
    """)
    
    st.write("Desarrollado con ❤️ usando Streamlit y TensorFlow")

# Sección de carga del modelo
with st.expander("Estado del modelo", expanded=True):
    st.write("Verificando disponibilidad del modelo...")
    
    # Carga del modelo según la opción seleccionada
    if opcion_modelo == "Hugging Face":
        modelo, error = cargar_modelo_huggingface()
        
        if modelo is not None:
            st.success("✅ Modelo cargado correctamente desde Hugging Face")
            modelo_disponible = True
        else:
            st.error(f"❌ Error al cargar el modelo: {error}")
            st.error(f"Detalles: Intentando cargar desde Ruthzen/RDCNN, archivo best_model.h5")
            st.info("Prueba cargando tu modelo local usando la opción en el panel lateral.")
            modelo_disponible = False
    else:
        # Carga desde archivo local
        if 'archivo_modelo' in locals() and archivo_modelo is not None:
            modelo, error = cargar_modelo_local("modelo_temporal.h5")
            
            if modelo is not None:
                st.success(f"✅ Modelo '{archivo_modelo.name}' cargado correctamente")
                modelo_disponible = True
            else:
                st.error(f"❌ Error al cargar el modelo local: {error}")
                modelo_disponible = False
        else:
            st.warning("Por favor, sube un archivo de modelo .h5 usando el panel lateral.")
            modelo_disponible = False
    
    # Mostrar información del modelo si está disponible
    if 'modelo_disponible' in locals() and modelo_disponible:
        st.write("Información del modelo:")
        st.code(f"Capas: {len(modelo.layers)}")
        st.code(f"Forma de entrada: {modelo.input_shape}")
        st.code(f"Forma de salida: {modelo.output_shape}")

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
        if 'modelo_disponible' in locals() and modelo_disponible:
            with st.spinner("Analizando imagen..."):
                # Preprocesar la imagen
                imagen_preprocesada = preprocesar_imagen(imagen)
                
                # Realizar predicción
                resultado = predecir_retinopatia(modelo, imagen_preprocesada)
                
                # Interpretar resultado (asumiendo que es un modelo binario)
                probabilidad = resultado[0][0]
                
                # Mostrar resultado
                st.subheader("Resultado del análisis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if probabilidad > 0.5:
                        st.error(f"Retinopatía diabética detectada")
                        st.metric("Probabilidad", f"{probabilidad:.2%}")
                    else:
                        st.success(f"No se detecta retinopatía diabética")
                        st.metric("Confianza", f"{1-probabilidad:.2%}")
                
                with col2:
                    # Visualización de la probabilidad
                    st.write("Nivel de confianza:")
                    st.progress(float(probabilidad))
                
                # Consejos adicionales
                st.info("Este análisis es preliminar y no sustituye el diagnóstico médico profesional. Consulte a un oftalmólogo para una evaluación completa.")
        else:
            st.error("No se puede realizar el análisis porque el modelo no está disponible.")
            if opcion_modelo == "Hugging Face":
                st.info("Prueba cargando tu modelo local desde el panel lateral.")
            else:
                st.info("Por favor, sube tu archivo de modelo .h5.")

# Información adicional
st.markdown("---")
st.subheader("Sobre esta aplicación")
st.markdown("""
Esta aplicación utiliza un modelo de Redes Neuronales Convolucionales (CNN) entrenado para 
detectar signos de retinopatía diabética en imágenes de fondo de retina.

La retinopatía diabética es una complicación de la diabetes que afecta a los ojos y puede 
llevar a la pérdida de visión si no se detecta y trata a tiempo.
""")
