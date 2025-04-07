import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import gdown
import os

# Configuración de la página
st.set_page_config(
    page_title="Herramienta Diagnóstica de Imágenes Médicas",
    page_icon="🩺",
    layout="wide"
)

# Funciones auxiliares
@st.cache_resource
def cargar_modelo():
    """Cargar el modelo entrenado desde Google Drive con manejo mejorado de errores"""
    try:
        # Ruta donde se guardará temporalmente el modelo descargado
        modelo_path = 'best_model.keras'
        
        # Verificar si el modelo ya existe localmente
        if not os.path.exists(modelo_path):
            # Obtener ID del modelo desde secrets o usar el valor por defecto
            drive_id = st.secrets.get("DRIVE_MODEL_ID", "13S8aXIDQpixM5Siy-0tWHSm2MEHw1Ksh")
            
            # Método 1: Intentar con gdown directamente
            try:
                with st.spinner("Descargando modelo desde Google Drive (Método 1)..."):
                    drive_url = f"https://drive.google.com/uc?id={drive_id}"
                    output = gdown.download(drive_url, modelo_path, quiet=False)
                    
                    if output is None:
                        raise Exception("La descarga con gdown ha fallado")
            except Exception as e1:
                st.warning(f"Primer método de descarga falló: {str(e1)}")
                
                # Método 2: Intentar con parámetros adicionales
                try:
                    with st.spinner("Descargando modelo con método alternativo..."):
                        # Intentar usar la opción fuzzy=True que puede ayudar con ciertos enlaces
                        output = gdown.download(drive_url, modelo_path, quiet=False, fuzzy=True)
                        
                        if output is None:
                            raise Exception("La descarga con gdown (fuzzy) ha fallado")
                except Exception as e2:
                    st.warning(f"Segundo método de descarga falló: {str(e2)}")
                    
                    # Método 3: Usar URL directa si está configurada
                    try:
                        with st.spinner("Intentando descarga directa..."):
                            direct_url = st.secrets.get("DIRECT_MODEL_URL", None)
                            if direct_url:
                                import requests
                                response = requests.get(direct_url)
                                if response.status_code == 200:
                                    with open(modelo_path, 'wb') as f:
                                        f.write(response.content)
                                    st.success("Descarga directa exitosa")
                                else:
                                    raise Exception(f"Error en descarga directa: {response.status_code}")
                            else:
                                raise Exception("No hay URL directa configurada")
                    except Exception as e3:
                        st.error(f"Todos los métodos de descarga fallaron.")
                        st.info("""
                        Por favor, resuelva el problema con una de estas opciones:
                        1. Verifique la URL y permisos del archivo en Google Drive
                        2. Configure su archivo como 'Cualquiera con el enlace puede ver'
                        3. Suba el modelo a algún servicio como GitHub Releases, AWS S3 o Hugging Face
                        4. Coloque el archivo 'best_model.keras' en la misma carpeta que este script
                        """)
                        return None
        
        # Si llegamos aquí, o bien el archivo ya existía localmente o se descargó con éxito
        if os.path.exists(modelo_path):
            # Cargar el modelo descargado
            modelo = tf.keras.models.load_model(modelo_path)
            st.success("Modelo cargado exitosamente")
            return modelo
        else:
            st.error("No se pudo encontrar o descargar el modelo")
            return None
            
    except Exception as e:
        st.error(f"Error inesperado al cargar el modelo: {str(e)}")
        return None

def preprocesar_imagen(imagen, tamano_objetivo=(256, 256)):
    """Preprocesar la imagen para la predicción del modelo"""
    # Redimensionar al tamaño objetivo
    imagen = imagen.resize(tamano_objetivo)
    # Convertir a array y normalizar
    array_img = img_to_array(imagen)
    array_img = array_img / 255.0
    # Expandir dimensiones para el lote
    array_img = np.expand_dims(array_img, axis=0)
    return array_img

def predecir_imagen(modelo, array_img, nombres_clases):
    """Realizar predicción en la imagen"""
    prediccion = modelo.predict(array_img)
    clase_predicha = int(np.round(prediccion[0][0]))
    probabilidad = prediccion[0][0] if clase_predicha == 1 else 1 - prediccion[0][0]
    
    return {
        'clase': nombres_clases[clase_predicha],
        'probabilidad': float(probabilidad),
        'prediccion_bruta': float(prediccion[0][0])
    }

def crear_visualizacion_resultado(resultado, imagen, nombres_clases):
    """Crear una representación visual del resultado diagnóstico"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    # Mostrar la imagen
    ax1.imshow(imagen)
    ax1.set_title("Imagen Analizada")
    ax1.axis('off')
    
    # Crear la visualización del resultado diagnóstico
    # Gráfico vacío con texto
    ax2.axis('off')
    
    # Determinar el color del resultado basado en la predicción
    color_resultado = 'green' if resultado['clase'] == nombres_clases[0] else 'red'
    confianza = resultado['probabilidad'] * 100
    
    # Mostrar información diagnóstica
    ax2.text(0.5, 0.8, f"RESULTADO DEL DIAGNÓSTICO", 
             ha='center', va='center', fontsize=18, fontweight='bold')
    
    ax2.text(0.5, 0.6, f"{resultado['clase']}", 
             ha='center', va='center', color=color_resultado, fontsize=24, fontweight='bold')
    
    ax2.text(0.5, 0.4, f"Confianza: {confianza:.1f}%", 
             ha='center', va='center', fontsize=16)
    
    # Añadir un cuadro coloreado basado en el resultado
    rect = plt.Rectangle((0.1, 0.2), 0.8, 0.6, fill=True, color=color_resultado, alpha=0.1)
    ax2.add_patch(rect)
    
    plt.tight_layout()
    return fig

# Interfaz principal de la aplicación
st.title("🩺 Herramienta Diagnóstica de Imágenes Médicas")
st.write("Suba una imagen médica para recibir un diagnóstico automatizado usando nuestro modelo de IA")

# Barra lateral para configuración
with st.sidebar:
    st.header("Configuración")
    
    # Nombres de las clases
    st.subheader("Nombres de las clases")
    clase_0 = st.text_input("Nombre de clase negativa", "Normal")
    clase_1 = st.text_input("Nombre de clase positiva", "Anormal")
    nombres_clases = [clase_0, clase_1]
    
    # Caja de información
    st.info("""
    Esta aplicación utiliza un modelo de aprendizaje profundo para analizar imágenes médicas.
    
    El modelo ha sido entrenado en un conjunto de datos de imágenes médicas para detectar anomalías.
    
    Para mejores resultados, suba imágenes claras en formato JPG, PNG o TIFF.
    """)
    
    # Información del modelo
    with st.expander("Acerca del Modelo"):
        st.write("""
        El modelo utilizado en esta aplicación es una Red Neuronal Convolucional (CNN) 
        entrenada para la clasificación binaria de imágenes médicas.
        
        Fue entrenado utilizando validación cruzada de 5 pliegues para asegurar robustez 
        y utiliza aumento de datos para mejorar la generalización.
        
        El modelo logró:
        - Precisión: ~85-90%
        - Sensibilidad: ~85%
        - Especificidad: ~88%
        
        Nota: Esta herramienta está destinada a asistir en el diagnóstico y 
        no debe reemplazar el consejo médico profesional.
        """)

# Área de contenido principal
# Crear un widget para subir archivos
archivo_subido = st.file_uploader("Subir una imagen médica", type=["jpg", "jpeg", "png", "tif", "tiff"])

# Crear dos columnas para el diseño
col1, col2 = st.columns([1, 1])

# Si se sube un archivo
if archivo_subido is not None:
    # Cargar y mostrar la imagen
    with col1:
        imagen = Image.open(archivo_subido)
        st.image(imagen, caption="Imagen Subida", use_column_width=True)
    
    # Añadir un botón para ejecutar el diagnóstico
    boton_diagnostico = st.button("Ejecutar Diagnóstico", type="primary")
    
    if boton_diagnostico:
        # Cargar el modelo
        modelo = cargar_modelo()
        
        if modelo:
            with st.spinner("Analizando imagen..."):
                # Preprocesar la imagen
                array_img = preprocesar_imagen(imagen)
                
                # Hacer predicción
                resultado = predecir_imagen(modelo, array_img, nombres_clases)
                
                # Mostrar resultado diagnóstico
                st.subheader("Resultado Diagnóstico")
                
                # Crear un cuadro coloreado para el resultado con CSS personalizado
                color_resultado = "green" if resultado['clase'] == nombres_clases[0] else "red"
                confianza = resultado['probabilidad'] * 100
                
                st.markdown(f"""
                <div style="padding: 20px; 
                            border-radius: 10px; 
                            background-color: {color_resultado}15; 
                            border: 2px solid {color_resultado};">
                    <h2 style="text-align: center; color: {color_resultado};">{resultado['clase']}</h2>
                    <p style="text-align: center; font-size: 18px;">Confianza: {confianza:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Crear visualización
                fig = crear_visualizacion_resultado(resultado, imagen, nombres_clases)
                st.pyplot(fig)
                
                # Mostrar detalles técnicos adicionales en un expansor
                with st.expander("Detalles Técnicos"):
                    st.json({
                        "Clase Predicha": resultado['clase'],
                        "Confianza": f"{resultado['probabilidad']:.4f}",
                        "Salida Bruta del Modelo": resultado['prediccion_bruta']
                    })
                
                # Añadir descargo de responsabilidad
                st.caption("""
                AVISO LEGAL: Esta herramienta es solo para fines educativos y no está destinada para uso clínico. 
                Siempre consulte con profesionales de la salud para diagnósticos médicos.
                """)

else:
    # Mostrar mensaje de instrucción cuando no se sube ningún archivo
    st.info("Por favor, suba una imagen para comenzar.")
    
    # Añadir una imagen de demostración (opcional)
    st.markdown("### Vista Previa de Resultado de Muestra")
    st.image("https://via.placeholder.com/800x400.png?text=Ejemplo+de+Resultado+Diagnóstico", 
             caption="Ejemplo de visualización de diagnóstico (suba una imagen para ver sus resultados)")

# Pie de página
st.markdown("---")
st.markdown("Herramienta Diagnóstica de Imágenes Médicas | Desarrollado con TensorFlow")
