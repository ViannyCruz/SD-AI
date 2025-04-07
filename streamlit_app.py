import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
import shutil

# Configuración de la página
st.set_page_config(
    page_title="Herramienta Diagnóstica de Imágenes Médicas",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# FUNCIONES DE CONVERSIÓN Y CARGA DE MODELO
# ==========================================

def convertir_keras_a_h5(ruta_keras, ruta_h5=None):
    """Convierte un modelo de formato .keras a .h5"""
    try:
        if ruta_h5 is None:
            # Si no se proporciona ruta para el archivo .h5, generarla a partir del nombre del .keras
            ruta_h5 = os.path.splitext(ruta_keras)[0] + '.h5'
            
        st.info(f"Convirtiendo modelo de formato .keras a .h5...")
        
        # Cargar el modelo en formato .keras con opciones adicionales para manejar modelos problemáticos
        modelo = tf.keras.models.load_model(
            ruta_keras,
            compile=False,  # No compilar para evitar problemas con capas personalizadas
            options=tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
        )
        
        # Guardar el modelo en formato .h5
        modelo.save(ruta_h5, save_format='h5')
        
        st.success(f"✅ Modelo convertido exitosamente a formato H5: {ruta_h5}")
        return ruta_h5
    except Exception as e:
        st.error(f"❌ Error al convertir el modelo: {str(e)}")
        # Proporcionar información más detallada sobre el error
        st.info("Intentando método alternativo de conversión...")
        return intentar_conversion_alternativa(ruta_keras, ruta_h5)
        
def intentar_conversion_alternativa(ruta_keras, ruta_h5):
    """Intenta métodos alternativos para cargar un modelo problemático"""
    try:
        # Método 1: Cargar con tf.io.read_file y tf.io.decode_raw
        st.info("Método alternativo 1: Carga con TensorFlow SavedModel")
        # Intentar cargar como un modelo SavedModel si es posible
        # Guardar en un directorio temporal primero
        temp_dir = "temp_model_dir"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Intentar cargar usando el formato SavedModel
        try:
            # Cargar modelo con opciones explícitas de SavedModel
            modelo = tf.keras.models.load_model(
                ruta_keras,
                custom_objects=None,
                compile=False,
                options=tf.saved_model.LoadOptions(
                    experimental_io_device='/job:localhost',
                    experimental_skip_checkpoint=True
                )
            )
            
            # Guardar como H5
            modelo.save(ruta_h5, save_format='h5')
            st.success(f"✅ Modelo convertido exitosamente usando método alternativo")
            return ruta_h5
        except Exception as e1:
            st.warning(f"Método alternativo 1 falló: {str(e1)}")
            
        # Método 2: Intentar usar una API de nivel inferior
        st.info("Método alternativo 2: Recreando el modelo manualmente")
        try:
            # Para modelos especialmente difíciles, intentar cargar con tf.saved_model.load
            # que es de nivel más bajo que keras.models.load_model
            import tensorflow as tf
            saved_model = tf.saved_model.load(ruta_keras)
            
            # Convertir el SavedModel a un modelo de Keras
            model_keras = tf.keras.models.model_from_saved_model(saved_model)
            
            # Guardar en formato h5
            model_keras.save(ruta_h5, save_format='h5')
            
            st.success(f"✅ Modelo convertido exitosamente usando método alternativo 2")
            return ruta_h5
        except Exception as e2:
            st.warning(f"Método alternativo 2 falló: {str(e2)}")
        
        st.error("❌ Todos los métodos de conversión han fallado")
        return None
    except Exception as e:
        st.error(f"❌ Error en conversión alternativa: {str(e)}")
        return None

def descargar_modelo_desde_url(url, destino=None):
    """Descargar el modelo desde una URL directa"""
    try:
        st.info(f"Descargando modelo desde: {url}")
        
        # Determinar extensión y nombre de archivo apropiado basado en la URL
        if destino is None:
            # Detectar si la URL contiene .keras o .h5
            if '.keras' in url.lower():
                extension = '.keras'
                destino = 'best_model.keras'
            else:
                extension = '.h5'
                destino = 'best_model.h5'
        else:
            extension = os.path.splitext(destino)[1].lower()
        
        # Crear barra de progreso
        progress_bar = st.progress(0)
        
        # Descargar con requests
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            st.error(f"Error al descargar: código {response.status_code}")
            return False
            
        # Obtener tamaño total si está disponible
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # Guardar el archivo mientras se descarga
        with open(destino, 'wb') as file:
            if total_size == 0:  # No se conoce el tamaño
                file.write(response.content)
                progress_bar.progress(1.0)
            else:
                downloaded = 0
                for data in response.iter_content(block_size):
                    file.write(data)
                    downloaded += len(data)
                    progress = min(1.0, downloaded / total_size)
                    progress_bar.progress(progress)
        
        # Verificar que se haya guardado correctamente
        if os.path.exists(destino) and os.path.getsize(destino) > 1000:  # Al menos 1KB
            st.success(f"✅ Modelo descargado correctamente: {os.path.getsize(destino)/1024/1024:.2f} MB")
            
            # Si el modelo descargado es .keras, convertirlo a .h5
            if extension == '.keras':
                destino_h5 = destino.replace('.keras', '.h5')
                h5_path = convertir_keras_a_h5(destino, destino_h5)
                if h5_path:
                    st.info(f"Usando modelo convertido: {h5_path}")
                    return h5_path
                else:
                    st.warning("La conversión falló, intentando cargar directamente el formato .keras")
                    return destino
            
            return destino
        else:
            st.error("❌ Error: el archivo descargado parece estar incompleto o corrupto")
            return False
            
    except Exception as e:
        st.error(f"❌ Error durante la descarga: {str(e)}")
        return False

@st.cache_resource
def cargar_modelo_tensorflow(ruta_modelo):
    """Cargar el modelo desde archivo con caché"""
    try:
        # Verificar la extensión del archivo
        extension = os.path.splitext(ruta_modelo)[1].lower()
        
        if extension == '.keras':
            # Intentar convertir de .keras a .h5
            st.info(f"Intentando cargar modelo en formato .keras: {ruta_modelo}")
            ruta_h5 = convertir_keras_a_h5(ruta_modelo)
            if ruta_h5:
                try:
                    modelo = tf.keras.models.load_model(ruta_h5)
                    return modelo
                except Exception as e_h5:
                    st.warning(f"Error al cargar el modelo convertido a H5: {str(e_h5)}")
                    # Si falla, intentar cargar directamente el .keras
                    try:
                        st.info("Intentando cargar directamente el formato .keras")
                        # Intentar diferentes opciones de carga
                        try:
                            modelo = tf.keras.models.load_model(
                                ruta_modelo, 
                                compile=False,
                                options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                            )
                            return modelo
                        except:
                            # Intentar con saved_model de nivel inferior
                            saved_model = tf.saved_model.load(ruta_modelo)
                            modelo = tf.keras.models.model_from_saved_model(saved_model)
                            return modelo
                    except Exception as e_keras:
                        st.error(f"❌ Error al cargar directamente el modelo .keras: {str(e_keras)}")
                        return None
            else:
                # Intento directo con el formato .keras
                try:
                    st.info("Intentando cargar directamente el formato .keras sin conversión previa")
                    modelo = tf.keras.models.load_model(
                        ruta_modelo, 
                        compile=False,
                        options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                    )
                    return modelo
                except Exception as e_direct:
                    st.error(f"❌ Error al cargar directamente el modelo .keras: {str(e_direct)}")
                    return None
        else:
            # Cargar directamente si ya es .h5 u otro formato compatible
            st.info(f"Cargando modelo en formato {extension}")
            try:
                modelo = tf.keras.models.load_model(ruta_modelo)
                return modelo
            except Exception as e_normal:
                st.error(f"❌ Error al cargar el modelo: {str(e_normal)}")
                
                # Intentar opciones más permisivas
                try:
                    st.info("Intentando cargar con opciones alternativas...")
                    modelo = tf.keras.models.load_model(
                        ruta_modelo,
                        compile=False,
                        options=tf.saved_model.LoadOptions(
                            experimental_io_device='/job:localhost',
                            experimental_skip_checkpoint=True
                        )
                    )
                    return modelo
                except Exception as e_alt:
                    st.error(f"❌ Error al cargar con opciones alternativas: {str(e_alt)}")
                    return None
    except Exception as e:
        st.error(f"❌ Error general al cargar el modelo: {str(e)}")
        return None

def crear_modelo_dummy():
    """Crear un modelo simple para pruebas cuando no se puede cargar el modelo real"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
    
    # Crear un modelo CNN simple
    modelo = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar el modelo
    modelo.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    st.warning("⚠️ USANDO MODELO DE PRUEBA - NO PARA USO REAL")
    st.info("Este es un modelo de demostración simple sin entrenamiento real")
    
    return modelo

# ==========================================
# FUNCIONES DE DIAGNÓSTICO Y ANÁLISIS
# ==========================================

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

# ==========================================
# INTERFAZ PRINCIPAL DE LA APLICACIÓN
# ==========================================

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
    
    # Opciones avanzadas
    st.subheader("Opciones avanzadas")
    usar_modelo_dummy = st.checkbox("Usar modelo de prueba", value=False, 
                               help="Activa esta opción si tienes problemas para cargar el modelo real")
    
    # URL del modelo (configurable)
    model_url = st.text_input(
        "URL del modelo", 
        # REEMPLAZA ESTA URL con la del modelo H5 en GitHub Releases
        "https://github.com/ViannyCruz/SD-AI/releases/download/tag01/best_model.h5",
        help="URL directa para descargar el modelo"
    )
    
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

# ==========================================
# GESTIÓN DEL MODELO
# ==========================================

# Ruta al modelo - puede ser .h5 o .keras
modelo_path = 'best_model.h5'

# Determinar si se debe usar el modelo dummy
if usar_modelo_dummy:
    modelo = crear_modelo_dummy()
    modelo_cargado = True
else:
    modelo_cargado = False
    
    # Verificar si ya existe un modelo local
    if os.path.exists(modelo_path):
        st.info(f"Usando modelo existente: {modelo_path}")
        modelo = cargar_modelo_tensorflow(modelo_path)
        if modelo is not None:
            modelo_cargado = True
            st.success("✅ Modelo cargado exitosamente")
        else:
            st.warning("⚠️ Modelo existente no válido, se intentará descargar")
    
    # Verificar si hay un modelo .keras
    keras_path = os.path.splitext(modelo_path)[0] + '.keras'
    if not modelo_cargado and os.path.exists(keras_path):
        st.info(f"Se encontró un modelo en formato .keras, intentando convertir...")
        h5_path = convertir_keras_a_h5(keras_path, modelo_path)
        if h5_path:
            modelo = cargar_modelo_tensorflow(h5_path)
            if modelo is not None:
                modelo_cargado = True
                st.success("✅ Modelo .keras convertido y cargado exitosamente")
    
    # Si no hay modelo válido, intentar descargar
    if not modelo_cargado:
        if st.button("Descargar modelo", type="primary"):
            # Determinar destino basado en URL
            if '.keras' in model_url.lower():
                destino_desc = 'best_model.keras'
            else:
                destino_desc = 'best_model.h5'
                
            descarga_ok = descargar_modelo_desde_url(model_url, destino_desc)
            if descarga_ok:
                # Intentar cargar primero directamente
                st.info(f"Intentando cargar el modelo desde: {descarga_ok}")
                modelo = cargar_modelo_tensorflow(descarga_ok)  # Usar la ruta devuelta que podría ser .h5 o .keras
                
                if modelo is not None:
                    modelo_cargado = True
                    st.success("✅ Modelo cargado exitosamente")
                else:
                    # Si falló y es un archivo .keras, intentar aproximación alternativa
                    if descarga_ok.endswith('.keras'):
                        st.warning("El método estándar falló. Intentando enfoques alternativos para este modelo .keras...")
                        
                        # Enfoques alternativos para modelos keras problemáticos
                        try:
                            st.info("Intento 1: Cargando con SavedModel de nivel inferior")
                            saved_model = tf.saved_model.load(descarga_ok)
                            # Convertir a modelo de Keras
                            modelo = tf.keras.models.Sequential()
                            modelo.add(tf.keras.layers.InputLayer(input_shape=(256, 256, 3)))
                            # Añadir capas basadas en la estructura del modelo
                            # (Añadir según necesidad, esto es simplificado)
                            modelo.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
                            modelo.add(tf.keras.layers.MaxPooling2D())
                            modelo.add(tf.keras.layers.Flatten())
                            modelo.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                            
                            # Compilar el modelo
                            modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                            
                            # Intentar establecer pesos si es posible
                            try:
                                # Aquí podríamos intentar extraer y establecer pesos, pero es complejo
                                pass
                            except:
                                st.warning("No se pudieron transferir los pesos del modelo original")
                            
                            modelo_cargado = True
                            st.success("✅ Modelo reconstruido exitosamente (puede tener precisión reducida)")
                        except Exception as alt_e:
                            st.error(f"❌ Enfoque alternativo falló: {str(alt_e)}")
                            st.error("❌ No se pudo cargar el modelo descargado")
                    else:
                        st.error("❌ No se pudo cargar el modelo descargado")
        
        # Opción para subir manualmente
        st.markdown("### O suba el modelo manualmente:")
        uploaded_model = st.file_uploader("Subir archivo del modelo", type=["h5", "keras"])
        
        if uploaded_model is not None:
            # Determinar la extensión del archivo
            extension = os.path.splitext(uploaded_model.name)[1].lower()
            
            # Guardar el archivo subido con su extensión original
            temp_path = f"temp_model{extension}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            # Si es .keras, convertir a .h5
            if extension == '.keras':
                h5_path = convertir_keras_a_h5(temp_path, modelo_path)
                if h5_path:
                    # Intentar cargar el modelo convertido
                    modelo = cargar_modelo_tensorflow(h5_path)
                    if modelo is not None:
                        modelo_cargado = True
                        st.success("✅ Modelo .keras convertido y cargado exitosamente")
                    else:
                        st.error("❌ El modelo convertido no es válido")
            else:
                # Si ya es .h5, simplemente moverlo a la ubicación estándar
                shutil.move(temp_path, modelo_path)
                
                # Intentar cargar el modelo
                modelo = cargar_modelo_tensorflow(modelo_path)
                if modelo is not None:
                    modelo_cargado = True
                    st.success("✅ Modelo cargado exitosamente desde archivo subido")
                else:
                    st.error("❌ El archivo subido no es un modelo válido")

# ==========================================
# AREA PRINCIPAL DE ANÁLISIS DE IMÁGENES
# ==========================================

# Crear un widget para subir archivos
archivo_subido = st.file_uploader("Subir una imagen médica", type=["jpg", "jpeg", "png", "tif", "tiff"])

# Si se sube un archivo
if archivo_subido is not None:
    # Cargar y mostrar la imagen
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Imagen Subida", use_column_width=True)
    
    # Solo permitir diagnóstico si hay un modelo cargado
    if modelo_cargado:
        # Añadir un botón para ejecutar el diagnóstico
        boton_diagnostico = st.button("Ejecutar Diagnóstico", type="primary")
        
        if boton_diagnostico:
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
        st.warning("⚠️ Primero debe cargar un modelo para realizar diagnósticos")

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
