import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Detector de Flamencos y Pingüinos",
    page_icon="🦩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff4b4b;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE UTILIDAD ---
@st.cache_resource
def load_model(model_path):
    """Carga el modelo YOLO y lo guarda en caché para velocidad."""
    if not os.path.exists(model_path):
        st.error(f"⚠️ No se encontró el modelo en: {model_path}. Asegúrate de haber ejecutado 'save_best_model' y tener la carpeta 'src'.")
        return None
    return YOLO(model_path)

def process_image(model, image, conf_threshold):
    """Procesa una imagen y devuelve la imagen con anotaciones."""
    # Realizar predicción
    results = model.predict(image, conf=conf_threshold)
    
    # Renderizar resultados (YOLO devuelve un array numpy BGR, lo convertimos a RGB para Streamlit)
    res_plotted = results[0].plot()
    res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Contar detecciones
    counts = {}
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] = counts.get(label, 0) + 1
        
    return res_plotted, counts

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/60/TWICE_LOGO.png", width=100)
    st.title("Configuración")
    st.write("Ajusta los parámetros del modelo:")
    
    conf_threshold = st.slider(
        "Umbral de Confianza", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.40, 
        step=0.05
    )
    
    st.markdown("---")
    st.info("Este proyecto detecta y clasifica Flamencos 🦩 y Pingüinos 🐧 utilizando YOLOv11.")

# --- CARGA DEL MODELO ---
# Asumimos que el modelo está en src/best.pt según tu estructura anterior
MODEL_PATH = "src/best.pt" 

# Si no existe localmente para pruebas rápidas, intenta buscar en la raíz o avisa
if not os.path.exists(MODEL_PATH) and os.path.exists("best.pt"):
    MODEL_PATH = "src/best.pt"

model = load_model(MODEL_PATH)

# --- PÁGINA PRINCIPAL ---
st.title("🦩 Detector de Fauna Antártica y Tropical")
st.markdown("### Clasificación inteligente de Flamencos y Pingüinos")

# Pestañas para organizar la vista
tab1, tab2, tab3 = st.tabs(["🏠 Inicio", "📸 Cámara en Vivo", "📂 Subir Imagen"])

# --- TAB 1: INICIO ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Bienvenido al panel de control del detector.**
        
        Este sistema utiliza Deep Learning para identificar especies en imágenes.
        
        **Características:**
        - Detección en tiempo real.
        - Alta precisión con modelos YOLO.
        - Diferenciación entre Flamencos y Pingüinos.
        """)
        st.info("👈 Usa el menú lateral para ajustar la sensibilidad del detector.")
    
    with col2:
        # Placeholder visual o imagen de ejemplo
        st.markdown(
            """
            <div style="background-color:white; padding:20px; border-radius:10px; border: 1px solid #ddd;">
                <h4 style="text-align:center;">Métricas del Modelo</h4>
                <ul>
                    <li>Modelo Base: <b>YOLO11s</b></li>
                    <li>Clases: <b>Flamingo, Penguin</b></li>
                    <li>Entorno: <b>Python + Streamlit</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

# --- TAB 2: CÁMARA (WEBCAM SNAPSHOT) ---
with tab2:
    st.header("Detección mediante Webcam")
    st.write("Toma una foto con tu cámara web para analizarla al instante.")
    
    img_file_buffer = st.camera_input("Sonríe a la cámara")

    if img_file_buffer is not None:
        # Convertir el buffer a imagen PIL
        image = Image.open(img_file_buffer)
        
        if model:
            with st.spinner("Analizando imagen..."):
                result_img, counts = process_image(model, image, conf_threshold)
                
                st.image(result_img, caption="Imagen Procesada", use_container_width=True)
                
                # Mostrar estadísticas
                if counts:
                    st.success(f"¡Detección completada! Se encontraron: {counts}")
                else:
                    st.warning("No se detectaron animales con el umbral actual.")

# --- TAB 3: SUBIR ARCHIVO ---
with tab3:
    st.header("Análisis de Archivos")
    uploaded_file = st.file_uploader("Arrastra una imagen aquí (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Mostrar original y procesada lado a lado
        col_orig, col_proc = st.columns(2)
        
        with col_orig:
            st.image(image, caption="Imagen Original", use_container_width=True)
            
        if model:
            if st.button("🔍 Detectar Animales", key="detect_upload"):
                with col_proc:
                    with st.spinner("Procesando..."):
                        result_img, counts = process_image(model, image, conf_threshold)
                        st.image(result_img, caption="Resultado del Modelo", use_container_width=True)
                        
                # Resultados en texto debajo
                st.markdown("### Resultados del análisis")
                if "Flamingo" in counts or "Flamenco" in counts:
                    st.metric("Flamencos Detectados", counts.get("Flamingo", counts.get("Flamenco", 0)), delta="Tropical 🦩")
                
                if "Penguin" in counts or "Pinguino" in counts:
                    st.metric("Pingüinos Detectados", counts.get("Penguin", counts.get("Pinguino", 0)), delta="Antártico 🐧", delta_color="inverse")