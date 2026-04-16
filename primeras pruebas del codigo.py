import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="Asistente Estadístico IA", layout="wide")

st.title("📊 Plataforma de Análisis Estadístico y Pruebas Z")
st.markdown("""
Esta aplicación permite documentar el proceso creativo y las limitaciones de la IA en el desarrollo de software. [cite: 1]
""")

# --- MÓDULO 1: CARGA DE DATOS [cite: 7] ---
st.sidebar.header("Configuración de Datos")
metodo_carga = st.sidebar.radio("Selecciona método:", ["Cargar CSV", "Generar Datos Sintéticos"])

df = None

if metodo_carga == "Cargar CSV":
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    n_sintetico = st.sidebar.slider("Tamaño de muestra (n >= 30)", 30, 500, 100)
    media_sintetica = st.sidebar.number_input("Media real", value=50.0)
    st_dev_sintetica = st.sidebar.number_input("Desviación estándar real", value=10.0)
    
    if st.sidebar.button("Generar Datos"):
        data = np.random.normal(media_sintetica, st_dev_sintetica, n_sintetico)
        df = pd.DataFrame(data, columns=["Variable_Sintetica"])

# --- MÓDULO 2: VISUALIZACIÓN [cite: 10] ---
if df is not None:
    st.header("1. Visualización de Distribuciones")
    columna = st.selectbox("Selecciona la variable a analizar:", df.columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma y KDE [cite: 11, 12]
        fig_hist = ff.create_distplot([df[columna]], [columna], bin_size=.5)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        # Boxplot [cite: 13]
        fig_box = px.box(df, y=columna, title=f"Boxplot de {columna}")
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Cuestionario de Interpretación [cite: 14] ---
    st.subheader("Interpretación del Estudiante")
    normalidad = st.radio("¿La distribución parece normal?", ["Sí", "No", "Incierto"])
    sesgo = st.text_area("¿Hay sesgo o outliers detectados?")
    
    st.info("Guarda estas respuestas para compararlas con la IA más adelante.")

else:
    st.warning("Por favor, carga o genera datos para comenzar.")