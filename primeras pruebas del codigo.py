import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import google.generativeai as genai  # Librería de Google

# 1. CONFIGURACIÓN DE INTERFAZ
st.set_page_config(page_title="Asistente Estadístico Z - Gemini", layout="wide")

st.title("📊 Plataforma de Análisis Estadístico y Pruebas Z")
st.markdown("Herramienta para documentar procesos creativos y manejo de errores de IA en software.")

# 2. CONFIGURACIÓN DE IA (Barra lateral)
st.sidebar.header("Conexión con Google AI")
api_key = st.sidebar.text_input("Gemini API Key:", type="password")

# 3. CARGA DE DATOS
st.sidebar.header("Gestión de Datos")
metodo = st.sidebar.radio("Origen de datos:", ["Generar Automático", "Subir Archivo CSV"])

df = None
if metodo == "Subir Archivo CSV":
    up = st.sidebar.file_uploader("Archivo CSV", type=["csv"])
    if up: df = pd.read_csv(up)
else:
    n = st.sidebar.slider("Tamaño de muestra", 30, 500, 100)
    if st.sidebar.button("Generar Datos"):
        df = pd.DataFrame(np.random.normal(50, 10, n), columns=["Variable"])

# --- LÓGICA DE PROCESAMIENTO ---
if df is not None:
    col = st.selectbox("Selecciona variable:", df.columns)
    
    # 4. GRÁFICAS DE DISTRIBUCIÓN
    st.header("1. Análisis Visual")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ff.create_distplot([df[col]], [col]), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df, y=col, title="Boxplot"), use_container_width=True)

    # Inputs del estudiante
    st.subheader("Documentación del Estudiante")
    normal = st.radio("¿Sigue una distribución normal?", ["Sí", "No"])
    comentarios = st.text_area("Notas sobre sesgo u outliers:")

    # 5. CÁLCULOS DE PRUEBA Z
    st.header("2. Inferencia Estadística (Prueba Z)")
    c_z1, c_z2 = st.columns(2)
    
    with c_z1:
        mu_h0 = st.number_input("H0 (Media)", value=float(df[col].mean()))
        alpha = st.number_input("Significancia (α)", 0.01, 0.10, 0.05)
        cola = st.selectbox("Tipo:", ["Bilateral", "Superior", "Inferior"])
        std_pob = st.number_input("Sigma (σ)", value=float(df[col].std()))

    n_size = len(df[col])
    media_m = df[col].mean()
    z_val = (media_m - mu_h0) / (std_pob / np.sqrt(n_size))
    
    if cola == "Bilateral":
        p = 2 * (1 - stats.norm.cdf(abs(z_val)))
        z_c = stats.norm.ppf(1 - alpha/2)
        r = abs(z_val) > z_c
    elif cola == "Superior":
        p = 1 - stats.norm.cdf(z_val)
        z_c = stats.norm.ppf(1 - alpha)
        r = z_val > z_c
    else:
        p = stats.norm.cdf(z_val)
        z_c = stats.norm.ppf(alpha)
        r = z_val < z_c

    with c_z2:
        st.metric("Estadístico Z", f"{z_val:.4f}")
        st.metric("P-Value", f"{p:.4f}")
        if r: st.error("Decisión: Rechazar H0")
        else: st.success("Decisión: No rechazar H0")

    # Curva de Gauss
    x_axis = np.linspace(-4, 4, 1000)
    fig_gauss = go.Figure()
    fig_gauss.add_trace(go.Scatter(x=x_axis, y=stats.norm.pdf(x_axis), mode='lines', name='Normal'))
    fig_gauss.add_vline(x=z_val, line_color="green", annotation_text="Tu Z")
    st.plotly_chart(fig_gauss, use_container_width=True)

    # 6. ASISTENTE IA (IMPLEMENTACIÓN CON GEMINI)
    if api_key:
        st.header("3. Consultoría con Gemini")
        if st.button("Obtener Análisis de Gemini"):
            try:
                # Configurar Gemini
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Construcción del prompt
                prompt = (
                    f"Actúa como un experto en estadística. Analiza los siguientes resultados: "
                    f"Media Muestral: {media_m:.2f}, Hipótesis Nula (H0): {mu_h0}, "
                    f"Estadístico Z: {z_val:.2f}, P-Value: {p:.4f}, Nivel de significancia: {alpha}. "
                    f"¿Es el resultado estadísticamente significativo? Explica brevemente por qué "
                    f"y qué significa para el problema."
                )
                
                with st.spinner("Gemini está analizando los datos..."):
                    response = model.generate_content(prompt)
                
                if response.text:
                    st.info("### Análisis de la IA:")
                    st.markdown(response.text)
                else:
                    st.warning("Gemini no pudo generar una respuesta clara.")
                    
            except Exception as e:
                st.error(f"Falla técnica con la API de Google: {e}")
    else:
        st.info("Ingresa tu Gemini API Key en la barra lateral para activar la IA.")