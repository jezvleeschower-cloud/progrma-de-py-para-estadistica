import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import google.generativeai as genai

# 1. CONFIGURACIÓN DE INTERFAZ
st.set_page_config(page_title="Plataforma Estadística Z - Gemini", layout="wide")

st.title("📊 Plataforma de Análisis Estadístico y Pruebas Z")
st.markdown("Análisis automatizado con integración oficial de **Gemini 1.5 Flash**.")

# --- SECCIÓN DE CONFIGURACIÓN (Panel Principal) ---
st.header("⚙️ Configuración del Análisis")
expander_config = st.expander("Paso 1: Configurar API Gemini y Datos", expanded=True)

with expander_config:
    col_setup1, col_setup2 = st.columns(2)
    
    with col_setup1:
        st.subheader("Conexión con Gemini")
        api_key = st.text_input("Gemini API Key:", type="password", help="Pega aquí tu clave de Google AI Studio.")
    
    with col_setup2:
        st.subheader("Gestión de Datos")
        metodo = st.radio("Origen de datos:", ["Generar Automático", "Subir Archivo CSV"], horizontal=True)
        
        df = None
        if metodo == "Subir Archivo CSV":
            up = st.file_uploader("Archivo CSV", type=["csv"])
            if up: df = pd.read_csv(up)
        else:
            n_size = st.slider("Muestra:", 30, 500, 100)
            if st.button("Generar Datos"):
                df = pd.DataFrame(np.random.normal(50, 10, n_size), columns=["Variable"])

st.divider()

# --- PROCESAMIENTO ESTADÍSTICO ---
if df is not None:
    col = st.selectbox("Selecciona la columna para el análisis:", df.columns)
    
    # 2. NORMALIDAD AUTOMÁTICA
    st.header("1. Análisis de Distribución y Normalidad")
    stat_sh, p_norm = stats.shapiro(df[col])
    es_normal = p_norm > 0.05
    
    if es_normal:
        st.success(f"✅ Validación Automática: Los datos siguen una distribución normal (p-value: {p_norm:.4f}).")
    else:
        st.warning(f"⚠️ Validación Automática: Los datos NO parecen normales (p-value: {p_norm:.4f}).")

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1: 
        st.plotly_chart(ff.create_distplot([df[col]], [col]), use_container_width=True)
    with col_vis2: 
        st.plotly_chart(px.box(df, y=col, title="Diagrama de Caja (Outliers)"), use_container_width=True)

    # 3. PRUEBA Z
    st.header("2. Inferencia Estadística (Prueba Z)")
    cz1, cz2 = st.columns(2)
    
    with cz1:
        st.subheader("Parámetros de la Prueba")
        mu_h0 = st.number_input("Hipótesis Nula (Media H0)", value=float(df[col].mean()))
        alpha = st.slider("Nivel de Significancia (α)", 0.01, 0.10, 0.05)
        cola = st.selectbox("Tipo de Prueba:", ["Bilateral", "Superior (Derecha)", "Inferior (Izquierda)"])
        std_pob = st.number_input("Desviación Estándar Poblacional (σ)", value=float(df[col].std()))

    # Cálculos
    n_count = len(df[col])
    media_m = df[col].mean()
    z_stat = (media_m - mu_h0) / (std_pob / np.sqrt(n_count))
    
    if "Bilateral" in cola:
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        rechazo = p_val < alpha
    elif "Superior" in cola:
        p_val = 1 - stats.norm.cdf(z_stat)
        rechazo = p_val < alpha
    else:
        p_val = stats.norm.cdf(z_stat)
        rechazo = p_val < alpha

    with cz2:
        st.subheader("Resultados")
        st.metric("Estadístico Z", f"{z_stat:.4f}")
        st.metric("P-Value", f"{p_val:.4f}")
        if rechazo: st.error("Decisión: RECHAZAR Hipótesis Nula (H0)")
        else: st.success("Decisión: NO RECHAZAR Hipótesis Nula (H0)")

    # --- RESTAURACIÓN: VISUALIZACIÓN DE LA REGIÓN CRÍTICA ---
    st.subheader("Visualización de la Región Crítica")
    x_ax = np.linspace(-4, 4, 1000)
    y_ax = stats.norm.pdf(x_ax)
    
    fig_gauss = go.Figure()
    fig_gauss.add_trace(go.Scatter(x=x_ax, y=y_ax, mode='lines', name='Distribución Normal'))
    
    # Línea del Estadístico Z del usuario
    fig_gauss.add_vline(x=z_stat, line_color="green", line_dash="dash", 
                        annotation_text=f"Tu Z: {z_stat:.2f}", annotation_position="top left")
    
    st.plotly_chart(fig_gauss, use_container_width=True)

    # 4. CONSULTORÍA CON GEMINI 1.5 FLASH
    st.header("3. Interpretación con Gemini 1.5 Flash")
    if api_key:
        if st.button("🚀 Consultar con Gemini"):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-flash-latest')
                
                prompt = f"""Actúa como experto en estadística. 
                Analiza estos resultados: Media Muestral={media_m:.2f}, H0={mu_h0}, Z={z_stat:.2f}, P-value={p_val:.4f}, Normalidad={es_normal}. 
                ¿Es el resultado estadísticamente significativo y qué significa para el estudio?"""
                
                with st.spinner("Gemini está analizando..."):
                    response = model.generate_content(prompt)
                    if response.text:
                        st.info(response.text)
                    else:
                        st.warning("La IA devolvió una respuesta vacía.")
            except Exception as e:
                st.error(f"Error de conexión: {e}")
    else:
        st.info("⚠️ Ingresa tu API Key arriba para activar la IA.")
else:
    st.info("Configura los datos en la sección superior para comenzar.")