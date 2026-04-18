import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy import stats
import google.generativeai as genai

# 1. CONFIGURACIÓN DE INTERFAZ Y ESTILOS
st.set_page_config(page_title="Plataforma Estadística Z - Gemini", layout="wide")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 2px solid #636EFA;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    .stInfo {
        background-color: #0F172A;
        color: #FFFFFF !important;
        border: 1px solid #00CC96;
        border-radius: 10px;
        padding: 20px;
    }
    .stInfo p, .stInfo li, .stInfo h1, .stInfo h2, .stInfo h3 {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Plataforma de Análisis Estadístico y Pruebas Z")

# --- CONFIGURACIÓN ---
with st.container():
    expander_config = st.expander("🛠️ Configuración de API y Datos", expanded=True)
    with expander_config:
        c1, c2 = st.columns(2)
        with c1:
            api_key = st.text_input("Gemini API Key:", type="password")
        with c2:
            st.write("**Selecciona el Origen de Datos:**")
            # CAMBIO: Ahora es un selector de segmentos (botones) en lugar de radio buttons
            metodo = st.segmented_control(
                "Origen:", 
                options=["Generar Automático", "Subir Archivo CSV"],
                default="Generar Automático"
            )
            
            df = None
            if metodo == "Subir Archivo CSV":
                up = st.file_uploader("Subir CSV", type=["csv"])
                if up: df = pd.read_csv(up)
            else:
                n_size = st.slider("Muestra:", 30, 500, 100)
                if st.button("✨ Generar Datos Ahora"):
                    df = pd.DataFrame(np.random.normal(15.5, 2, n_size), columns=["Variable"])

st.divider()

if df is not None:
    col = st.selectbox("🎯 Variable a analizar:", df.columns)
    
    tab1, tab2, tab3 = st.tabs(["📈 Análisis Visual", "🧪 Prueba Z", "🤖 Informe Completo"])

    with tab1:
        st.header("1. Análisis de Distribución")
        
        # Fila superior: 2 columnas
        col_vis1, col_vis2 = st.columns(2)
        with col_vis1: 
            fig_dist = ff.create_distplot([df[col]], [col], colors=['#00CC96'])
            fig_dist.update_layout(title="Histograma y Curva de Densidad")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_vis2: 
            fig_box = px.box(df, y=col, title="Diagrama de Caja y Bigotes (Outliers)", points="all", color_discrete_sequence=['#AB63FA'])
            st.plotly_chart(fig_box, use_container_width=True)

        # NUEVA GRÁFICA: Gráfico de Dispersión Acumulada (Violin Plot) ocupando todo el ancho
        st.subheader("Distribución Detallada de la Muestra")
        fig_violin = px.violin(df, y=col, box=True, points="all", 
                               title="Gráfico de Violín y Nube de Puntos (Distribución Completa)",
                               color_discrete_sequence=['#FFA15A'])
        fig_violin.update_layout(height=500) # Se alarga para ocupar más espacio
        st.plotly_chart(fig_violin, use_container_width=True)

        # Botón de interpretación de Gemini para las TRES gráficas
        if api_key:
            if st.button("🔍 Analizar este conjunto de gráficas"):
                try:
                    genai.configure(api_key=api_key.strip())
                    model = genai.GenerativeModel('gemini-flash-latest')
                    stats_desc = df[col].describe()
                    prompt_graf = f"""Analiza estas tres gráficas (Histograma, Boxplot y Violín) con estos datos:
                    Variable: {col}, Media: {stats_desc['mean']:.2f}, Desviación: {stats_desc['std']:.2f}.
                    Explica qué nos dicen sobre la concentración de los datos y si hay anomalías visibles."""
                    with st.spinner("Gemini analizando visualmente..."):
                        res = model.generate_content(prompt_graf)
                        st.info(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.header("2. Inferencia Estadística")
        cz1, cz2 = st.columns(2)
        
        with cz1:
            mu_h0 = st.number_input("Media H0", value=15.5)
            alpha = st.select_slider("Significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)
            cola = st.selectbox("Prueba:", ["Bilateral", "Superior (Derecha)", "Inferior (Izquierda)"])
            std_pob = st.number_input("σ Poblacional", value=float(df[col].std()))

        media_m = df[col].mean()
        z_stat = (media_m - mu_h0) / (std_pob / np.sqrt(len(df[col])))
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat))) if "Bilateral" in cola else (1 - stats.norm.cdf(z_stat) if "Superior" in cola else stats.norm.cdf(z_stat))
        rechazo = p_val < alpha

        with cz2:
            st.subheader("Resultados")
            m1, m2 = st.columns(2)
            m1.metric("Z", f"{z_stat:.4f}")
            m2.metric("P-Value", f"{p_val:.4f}")
            if rechazo: st.error("### RECHAZAR H0")
            else: st.success("### NO RECHAZAR H0")

        # Gráfica de Gauss centrada y ancha
        x = np.linspace(-4, 4, 1000)
        fig_g = go.Figure(go.Scatter(x=x, y=stats.norm.pdf(x), name='Normal', fill='tozeroy', line=dict(color='#636EFA')))
        fig_g.add_vline(x=z_stat, line_color="#EF553B", line_width=4, annotation_text=f"Tu Z: {z_stat:.2f}")
        fig_g.update_layout(title="Posición en la Campana de Gauss", height=400)
        st.plotly_chart(fig_g, use_container_width=True)

    with tab3:
        st.header("3. Informe de Consultoría")
        if api_key:
            if st.button("🚀 Generar Informe Final"):
                try:
                    genai.configure(api_key=api_key.strip())
                    model = genai.GenerativeModel('gemini-flash-latest')
                    prompt = f"Explica el veredicto: Media={media_m:.2f}, H0={mu_h0}, Z={z_stat:.2f}, P-value={p_val:.4f}."
                    with st.spinner("Redactando..."):
                        response = model.generate_content(prompt)
                        st.info(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")