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
    .explanation-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid #444;
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
        
        col_vis1, col_vis2 = st.columns(2)
        with col_vis1: 
            fig_dist = ff.create_distplot([df[col]], [col], colors=['#00CC96'])
            fig_dist.update_layout(title="Histograma y Curva de Densidad")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col_vis2: 
            fig_box = px.box(df, y=col, title="Diagrama de Caja y Outliers", points="all", color_discrete_sequence=['#AB63FA'])
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Distribución Detallada de la Muestra")
        fig_violin = px.violin(df, y=col, box=True, points="all", 
                               title="Gráfico de Violín y Nube de Puntos",
                               color_discrete_sequence=['#FFA15A'])
        fig_violin.update_layout(height=500)
        st.plotly_chart(fig_violin, use_container_width=True)

        # BOTÓN RESTAURADO: Interpretación de Gráficas
        if api_key:
            if st.button("🔍 ¿Qué significan estas gráficas?"):
                try:
                    genai.configure(api_key=api_key.strip())
                    model = genai.GenerativeModel('gemini-flash-latest') # MODELO FIJO
                    stats_desc = df[col].describe()
                    
                    prompt_graf = f"""Actúa como un profesor que explica de forma muy simple a alguien que NO sabe nada de probabilidad.
                    Analiza estas gráficas basándote en: Media={stats_desc['mean']:.2f}, Mínimo={stats_desc['min']:.2f}, Máximo={stats_desc['max']:.2f}.
                    Explica:
                    1. Si los datos están muy amontonados o dispersos.
                    2. Si hay valores "extraños" o muy alejados del resto (outliers).
                    3. Qué nos dice la forma de la campana sobre la mayoría de la gente/objetos medidos.
                    No uses términos técnicos como 'curtosis' o 'desviación estándar'. Usa ejemplos claros."""
                    
                    with st.spinner("Gemini está traduciendo las gráficas a lenguaje sencillo..."):
                        res = model.generate_content(prompt_graf)
                        st.info(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.header("2. Inferencia Estadística")
        cz1, cz2 = st.columns(2)
        
        with cz1:
            mu_h0 = st.number_input("Valor esperado (Media H0)", value=15.50)
            alpha = st.select_slider("Nivel de Error permitido (α)", options=[0.01, 0.05, 0.10], value=0.05)
            cola = st.selectbox("Tipo de comparación:", ["Bilateral", "Superior (Derecha)", "Inferior (Izquierda)"])
            std_pob = st.number_input("Variación conocida (σ)", value=float(df[col].std()))

        media_m = df[col].mean()
        n = len(df[col])
        z_stat = (media_m - mu_h0) / (std_pob / np.sqrt(n))
        
        if "Bilateral" in cola:
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif "Superior" in cola:
            p_val = 1 - stats.norm.cdf(z_stat)
        else:
            p_val = stats.norm.cdf(z_stat)
            
        rechazo = p_val < alpha

        with cz2:
            st.subheader("Resultados")
            m1, m2 = st.columns(2)
            m1.metric("Puntuación Z", f"{z_stat:.4f}")
            m2.metric("Probabilidad (P)", f"{p_val:.4f}")

            if rechazo:
                st.error("### DECISIÓN: HAY UN CAMBIO IMPORTANTE")
                st.markdown(f"""<div class="explanation-box" style="background-color: #441111;">
                <b>¿Por qué?</b> La probabilidad de que esto sea azar es de solo {p_val*100:.2f}%, lo cual es menor al {alpha*100:.0f}% que permitimos. 
                Podemos decir que el resultado NO es normal y algo ha cambiado.</div>""", unsafe_allow_html=True)
            else:
                st.success("### DECISIÓN: TODO SIGUE IGUAL")
                st.markdown(f"""<div class="explanation-box" style="background-color: #113311;">
                <b>¿Por qué?</b> La probabilidad de que esto sea casualidad es del {p_val*100:.2f}%. Como es mayor a nuestro límite ({alpha*100:.0f}%), 
                no tenemos pruebas suficientes para decir que algo es diferente.</div>""", unsafe_allow_html=True)

        st.subheader("Visualización en la Campana")
        x = np.linspace(-4, 4, 1000)
        fig_g = go.Figure(go.Scatter(x=x, y=stats.norm.pdf(x), name='Normal', fill='tozeroy', line=dict(color='#636EFA')))
        
        # Zonas de rechazo
        if "Bilateral" in cola:
            z_c = stats.norm.ppf(1 - alpha/2)
            fig_g.add_vrect(x0=z_c, x1=4, fillcolor="red", opacity=0.2, annotation_text="Zona de Cambio")
            fig_g.add_vrect(x0=-4, x1=-z_c, fillcolor="red", opacity=0.2)
        elif "Superior" in cola:
            z_c = stats.norm.ppf(1 - alpha)
            fig_g.add_vrect(x0=z_c, x1=4, fillcolor="red", opacity=0.2, annotation_text="Zona de Cambio")
        else:
            z_c = stats.norm.ppf(alpha)
            fig_g.add_vrect(x0=-4, x1=z_c, fillcolor="red", opacity=0.2, annotation_text="Zona de Cambio")

        fig_g.add_vline(x=z_stat, line_color="#EF553B", line_width=4, annotation_text=f"Tu Dato: {z_stat:.2f}")
        st.plotly_chart(fig_g, use_container_width=True)

    with tab3:
        st.header("3. Informe de Consultoría con Gemini")
        if api_key:
            if st.button("🚀 Generar Resumen"):
                try:
                    genai.configure(api_key=api_key.strip())
                    model = genai.GenerativeModel('gemini-flash-latest') # MODELO FIJO
                    
                    prompt = f"""Actúa como un consultor experto hablando con un cliente que NO sabe estadística.
                    Resultados: Promedio de la muestra={media_m:.2f}, Valor esperado={mu_h0}, Probabilidad P={p_val:.4f}.
                    Explica de forma muy humana:
                    1. ¿Qué encontramos en los datos? 
                    2. ¿Deberíamos preocuparnos o tomar alguna acción?
                    3. Usa una analogía sencilla (como el clima o deportes) para explicar por qué aceptamos o rechazamos la idea inicial.
                    Evita términos como 'hipótesis nula', 'distribución' o 'Z-score'."""
                    
                    with st.spinner("Gemini está redactando el informe final..."):
                        response = model.generate_content(prompt)
                        st.info(response.text)
                except Exception as e:
                    st.error(f"Falla en la API: {e}")