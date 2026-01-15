import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Control Hub Pro", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return None

st.title("🏭 Centro de Control Metalúrgico: Auditoría & IA")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1️⃣ Gestión de Datos")
    archivo = st.file_uploader("Subir dataset", type=["csv", "xlsx"])
    
    modo_datos = st.radio("Selecciona modo de datos:", 
                          ["Dataset Original", "Sin Outliers (Auditado)"],
                          help="El modo 'Sin Outliers' elimina ruido usando el método IQR.")
    
    factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5, 
                           disabled=(modo_datos == "Dataset Original"))
    
    st.divider()
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            cols_todas = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.header("2️⃣ Configuración")
            col_id = st.selectbox("🔑 ID de Rastreo (Fecha_Turno):", cols_todas)
            target = st.selectbox("🎯 Variable Objetivo (Y)", num_cols)
            features = st.multiselect("🔍 Variables de Entrada (X)", [c for c in cols_todas if c not in [target, col_id]])
            
            st.divider()
            btn_entrenar = st.button("🚀 ENTRENAR Y AUDITAR", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO ---
if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features:
        st.error("⚠️ Selecciona variables X.")
    else:
        with st.spinner('Procesando inteligencia de planta...'):
            df_base = df[[col_id, target] + features].dropna().copy()
            
            # Limpieza de Outliers
            if modo_datos == "Sin Outliers (Auditado)":
                indices_out = set()
                for col in [target] + [f for f in features if df_base[f].dtype in [np.float64, np.int64]]:
                    q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                    iqr = q3 - q1
                    indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
                df_final = df_base.drop(list(indices_out), errors='ignore')
                etiqueta = "AUDITADO (SIN OUTLIERS)"
            else:
                df_final = df_base
                etiqueta = "ORIGINAL (CON RUIDO)"

            # Encoding e IA
            X_encoded = df_final[features].copy()
            mapeos = {}
            for col in features:
                if X_encoded[col].dtype == 'object':
                    cats = sorted(X_encoded[col].unique())
                    mapeos[col] = cats
                    X_encoded[col] = X_encoded[col].map({v: i for i, v in enumerate(cats)})

            X_train, X_test, y_train, y_test = train_test_split(X_encoded, df_final[target], test_size=0.3, random_state=42)
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Session State
            st.session_state['res'] = {
                'test': r2_score(y_test, preds), 'rmse': np.sqrt(mean_squared_error(y_test, preds)),
                'model': model, 'target': target, 'features': features, 'mapeos': mapeos,
                'df_work': df_final, 'modo': etiqueta, 'col_id': col_id,
                'df_audit': pd.DataFrame({
                    'ID_Turno': df_final.loc[y_test.index, col_id],
                    'Real': y_test.values, 'Pred': preds, 'Error': np.abs(y_test.values - preds)
                }).sort_values('Error', ascending=False)
            }

# --- VISUALIZACIÓN ---
if 'res' in st.session_state:
    # Métricas de Cabecera
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión (R²)", f"{st.session_state['res']['test']:.4f}")
    m2.metric("Error (RMSE)", f"{st.session_state['res']['rmse']:.3f}")
    m3.metric("Modo", st.session_state['res']['modo'])
    m4.metric("Datos Usados", len(st.session_state['res']['df_work']))

    tabs = st.tabs(["📊 Diagnóstico Ejes", "🚩 Auditoría 360°", "🎯 Simulador Pro", "👁️ Datos & Histogramas"])

    with tabs[0]:
        cx1, cx2 = st.columns([1, 2])
        with cx1:
            eje_x = st.selectbox("Eje X:", ["Real"] + st.session_state['res']['features'])
            eje_y = st.selectbox("Eje Y:", ["Pred", "Real"])
        with cx2:
            if eje_x in st.session_state['res']['features']:
                fig_sc = px.scatter(st.session_state['res']['df_work'], x=eje_x, y=st.session_state['res']['target'], 
                                   hover_data=[st.session_state['res']['col_id']], trendline="ols", title=f"Relación: {eje_x}")
            else:
                fig_sc = px.scatter(st.session_state['res']['df_audit'], x='Real', y='Pred', 
                                   hover_data=['ID_Turno'], trendline="ols", title="Ajuste Real vs Predicho")
            st.plotly_chart(fig_sc, use_container_width=True)

    with tabs[1]:
        st.subheader("🚩 Radar de Desviaciones Críticas")
        df_audit = st.session_state['res']['df_audit'].copy()
        df_audit = df_audit.merge(st.session_state['res']['df_work'], left_on='ID_Turno', right_on=st.session_state['res']['col_id'], how='left')
        df_audit['Desviación_%'] = (df_audit['Error'] / df_audit['Real']) * 100

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("### 🏆 Top 10 Turnos con Mayor Desviación")
            st.table(df_audit[['ID_Turno', 'Real', 'Pred', 'Error', 'Desviación_%']].head(10))
        with col_a2:
            st.markdown("### 🔍 Error por Rango de Operación")
            fig_err_range = px.scatter(df_audit, x='Real', y='Error', size='Error', color='Error', title="Magnitud del Error vs Recuperación Real")
            st.plotly_chart(fig_err_range, use_container_width=True)

        st.divider()
        col_a3, col_a4 = st.columns(2)
        with col_a3:
            st.markdown("### 🌡️ ¿Qué variable causa el error?")
            var_analisis = st.selectbox("Analizar error contra:", st.session_state['res']['features'])
            fig_correl = px.scatter(df_audit, x=var_analisis, y='Error', trendline="ols", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_correl, use_container_width=True)
        with col_a4:
            st.markdown("### 📋 Auditoría Completa")
            st.dataframe(df_audit[['ID_Turno', 'Real', 'Pred', 'Error', 'Desviación_%']], use_container_width=True)

    with tabs[2]:
        st.subheader("Simulador What-If")
        cin, cout = st.columns(2)
        inputs = {}
        with cin:
            for f in st.session_state['res']['features']:
                if f in st.session_state['res']['mapeos']:
                    opc = st.session_state['res']['mapeos'][f]
                    sel = st.selectbox(f"Seleccionar {f}", opc)
                    inputs[f] = opc.index(sel)
                else:
                    v_min, v_max = float(st.session_state['res']['df_work'][f].min()), float(st.session_state['res']['df_work'][f].max())
                    inputs[f] = st.slider(f, v_min, v_max, float(st.session_state['res']['df_work'][f].mean()))
        with cout:
            pred_val = st.session_state['res']['model'].predict(pd.DataFrame([inputs]))[0]
            st.markdown(f"<div style='background-color:#0E1117; padding:40px; border-radius:15px; border: 2px solid #00FF00; text-align:center'><h2 style='color:white'>PREDICCIÓN</h2><h1 style='color:#00FF00; font-size:70px'>{pred_val:.3f}</h1></div>", unsafe_allow_html=True)

    with tabs[3]:
        ct1, ct2 = st.columns(2)
        with ct1:
            st.subheader("Visor de Datos")
            st.dataframe(st.session_state['res']['df_work'], use_container_width=True)
        with ct2:
            st.subheader("Histogramas de Control")
            v_h = st.selectbox("Variable:", [st.session_state['res']['target']] + st.session_state['res']['features'])
            fig_h = px.histogram(st.session_state['res']['df_work'], x=v_h, nbins=35, marginal="box", labels={"count": "Número de Mediciones"})
            st.plotly_chart(fig_h, use_container_width=True)

elif archivo:
    st.info("👈 Configura la columna ID y las variables, luego entrena.")
