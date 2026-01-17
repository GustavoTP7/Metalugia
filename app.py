import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
            features_seleccionadas = st.multiselect("🔍 Variables de Entrada (X)", [c for c in cols_todas if c not in [target, col_id]])
            
            st.divider()
            btn_entrenar = st.button("🚀 ENTRENAR Y AUDITAR", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO ---
if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features_seleccionadas:
        st.error("⚠️ Selecciona variables X.")
    else:
        with st.spinner('Procesando inteligencia de planta...'):
            # MEJORA: Ordenamiento dinámico para flexibilidad total
            features_modelo = sorted(features_seleccionadas)
            
            df_base = df[[col_id, target] + features_modelo].dropna().copy()
            
            # Auditoría de Outliers
            if modo_datos == "Sin Outliers (Auditado)":
                indices_out = set()
                for col in [target] + features_modelo:
                    q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                    iqr = q3 - q1
                    indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
                df_final = df_base.drop(list(indices_out), errors='ignore')
                etiqueta = "AUDITADO (SIN OUTLIERS)"
            else:
                df_final = df_base
                etiqueta = "ORIGINAL (CON RUIDO)"

            X = df_final[features_modelo]
            y = df_final[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # --- NUEVAS MÉTRICAS: Bias y MAE ---
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            bias = np.mean(preds - y_test)

            # Session State
            st.session_state['res'] = {
                'test': r2_score(y_test, preds), 'rmse': rmse, 'mae': mae, 'bias': bias,
                'model': model, 'target': target, 'features': features_modelo,
                'df_work': df_final, 'modo': etiqueta, 'col_id': col_id,
                'df_audit': pd.DataFrame({
                    'ID_Turno': df_final.loc[y_test.index, col_id],
                    'Real': y_test.values, 'Pred': preds, 
                    'Error': np.abs(y_test.values - preds)
                }).sort_values('Error', ascending=False)
            }

# --- VISUALIZACIÓN ---
if 'res' in st.session_state:
    res = st.session_state['res']
    
    # 4 Métricas principales
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión (R²)", f"{res['test']:.4f}")
    m2.metric("Riesgo (RMSE)", f"{res['rmse']:.3f}")
    m3.metric("Sesgo (Bias)", f"{res['bias']:.3f}", delta_color="inverse")
    m4.metric("Modo Datos", res['modo'])

    tabs = st.tabs(["📊 Diagnóstico & Sensibilidad", "🚩 Auditoría 360°", "🎯 Simulador What-If", "👁️ Datos"])

    with tabs[0]:
        c_graf1, c_graf2 = st.columns([1.5, 1])
        with c_graf1:
            eje_x = st.selectbox("Analizar contra Variable:", ["Real"] + res['features'])
            if eje_x == "Real":
                fig = px.scatter(res['df_audit'], x='Real', y='Pred', hover_data=['ID_Turno'], trendline="ols", title="Ajuste General")
            else:
                fig = px.scatter(res['df_work'], x=eje_x, y=res['target'], trendline="ols", title=f"Impacto de {eje_x}")
            st.plotly_chart(fig, use_container_width=True)
        
        with c_graf2:
            # MEJORA: Gráfico de Sensibilidad (Importancia)
            st.subheader("Sensibilidad del Modelo")
            imp_df = pd.DataFrame({'Var': res['features'], 'Imp': res['model'].feature_importances_}).sort_values('Imp')
            fig_imp = px.bar(imp_df, x='Imp', y='Var', orientation='h', color='Imp', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)

    with tabs[1]:
        st.subheader("🚩 Radar de Desviaciones e Inconsistencias")
        df_audit = res['df_audit'].copy()
        
        # MEJORA: Semáforo de Alerta Operativa
        umbral = res['rmse'] * 1.5
        df_audit['Estado'] = df_audit['Error'].apply(lambda x: "🚩 ANOMALÍA" if x > umbral else "✅ NORMAL")
        
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.dataframe(df_audit.style.highlight_max(subset=['Error'], color='#ffcccc'), use_container_width=True)
        with col_t2:
            st.info(f"Se consideran anomalías turnos con error > {umbral:.2f}")
            csv = df_audit.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Reporte para Planta", csv, "auditoria.csv", "text/csv")

    with tabs[2]:
        st.subheader("Simulador What-If (Predicción en Tiempo Real)")
        cin, cout = st.columns([1, 1])
        inputs = {}
        with cin:
            for f in res['features']:
                v_min, v_max = float(res['df_work'][f].min()), float(res['df_work'][f].max())
                inputs[f] = st.slider(f"{f}", v_min, v_max, float(res['df_work'][f].mean()))
        
        with cout:
            # Blindaje de orden alfabético en el input
            df_input = pd.DataFrame([inputs])[res['features']]
            pred_val = res['model'].predict(df_input)[0]
            
            st.markdown(f"""
                <div style='background-color:#0E1117; padding:40px; border-radius:15px; border: 2px solid #00FF00; text-align:center'>
                    <h3 style='color:white'>PREDICCIÓN ESTIMADA DE {res['target'].upper()}</h3>
                    <h1 style='color:#00FF00; font-size:65px'>{pred_val:.3f}</h1>
                    <p style='color:gray'>Margen de error esperado: ±{res['rmse']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

    with tabs[3]:
        st.dataframe(res['df_work'], use_container_width=True)

elif archivo:
    st.info("👈 Selecciona tus variables y presiona el botón para generar el Hub.")
