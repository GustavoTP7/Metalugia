import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Hub", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return None

st.title("🏭 Centro de Control Metalúrgico")

# --- BARRA LATERAL (Panel de Mandos) ---
with st.sidebar:
    st.header("1️⃣ Carga y Filtros")
    archivo = st.file_uploader("Subir dataset", type=["csv", "xlsx"])
    factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5)
    
    st.divider()
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.header("2️⃣ Configuración")
            target = st.selectbox("🎯 Variable Objetivo (Y)", num_cols)
            features = st.multiselect("🔍 Variables de Entrada (X)", [c for c in num_cols if c != target])
            
            st.divider()
            btn_entrenar = st.button("🚀 ENTRENAR MODELO", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO ---
if archivo and btn_entrenar:
    if not features:
        st.error("⚠️ Selecciona variables X en la barra lateral.")
    else:
        with st.spinner('Entrenando...'):
            # Detección de Outliers
            indices_out = set()
            for col in [target] + features:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df[(df[col] < q1 - factor_iqr*iqr) | (df[col] > q3 + factor_iqr*iqr)].index)
            
            df_orig = df[[target] + features].dropna()
            df_sin = df_orig.drop(list(indices_out), errors='ignore')

            def procesar_motor(data):
                X, y = data[features], data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, tree_method='hist', random_state=42)
                
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                return {
                    "cv": np.mean(cv), "test": r2_score(y_test, pred), "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                    "model": model, "df_test": pd.DataFrame({'Real': y_test, 'Pred': pred}),
                    "imp": pd.DataFrame({'Var': features, 'Imp': model.feature_importances_})
                }

            st.session_state['res_o'] = procesar_motor(df_orig)
            st.session_state['res_s'] = procesar_motor(df_sin)
            st.session_state['features_list'] = features
            st.session_state['target_name'] = target
            st.session_state['df_limpio'] = df_sin

# --- VISUALIZACIÓN PRINCIPAL ---
if 'res_o' in st.session_state:
    # Métricas de Cabecera
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estabilidad (CV)", f"{st.session_state['res_s']['cv']:.4f}")
    c2.metric("Precisión (Test)", f"{st.session_state['res_s']['test']:.4f}")
    c3.metric("Error (RMSE)", f"{st.session_state['res_s']['rmse']:.3f}")
    c4.metric("Filas Auditadas", len(st.session_state['df_limpio']))

    tab_diag, tab_sim, tab_data = st.tabs(["📊 Diagnóstico", "🎯 Simulador", "👁️ Datos"])

    with tab_diag:
        col_left, col_right = st.columns(2)
        with col_left:
            fig_fit = px.scatter(st.session_state['res_s']['df_test'], x='Real', y='Pred', trendline="ols", 
                                 title="Ajuste: Real vs Predicho", color_discrete_sequence=['#00FF00'])
            st.plotly_chart(fig_fit, use_container_width=True)
        with col_right:
            imp_df = st.session_state['res_s']['imp'].sort_values('Imp', ascending=True)
            fig_imp = px.bar(imp_df, x='Imp', y='Var', orientation='h', title="Importancia de Variables")
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab_sim:
        st.subheader("Simulador en Tiempo Real")
        c_in, c_res = st.columns([1, 1])
        inputs = {}
        with c_in:
            for f in st.session_state['features_list']:
                val_min = float(st.session_state['df_limpio'][f].min())
                val_max = float(st.session_state['df_limpio'][f].max())
                val_avg = float(st.session_state['df_limpio'][f].mean())
                inputs[f] = st.slider(f, val_min, val_max, val_avg)
        
        with c_res:
            pred_val = st.session_state['res_s']['model'].predict(pd.DataFrame([inputs]))[0]
            st.markdown(f"""
                <div style="background-color:#0E1117; padding:40px; border-radius:15px; border: 2px solid #00FF00; text-align:center">
                    <h2 style="color:white; margin-bottom:0px">PREDICCIÓN</h2>
                    <h1 style="color:#00FF00; font-size:70px; margin-top:10px">{pred_val:.3f}</h1>
                    <p style="color:#888">{st.session_state['target_name']}</p>
                </div>
            """, unsafe_allow_html=True)

    with tab_data:
        st.dataframe(st.session_state['df_limpio'], use_container_width=True)

elif archivo:
    st.info("👈 Configura las variables y presiona el botón 'ENTRENAR MODELO' en la barra lateral.")
else:
    st.info("👋 Bienvenida/o. Por favor, sube un archivo CSV o Excel para comenzar.")
