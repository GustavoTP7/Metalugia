import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Hub Pro", layout="wide")

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
    
    # NUEVA OPCIÓN: Selección de modo de datos
    modo_datos = st.radio("Selecciona modo de datos:", 
                          ["Dataset Original", "Sin Outliers (Auditado)"],
                          help="El modo 'Sin Outliers' eliminará filas ruidosas basadas en el factor IQR.")
    
    factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5, 
                           disabled=(modo_datos == "Dataset Original"))
    
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
        with st.spinner('Entrenando y validando...'):
            # Preparar base
            df_base = df[[target] + features].dropna()
            
            # Lógica de Filtrado según el modo elegido
            if modo_datos == "Sin Outliers (Auditado)":
                indices_out = set()
                for col in [target] + features:
                    q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                    iqr = q3 - q1
                    indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
                df_final = df_base.drop(list(indices_out), errors='ignore')
                etiqueta = "AUDITADO (SIN OUTLIERS)"
            else:
                df_final = df_base
                etiqueta = "ORIGINAL (CON TODO)"

            # Motor de entrenamiento
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

            # Guardamos el resultado del entrenamiento
            st.session_state['res_actual'] = procesar_motor(df_final)
            st.session_state['features_list'] = features
            st.session_state['target_name'] = target
            st.session_state['df_usado'] = df_final
            st.session_state['modo_usado'] = etiqueta

# --- VISUALIZACIÓN PRINCIPAL ---
if 'res_actual' in st.session_state:
    st.info(f"📍 Mostrando resultados del modelo: **{st.session_state['modo_usado']}**")
    
    # Métricas de Cabecera
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estabilidad (CV)", f"{st.session_state['res_actual']['cv']:.4f}")
    c2.metric("Precisión (Test)", f"{st.session_state['res_actual']['test']:.4f}")
    c3.metric("Error (RMSE)", f"{st.session_state['res_actual']['rmse']:.3f}")
    c4.metric("Filas en Modelo", len(st.session_state['df_usado']))

    tab_diag, tab_sim, tab_data = st.tabs(["📊 Diagnóstico Ejes", "🎯 Simulador", "👁️ Visor & Histogramas"])

    with tab_diag:
        col_diag1, col_diag2 = st.columns([1, 2])
        with col_diag1:
            eje_x_diag = st.selectbox("Eje X (Dispersión):", ["Real"] + st.session_state['features_list'])
            eje_y_diag = st.selectbox("Eje Y (Dispersión):", ["Pred", "Real"])
        
        with col_diag2:
            if eje_x_diag in st.session_state['features_list']:
                fig_scatter = px.scatter(st.session_state['df_usado'], x=eje_x_diag, y=st.session_state['target_name'], 
                                        trendline="ols", title=f"Relación: {eje_x_diag} vs {st.session_state['target_name']}")
            else:
                fig_scatter = px.scatter(st.session_state['res_actual']['df_test'], x=eje_x_diag, y=eje_y_diag, 
                                        trendline="ols", title="Gráfico de Ajuste")
            st.plotly_chart(fig_scatter, use_container_width=True)

        imp_df = st.session_state['res_actual']['imp'].sort_values('Imp', ascending=True)
        st.plotly_chart(px.bar(imp_df, x='Imp', y='Var', orientation='h', title="Peso de las Variables", color='Imp'), use_container_width=True)

    with tab_sim:
        st.subheader("Simulador en Tiempo Real")
        c_in, c_res = st.columns([1, 1])
        inputs = {}
        with c_in:
            for f in st.session_state['features_list']:
                val_min = float(st.session_state['df_usado'][f].min())
                val_max = float(st.session_state['df_usado'][f].max())
                val_avg = float(st.session_state['df_usado'][f].mean())
                inputs[f] = st.slider(f"Ajustar {f}", val_min, val_max, val_avg)
        with c_res:
            pred_val = st.session_state['res_actual']['model'].predict(pd.DataFrame([inputs]))[0]
            st.markdown(f"""
                <div style="background-color:#0E1117; padding:40px; border-radius:15px; border: 2px solid #00FF00; text-align:center">
                    <h2 style="color:white; margin:0">PREDICCIÓN</h2>
                    <h1 style="color:#00FF00; font-size:70px">{pred_val:.3f}</h1>
                    <p style="color:#888">{st.session_state['target_name']}</p>
                </div>
            """, unsafe_allow_html=True)

    with tab_data:
        c_t1, c_t2 = st.columns([1, 1])
        with c_t1:
            st.subheader("Datos Utilizados")
            st.dataframe(st.session_state['df_usado'], use_container_width=True)
        with c_t2:
            st.subheader("Distribución")
            var_h = st.selectbox("Variable histograma:", [st.session_state['target_name']] + st.session_state['features_list'])
            st.plotly_chart(px.histogram(st.session_state['df_usado'], x=var_h, nbins=30, marginal="box", title=f"Frecuencia de {var_h}"), use_container_width=True)

elif archivo:
    st.info("👈 Configura y presiona 'ENTRENAR MODELO' en la barra lateral.")
