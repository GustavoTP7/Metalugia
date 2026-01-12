import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Control Hub", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

st.title("🏭 Centro de Simulación y Modelamiento")

# --- BARRA LATERAL ---
archivo = st.sidebar.file_uploader("1. Subir dataset", type=["csv", "xlsx"])
factor_iqr = st.sidebar.slider("2. Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5)

if archivo:
    df = cargar_datos(archivo)
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # --- ZONA DE CONFIGURACIÓN PRINCIPAL (AL INICIO) ---
        st.info("### 🛠️ Configuración Global")
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("🎯 Objetivo (Y)", num_cols)
        features = c2.multiselect("🔍 Entradas (X)", [c for c in num_cols if c != target])
        
        # Botón de entrenamiento prominente al inicio
        btn_entrenar = st.button("🚀 ENTRENAR Y ACTUALIZAR TODO EL PANEL", use_container_width=True, type="primary")

        if btn_entrenar:
            if not features:
                st.error("⚠️ Selecciona al menos una variable X antes de entrenar.")
            else:
                with st.spinner('Procesando modelos...'):
                    # Identificar Outliers
                    indices_out = set()
                    for col in [target] + features:
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        indices_out.update(df[(df[col] < q1 - factor_iqr*iqr) | (df[col] > q3 + factor_iqr*iqr)].index)
                    
                    df_orig = df[[target] + features].dropna()
                    df_sin = df_orig.drop(list(indices_out), errors='ignore')

                    def procesar(data):
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

                    st.session_state['res_o'] = procesar(df_orig)
                    st.session_state['res_s'] = procesar(df_sin)
                    st.session_state['features_list'] = features
                    st.session_state['target_name'] = target
                    st.session_state['df_limpio'] = df_sin
                    st.success("✅ ¡Modelos listos! Revisa las pestañas abajo.")

        # --- MOSTRAR RESULTADOS SI YA SE ENTRENÓ ---
        if 'res_o' in st.session_state:
            st.divider()
            # Métricas rápidas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Estabilidad (CV)", f"{st.session_state['res_s']['cv']:.4f}")
            m2.metric("Precisión (Test)", f"{st.session_state['res_s']['test']:.4f}")
            m3.metric("Error (RMSE)", f"{st.session_state['res_s']['rmse']:.3f}")
            m4.write(f"📊 Filas usadas: {len(st.session_state['df_limpio'])}")

            # --- PESTAÑAS DE ANÁLISIS ---
            tab_datos, tab_diagnostico, tab_simulador = st.tabs([
                "👁️ Visor de Datos", "📊 Diagnóstico & Ejes", "🎯 Simulador Real-Time"
            ])

            with tab_datos:
                st.subheader("Inspección de Datos Auditados")
                st.dataframe(st.session_state['df_limpio'], use_container_width=True)

            with tab_diagnostico:
                st.subheader("Análisis de Variables")
                col_x = st.selectbox("Eje X para gráfico de importancia:", st.session_state['features_list'])
                
                d1, d2 = st.columns(2)
                with d1:
                    fig = px.scatter(st.session_state['res_s']['df_test'], x='Real', y='Pred', trendline="ols", 
                                     title="Ajuste Real vs Predicho", color_discrete_sequence=['#00FF00'])
                    st.plotly_chart(fig, use_container_width=True)
                with d2:
                    imp_df = st.session_state['res_s']['imp'].sort_values('Imp', ascending=True)
                    st.plotly_chart(px.bar(imp_df, x='Imp', y='Var', orientation='h', title="Peso de las Variables"), use_container_width=True)

            with tab_simulador:
                st.subheader("🎯 Simulador de Escenarios")
                c_in, c_res = st.columns([1, 1])
                inputs = {}
                with c_in:
                    for f in st.session_state['features_list']:
                        val_min = float(st.session_state['df_limpio'][f].min())
                        val_max = float(st.session_state['df_limpio'][f].max())
                        val_avg = float(st.session_state['df_limpio'][f].mean())
                        inputs[f] = st.slider(f"Ajustar {f}", val_min, val_max, val_avg)
                
                with c_res:
                    pred_val = st.session_state['res_s']['model'].predict(pd.DataFrame([inputs]))[0]
                    st.markdown(f"""
                        <div style="background-color:#1E1E1E; padding:30px; border-radius:15px; border: 2px solid #00FF00; text-align:center">
                            <h3 style="color:white">PREDICCIÓN ESTIMADA</h3>
                            <h1 style="color:#00FF00; font-size:65px">{pred_val:.3f}</h1>
                            <p style="color:#888">{st.session_state['target_name']}</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("👈 Selecciona tus variables y presiona el botón 'ENTRENAR' para activar el panel.")

else:
    st.info("👋 Sube tu archivo CSV o Excel desde la barra lateral para comenzar.")
