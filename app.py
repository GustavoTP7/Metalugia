import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="App Metalúrgica Pro", layout="wide")
st.title("🏭 Inteligencia de Procesos: Limpieza Proactiva e Impacto")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Entrada")
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab_vista, tab_limpieza, tab_modelo, tab_simulador = st.tabs([
        "👁️ 1. Vista Previa", "🧹 2. Limpieza de Datos", "🛠️ 3. Entrenamiento e Impacto", "🎯 4. Panel de Control"
    ])

    with tab_vista:
        st.subheader("Inspección de Datos")
        st.dataframe(df.head(20), use_container_width=True)

    # --- TAB 2: LIMPIEZA (AUTO + MANUAL) ---
    with tab_limpieza:
        st.subheader("⚙️ Configuración de Limpieza")
        modo_limpieza = st.radio("Selecciona el método de limpieza:", 
                                  ["Manual (Elegir filas en tabla)", "Automático (IQR Global)"])
        
        col_ref = st.selectbox("Variable de referencia para análisis:", num_cols)
        
        # Lógica IQR
        Q1, Q3 = df[col_ref].quantile(0.25), df[col_ref].quantile(0.75)
        IQR = Q3 - Q1
        inf, sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers_detectados = df[(df[col_ref] < inf) | (df[col_ref] > sup)].copy()

        if modo_limpieza == "Manual (Elegir filas en tabla)":
            st.info("Selecciona manualmente qué filas deseas descartar en la columna 'ELIMINAR'.")
            outliers_detectados['ELIMINAR'] = True
            edited = st.data_editor(outliers_detectados, use_container_width=True, hide_index=True)
            st.session_state['indices_borrar'] = edited[edited['ELIMINAR'] == True].index
        else:
            st.success(f"Modo Automático: Se han marcado {len(outliers_detectados)} filas para eliminar automáticamente.")
            st.session_state['indices_borrar'] = outliers_detectados.index

    # --- TAB 3: ENTRENAMIENTO E IMPACTO ---
    with tab_modelo:
        st.subheader("🛠️ Evaluación de Impacto del Modelo")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo:", num_cols)
        features = c2.multiselect("🔍 Variables de Entrada:", [c for c in num_cols if c != target])
        
        if st.button("🚀 Entrenar y Comparar Resultados", use_container_width=True):
            if not features: 
                st.error("Selecciona variables.")
            else:
                # 1. Datasets
                df_sucio = df[[target] + features].dropna()
                df_limpio = df_sucio.drop(st.session_state.get('indices_borrar', []), errors='ignore')

                def entrenar(data):
                    X, y = data[features], data[target]
                    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
                    m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.06)
                    m.fit(X_t, y_t)
                    p = m.predict(X_v)
                    return {'R2': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)), 'n': len(data), 'model': m}

                s = entrenar(df_sucio)
                l = entrenar(df_limpio)

                # --- CUADRO DE IMPACTO ---
                st.markdown("### 📊 Comparativa: Bruto vs Limpio")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                # Cálculo de deltas para las métricas
                diff_r2 = l['R2'] - s['R2']
                diff_rmse = l['RMSE'] - s['RMSE']

                col_m1.metric("Precisión R²", f"{l['R2']:.4f}", f"{diff_r2:.4f}")
                col_m2.metric("Riesgo RMSE", f"{l['RMSE']:.4f}", f"{diff_rmse:.4f}", delta_color="inverse")
                col_m3.metric("Muestras Finales", l['n'], f"-{s['n'] - l['n']}")

                # Tabla formal
                tabla = pd.DataFrame({
                    'Estado': ['Con Outliers', 'Limpio (Post-IQR)'],
                    'R²': [f"{s['R2']:.4f}", f"{l['R2']:.4f}"],
                    'RMSE': [f"{s['RMSE']:.4f}", f"{l['RMSE']:.4f}"],
                    'Muestras': [s['n'], l['n']]
                })
                st.table(tabla)

                st.session_state.update({'mod': l['model'], 'feat': features, 'targ': target, 'db': df_limpio})

    # --- TAB 4: PANEL DE CONTROL ---
    with tab_simulador:
        if 'mod' in st.session_state:
            st.subheader("🎯 Simulador de Decisiones")
            col_sl, col_ga = st.columns([1, 2])
            with col_sl:
                inputs = {f: st.slider(f, float(st.session_state.db[f].min()), float(st.session_state.db[f].max()), float(st.session_state.db[f].mean())) for f in st.session_state.feat}
            
            pred = st.session_state.mod.predict(pd.DataFrame([inputs]))[0]
            with col_ga:
                st.metric(f"PREDICCIÓN DE {st.session_state.targ}", f"{pred:.2f}")
                
                sens = {}
                for f in st.session_state.feat:
                    df_p = pd.DataFrame([inputs])
                    df_p[f] += (st.session_state.db[f].max() - st.session_state.db[f].min()) * 0.05
                    sens[f] = st.session_state.mod.predict(df_p)[0] - pred
                
                fig = px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', 
                             title="Sensibilidad Operativa", color=list(sens.values()), 
                             color_continuous_scale="RdYlGn", labels={'x': 'Impacto (+)', 'y': 'Variable'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Entrena el modelo primero.")
else:
    st.info("Sube tu archivo Excel/CSV para activar la inteligencia de planta.")
