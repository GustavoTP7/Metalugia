import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import time

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="App Metalúrgica Pro", layout="wide")
st.title("🏭 Inteligencia de Procesos: Entrenamiento de Alta Precisión")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Entrada")
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    # Limpieza de nombres de columnas nada más cargar
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab_vista, tab_limpieza, tab_modelo, tab_analisis, tab_simulador = st.tabs([
        "👁️ 1. Vista Previa", "🧹 2. Limpieza", "🛠️ 3. Entrenamiento", "🔍 4. Análisis X-Y", "🎯 5. Simulador"
    ])

    with tab_vista:
        st.subheader("Datos Cargados")
        st.dataframe(df.head(50), use_container_width=True)

    with tab_limpieza:
        st.subheader("⚙️ Gestión de Outliers")
        modo = st.radio("Método de limpieza:", ["Manual (Seleccionar en tabla)", "Automático (IQR Global)"])
        col_ref = st.selectbox("Variable de referencia para limpieza:", num_cols)
        
        Q1, Q3 = df[col_ref].quantile(0.25), df[col_ref].quantile(0.75)
        IQR = Q3 - Q1
        inf, sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df[(df[col_ref] < inf) | (df[col_ref] > sup)].copy()

        if modo == "Manual (Seleccionar en tabla)":
            outliers['ELIMINAR'] = True
            edited = st.data_editor(outliers, use_container_width=True, hide_index=True)
            st.session_state['borrar'] = edited[edited['ELIMINAR'] == True].index
        else:
            st.session_state['borrar'] = outliers.index
            st.success(f"Modo Automático: {len(outliers)} filas marcadas para exclusión.")

    with tab_modelo:
        st.subheader("🛠️ Entrenamiento de Modelos (Limpio vs Bruto)")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Variables (X):", [c for c in num_cols if c != target])
        
        if st.button("🚀 Iniciar Entrenamiento de Alta Precisión", use_container_width=True):
            if not features:
                st.error("Debes seleccionar variables de entrada.")
            else:
                # CREAR BARRA DE PROGRESO E INDICADOR
                progreso = st.progress(0)
                status_text = st.empty()
                
                with st.spinner('Entrenando modelos... Esto puede tardar según el tamaño de la data.'):
                    # 1. Preparar Datasets
                    status_text.text("Preparando datos...")
                    df_s = df[[target] + features].dropna()
                    df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')
                    progreso.progress(20)

                    # 2. Función de entrenamiento (MÉTODO ORIGINAL ROBUSTO)
                    def entrenar_robusto(data):
                        X, y = data[features], data[target]
                        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Usamos el motor original con 150 estimadores para máxima precisión
                        m = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)
                        m.fit(X_t, y_t)
                        p = m.predict(X_v)
                        
                        df_res = X_v.copy()
                        df_res['REAL'] = y_v.values
                        df_res['PREDICCION'] = p
                        return {'R2': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)), 
                                'MAE': mean_absolute_error(y_v, p), 'model': m, 'n': len(data), 'df_val': df_res}

                    status_text.text("Entrenando Modelo 1 (Con Outliers)...")
                    res_s = entrenar_robusto(df_s)
                    progreso.progress(60)

                    status_text.text("Entrenando Modelo 2 (Limpio)...")
                    res_l = entrenar_robusto(df_l)
                    progreso.progress(100)
                    
                    status_text.text("¡Entrenamiento Completo!")
                    st.toast('Modelos listos para análisis', icon='✅')

                    # Mostrar Métricas
                    st.markdown("### 📊 Resultado de la Comparación")
                    met_df = pd.DataFrame({
                        'Métrica': ['R² Precisión', 'RMSE (Error)', 'Muestras'],
                        'Original (Sucio)': [f"{res_s['R2']:.4f}", f"{res_s['RMSE']:.4f}", res_s['n']],
                        'Limpio (Filtrado)': [f"{res_l['R2']:.4f}", f"{res_l['RMSE']:.4f}", res_l['n']]
                    })
                    st.table(met_df)
                    
                    # Guardar todo en session_state
                    st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'db': df_l, 'res_s': res_s, 'res_l': res_l})

    with tab_analisis:
        if 'res_l' in st.session_state:
            st.subheader("🔍 Explorador Comparativo X-Y")
            columnas = st.session_state.res_l['df_val'].columns.tolist()
            cx, cy = st.columns(2)
            eje_x = cx.selectbox("Eje X:", columnas, index=0)
            eje_y = cy.selectbox("Eje Y:", columnas, index=len(columnas)-1)

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(px.scatter(st.session_state.res_s['df_val'], x=eje_x, y=eje_y, trendline="ols", title="Dataset Original", color_discrete_sequence=["gray"]), use_container_width=True)
            with g2:
                st.plotly_chart(px.scatter(st.session_state.res_l['df_val'], x=eje_x, y=eje_y, trendline="ols", title="Dataset Limpio", color_discrete_sequence=["#2ecc71"]), use_container_width=True)
        else:
            st.warning("Por favor, entrena el modelo en la pestaña anterior.")

    with tab_simulador:
        if 'mod' in st.session_state:
            st.subheader("🎯 Panel de Control (Simulador)")
            col_sl, col_ga = st.columns([1, 2])
            with col_sl:
                inputs = {f: st.slider(f, float(st.session_state.db[f].min()), float(st.session_state.db[f].max()), float(st.session_state.db[f].mean())) for f in st.session_state.feat}
            
            pred = st.session_state.mod.predict(pd.DataFrame([inputs]))[0]
            with col_ga:
                st.metric(f"PREDICCIÓN DE {st.session_state.targ}", f"{pred:.2f}")
                sens = {f: st.session_state.mod.predict(pd.DataFrame([inputs]).assign(**{f: inputs[f] + (st.session_state.db[f].max()-st.session_state.db[f].min())*0.05}))[0] - pred for f in st.session_state.feat}
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Sensibilidad Operativa", color=list(sens.values()), color_continuous_scale="RdYlGn"), use_container_width=True)
else:
    st.info("👋 Sube un archivo Excel o CSV para comenzar.")
