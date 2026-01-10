import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="App Metalúrgica Pro", layout="wide")
st.title("🏭 Inteligencia de Procesos: Limpieza Multivariable")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Entrada")
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab_vista, tab_limpieza, tab_modelo, tab_analisis, tab_simulador = st.tabs([
        "👁️ 1. Vista Previa", "🧹 2. Limpieza Global", "🛠️ 3. Entrenamiento", "🔍 4. Análisis X-Y", "🎯 5. Simulador"
    ])

    with tab_vista:
        st.subheader("Datos Originales")
        st.dataframe(df.head(50), use_container_width=True)

    with tab_limpieza:
        st.subheader("⚙️ Gestión de Outliers (Multivariable)")
        
        # Selección de variables para monitorear outliers
        cols_limpieza = st.multiselect("Selecciona las variables para detectar outliers:", num_cols, default=num_cols[:3])
        
        modo = st.radio("Método de aplicación:", ["Automático (Limpiar todo lo detectado)", "Manual (Revisar lista global)"])

        # LÓGICA DE DETECCIÓN GLOBAL
        indices_outliers = set()
        for col in cols_limpieza:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            bajo = Q1 - 1.5 * IQR
            alto = Q3 + 1.5 * IQR
            # Guardamos los índices de las filas que fallan en esta columna
            outliers_col = df[(df[col] < bajo) | (df[col] > alto)].index
            indices_outliers.update(outliers_col)
        
        df_outliers = df.loc[list(indices_outliers)].copy()

        if modo == "Manual (Revisar lista global)":
            st.info(f"Se han detectado {len(df_outliers)} filas con anomalías en las variables seleccionadas.")
            df_outliers['ELIMINAR'] = True
            edited = st.data_editor(df_outliers, use_container_width=True, hide_index=True)
            st.session_state['borrar'] = edited[edited['ELIMINAR'] == True].index
        else:
            st.session_state['borrar'] = df_outliers.index
            st.success(f"Modo Automático Activo: Se descartarán {len(df_outliers)} filas que contienen ruidos.")

    with tab_modelo:
        st.subheader("🛠️ Entrenamiento de Alta Precisión")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Variables de Entrada (X):", [c for c in num_cols if c != target])
        
        if st.button("🚀 Iniciar Entrenamiento", use_container_width=True):
            if not features:
                st.error("Selecciona variables.")
            else:
                progreso = st.progress(0)
                with st.spinner('Procesando...'):
                    # Preparar data limpia vs sucia
                    df_s = df[[target] + features].dropna()
                    df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')
                    progreso.progress(30)

                    def entrenar_exacto(data):
                        X, y = data[features], data[target]
                        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
                        # Motor original robusto
                        m = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05)
                        m.fit(X_t, y_t)
                        p = m.predict(X_v)
                        res = X_v.copy()
                        res['REAL'], res['PREDICCION'] = y_v.values, p
                        return {'R2': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)), 'model': m, 'df_val': res, 'n': len(data)}

                    res_s = entrenar_exacto(df_s)
                    progreso.progress(70)
                    res_l = entrenar_exacto(df_l)
                    progreso.progress(100)

                    st.markdown("### 📊 Resultado del Filtro Multivariable")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Precisión Final (R²)", f"{res_l['R2']:.4f}", f"{res_l['R2']-res_s['R2']:.4f}")
                    col_m2.metric("Reducción de Error (RMSE)", f"{res_l['RMSE']:.4f}", f"{res_l['RMSE']-res_s['RMSE']:.4f}", delta_color="inverse")
                    col_m3.metric("Datos Utilizados", f"{res_l['n']}", f"-{res_s['n']-res_l['n']}")
                    
                    st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'db': df_l, 'res_s': res_s, 'res_l': res_l})
                    st.toast("¡Modelo entrenado con éxito!", icon="✅")

    with tab_analisis:
        if 'res_l' in st.session_state:
            st.subheader("🔍 Análisis de Ejes Personalizados")
            cols_graf = st.session_state.res_l['df_val'].columns.tolist()
            ex = st.selectbox("Eje X:", cols_graf, index=0)
            ey = st.selectbox("Eje Y:", cols_graf, index=len(cols_graf)-2)
            
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(px.scatter(st.session_state.res_s['df_val'], x=ex, y=ey, trendline="ols", title="Dataset Bruto", color_discrete_sequence=["gray"]), use_container_width=True)
            with g2:
                st.plotly_chart(px.scatter(st.session_state.res_l['df_val'], x=ex, y=ey, trendline="ols", title="Dataset Limpio Global", color_discrete_sequence=["#2ecc71"]), use_container_width=True)

    with tab_simulador:
        if 'mod' in st.session_state:
            st.subheader("🎯 Simulador What-If")
            col_sl, col_ga = st.columns([1, 2])
            with col_sl:
                inputs = {f: st.slider(f, float(st.session_state.db[f].min()), float(st.session_state.db[f].max()), float(st.session_state.db[f].mean())) for f in st.session_state.feat}
            
            pred = st.session_state.mod.predict(pd.DataFrame([inputs]))[0]
            with col_ga:
                st.metric(f"PREDICCIÓN {st.session_state.targ}", f"{pred:.2f}")
                sens = {f: st.session_state.mod.predict(pd.DataFrame([inputs]).assign(**{f: inputs[f] + (st.session_state.db[f].max()-st.session_state.db[f].min())*0.05}))[0] - pred for f in st.session_state.feat}
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Sensibilidad", color=list(sens.values()), color_continuous_scale="RdYlGn"), use_container_width=True)
else:
    st.info("Sube tu archivo para comenzar.")
