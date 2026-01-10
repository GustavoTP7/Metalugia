import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="App Metalúrgica Pro", layout="wide")
st.title("🏭 Inteligencia de Procesos: Análisis y Simulación")

# --- FUNCIÓN DE ENTRENAMIENTO (CON MANEJO DE ERRORES) ---
@st.cache_resource
def entrenar_modelo_optimo(_data, features, target):
    # Verificación de seguridad para evitar el KeyError
    for f in features:
        if f not in _data.columns:
            return None
    if target not in _data.columns:
        return None

    X, y = _data[list(features)], _data[target]
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, tree_method='hist')
    model.fit(X_t, y_t)
    p = model.predict(X_v)
    
    return {
        'R2': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)),
        'MAE': mean_absolute_error(y_v, p), 'Bias': np.mean(p - y_v),
        'y_real': y_v, 'y_pred': p, 'model': model, 'n': len(_data), 
        'df_val': X_v.assign(REAL=y_v, PRED=p)
    }

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Entrada")
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    # Limpiar caché si se sube un archivo nuevo para evitar KeyErrors
    if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo.name:
        st.cache_resource.clear()
        st.session_state.ultimo_archivo = archivo.name

    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    # Limpieza crucial de nombres de columnas
    df.columns = df.columns.astype(str).str.strip()
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab_vista, tab_limpieza, tab_modelo, tab_analisis, tab_simulador = st.tabs([
        "👁️ 1. Vista Previa", "🧹 2. Limpieza", "🛠️ 3. Entrenamiento e Impacto", "🔍 4. Análisis Comparativo X-Y", "🎯 5. Panel de Control"
    ])

    with tab_vista:
        st.subheader("Visualización de Datos Cargados")
        st.dataframe(df.head(50), use_container_width=True)

    with tab_limpieza:
        st.subheader("⚙️ Gestión de Outliers")
        modo = st.radio("Método:", ["Automático (IQR Global)", "Manual (Selección en tabla)"])
        col_ref = st.selectbox("Variable de referencia:", num_cols)
        
        Q1, Q3 = df[col_ref].quantile(0.25), df[col_ref].quantile(0.75)
        IQR = Q3 - Q1
        inf, sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df[(df[col_ref] < inf) | (df[col_ref] > sup)].copy()

        if modo == "Manual (Selección en tabla)":
            outliers['ELIMINAR'] = True
            edited = st.data_editor(outliers, use_container_width=True, hide_index=True)
            st.session_state['borrar'] = edited[edited['ELIMINAR'] == True].index
        else:
            st.session_state['borrar'] = outliers.index
            st.success(f"Modo Automático: {len(outliers)} filas listas para remover.")

    with tab_modelo:
        st.subheader("🛠️ Evaluación de Impacto")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Variables (X):", [c for c in num_cols if c != target])
        
        if st.button("🚀 Entrenar y Comparar"):
            if not features: 
                st.error("Selecciona al menos una variable.")
            else:
                df_s = df[[target] + features].dropna()
                df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')
                
                with st.spinner('Entrenando modelos rápidos...'):
                    res_s = entrenar_modelo_optimo(df_s, tuple(features), target)
                    res_l = entrenar_modelo_optimo(df_l, tuple(features), target)
                    
                    if res_s is None or res_l is None:
                        st.error("Error en las columnas. Por favor, recarga la página o el archivo.")
                    else:
                        met_df = pd.DataFrame({
                            'Métrica': ['R² Precisión', 'RMSE (Riesgo)', 'MAE (Error)', 'Muestras'],
                            'Original (Sucio)': [f"{res_s['R2']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['MAE']:.4f}", res_s['n']],
                            'Filtrado (Limpio)': [f"{res_l['R2']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['MAE']:.4f}", res_l['n']]
                        })
                        st.table(met_df)
                        st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'db': df_l, 'res_s': res_s, 'res_l': res_l})

    with tab_analisis:
        if 'res_l' in st.session_state:
            st.subheader("🔍 Explorador Comparativo")
            cx, cy = st.columns(2)
            # Aseguramos que los ejes existan en el nuevo resultado
            eje_x = cx.selectbox("Eje X:", st.session_state.feat)
            eje_y = cy.selectbox("Eje Y:", [st.session_state.targ] + st.session_state.feat)

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(px.scatter(st.session_state.res_s['df_val'], x=eje_x, y=eje_y, trendline="ols", title="Dataset Original", color_discrete_sequence=["gray"]), use_container_width=True)
            with g2:
                st.plotly_chart(px.scatter(st.session_state.res_l['df_val'], x=eje_x, y=eje_y, trendline="ols", title="Dataset Limpio", color_discrete_sequence=["#2ecc71"]), use_container_width=True)
        else:
            st.warning("Entrena el modelo primero.")

    with tab_simulador:
        if 'mod' in st.session_state:
            st.subheader("🎯 Panel de Control")
            col_sl, col_ga = st.columns([1, 2])
            with col_sl:
                inputs = {}
                for f in st.session_state.feat:
                    v_min, v_max = float(st.session_state.db[f].min()), float(st.session_state.db[f].max())
                    v_avg = float(st.session_state.db[f].mean())
                    inputs[f] = st.slider(f, v_min, v_max, v_avg)
            
            pred = st.session_state.mod.predict(pd.DataFrame([inputs]))[0]
            with col_ga:
                st.metric(f"PREDICCIÓN DE {st.session_state.targ}", f"{pred:.2f}")
                sens = {f: st.session_state.mod.predict(pd.DataFrame([inputs]).assign(**{f: inputs[f] + (st.session_state.db[f].max()-st.session_state.db[f].min())*0.05}))[0] - pred for f in st.session_state.feat}
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Sensibilidad Operativa", color=list(sens.values()), color_continuous_scale="RdYlGn"), use_container_width=True)
else:
    st.info("Sube un archivo para comenzar.")
