import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Auditoría por Turno Metalúrgico", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return None

st.title("🏭 Auditoría Detallada: Fecha y Turno")

# --- PANEL DE CONTROL ---
with st.sidebar:
    st.header("1️⃣ Carga de Datos")
    archivo = st.file_uploader("Subir dataset", type=["csv", "xlsx"])
    modo_datos = st.radio("Limpieza de Outliers:", ["Desactivada (Original)", "Activada (Auditado)"])
    factor_iqr = st.slider("Sensibilidad IQR", 1.0, 3.0, 1.5, disabled=(modo_datos == "Desactivada (Original)"))
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            cols = df.columns.tolist()
            st.header("2️⃣ Configuración")
            # Seleccionamos la columna que subiste (ej: 01/03/2025_TA)
            col_id_turno = st.selectbox("🔑 Columna Fecha_Turno:", cols, help="Selecciona la columna que une fecha y turno")
            target = st.selectbox("🎯 Objetivo (Y):", df.select_dtypes(include=np.number).columns.tolist())
            features = st.multiselect("🔍 Variables de Entrada (X):", [c for c in cols if c not in [target, col_id_turno]])
            
            btn_entrenar = st.button("🚀 INICIAR AUDITORÍA", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO ---
if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features:
        st.error("⚠️ Elige al menos una variable X.")
    else:
        with st.spinner('Analizando datos por turno...'):
            # 1. Preparar dataset base manteniendo el ID de Turno
            df_base = df[[col_id_turno, target] + features].dropna().copy()
            
            # 2. Manejo de Outliers (opcional)
            if modo_datos == "Activada (Auditado)":
                indices_out = set()
                for col in [target] + [f for f in features if df_base[f].dtype in [np.float64, np.int64]]:
                    q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                    iqr = q3 - q1
                    indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
                df_final = df_base.drop(list(indices_out), errors='ignore')
            else:
                df_final = df_base

            # 3. Encoding de variables de texto para el modelo
            X_encoded = df_final[features].copy()
            mapeos = {}
            for col in features:
                if X_encoded[col].dtype == 'object':
                    cats = sorted(X_encoded[col].unique())
                    mapeos[col] = cats
                    X_encoded[col] = X_encoded[col].map({v: i for i, v in enumerate(cats)})

            # 4. Entrenamiento (mantenemos el índice para el rastreo)
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, df_final[target], test_size=0.3, random_state=42)
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, tree_method='hist')
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # 5. Crear tabla de auditoría uniendo Predicción con ID de Turno
            df_audit = pd.DataFrame({
                'ID_Turno': df_final.loc[y_test.index, col_id_turno],
                'Valor_Real': y_test.values,
                'Predicción': preds,
                'Error_Abs': np.abs(y_test.values - preds)
            }).sort_values('Error_Abs', ascending=False)

            st.session_state['audit_data'] = df_audit
            st.session_state['model_info'] = {'model': model, 'features': features, 'mapeos': mapeos, 'df_work': df_final, 'target': target}

# --- VISUALIZACIÓN ---
if 'audit_data' in st.session_state:
    tab_res, tab_audit, tab_sim = st.tabs(["📊 Desempeño", "🔎 Dónde falló el turno", "🎯 Simulador"])

    with tab_res:
        c1, c2 = st.columns([2, 1])
        with c1:
            # Gráfico donde el HOVER muestra el turno
            fig = px.scatter(st.session_state['audit_data'], x='Valor_Real', y='Predicción', 
                             hover_data=['ID_Turno'], title="Ajuste Real vs Predicho (Pasa el mouse por los puntos)")
            fig.add_shape(type="line", x0=st.session_state['audit_data']['Valor_Real'].min(), y0=st.session_state['audit_data']['Valor_Real'].min(),
                          x1=st.session_state['audit_data']['Valor_Real'].max(), y1=st.session_state['audit_data']['Valor_Real'].max(),
                          line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Error Promedio (RMSE)", f"{np.sqrt(mean_squared_error(st.session_state['audit_data']['Valor_Real'], st.session_state['audit_data']['Predicción'])):.3f}")
            st.metric("Precisión (R²)", f"{r2_score(st.session_state['audit_data']['Valor_Real'], st.session_state['audit_data']['Predicción']):.3f}")

    with tab_audit:
        st.subheader("🚩 Ranking de Turnos con Mayor Desviación")
        st.write("Esta tabla te permite identificar exactamente en qué fecha y turno la planta se alejó del modelo:")
        st.dataframe(st.session_state['audit_data'], use_container_width=True)
        
        # Histograma de errores para ver si hay muchos turnos fallando o solo unos pocos
        fig_err = px.histogram(st.session_state['audit_data'], x="Error_Abs", nbins=20, title="Distribución del Error")
        st.plotly_chart(fig_err, use_container_width=True)

    with tab_sim:
        st.subheader("Simulador What-If")
        col_in, col_out = st.columns(2)
        inputs = {}
        with col_in:
            for f in st.session_state['model_info']['features']:
                if f in st.session_state['model_info']['mapeos']:
                    opc = st.session_state['model_info']['mapeos'][f]
                    sel = st.selectbox(f, opc)
                    inputs[f] = opc.index(sel)
                else:
                    v_min = float(st.session_state['model_info']['df_work'][f].min())
                    v_max = float(st.session_state['model_info']['df_work'][f].max())
                    inputs[f] = st.slider(f, v_min, v_max, float(st.session_state['model_info']['df_work'][f].mean()))
        with col_out:
            p = st.session_state['model_info']['model'].predict(pd.DataFrame([inputs]))[0]
            st.markdown(f"<h1 style='text-align: center; color: #00FF00;'>Predicción: {p:.3f}</h1>", unsafe_allow_html=True)

elif archivo:
    st.info("👈 Configura la columna de Fecha_Turno y presiona el botón.")
