import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

st.set_page_config(page_title="Auditoría Metalúrgica", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

st.title("🏭 Auditoría de Predicciones por Fecha y Turno")

with st.sidebar:
    st.header("1️⃣ Carga")
    archivo = st.file_uploader("Subir dataset", type=["csv", "xlsx"])
    modo_datos = st.radio("Modo:", ["Original", "Sin Outliers"])
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            # Identificar columnas
            cols = df.columns.tolist()
            st.header("2️⃣ Configuración")
            col_fecha = st.selectbox("📅 Columna de Fecha:", [c for c in cols if 'fecha' in c.lower() or 'date' in c.lower()] + cols)
            col_turno = st.selectbox("🕒 Columna de Turno:", [c for c in cols if 'turno' in c.lower()] + cols)
            target = st.selectbox("🎯 Objetivo (Y):", df.select_dtypes(include=np.number).columns.tolist())
            features = st.multiselect("🔍 Entradas (X):", [c for c in cols if c not in [target, col_fecha, col_turno]])
            
            btn_entrenar = st.button("🚀 ANALIZAR", use_container_width=True, type="primary")

if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features:
        st.error("Selecciona variables X.")
    else:
        # Crear Referencia Única para no perder la pista
        df['REF_AUDITORIA'] = df[col_fecha].astype(str) + " | " + df[col_turno].astype(str)
        
        # Limpieza (opcional según el modo)
        df_base = df[['REF_AUDITORIA', target] + features].dropna().copy()
        
        # Codificar Turnos si son texto para el modelo
        mapeos = {}
        X_input = df_base[features].copy()
        for col in features:
            if X_input[col].dtype == 'object':
                categorias = sorted(X_input[col].unique())
                mapeos[col] = categorias
                X_input[col] = X_input[col].map({val: i for i, val in enumerate(categorias)})

        # Separar datos (70/30) pero manteniendo el índice para recuperar la fecha
        X_train, X_test, y_train, y_test = train_test_split(X_input, df_base[target], test_size=0.3, random_state=42)
        
        # Entrenar
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, tree_method='hist')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # CREAR TABLA DE AUDITORÍA (Aquí está la clave)
        df_auditoria = pd.DataFrame({
            'Fecha_Turno': df_base.loc[y_test.index, 'REF_AUDITORIA'],
            'Real': y_test.values,
            'Predicho': preds,
            'Error_Absoluto': np.abs(y_test.values - preds)
        }).sort_values('Error_Absoluto', ascending=False)

        st.session_state['audit'] = df_auditoria
        st.session_state['res'] = {'model': model, 'features': features, 'mapeos': mapeos, 'df_work': df_base, 'target': target}

if 'audit' in st.session_state:
    t_res, t_audit, t_sim = st.tabs(["📊 Resultados Genrales", "🔎 Auditoría Detallada", "🎯 Simulador"])

    with t_res:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(st.session_state['audit'], x='Real', y='Predicho', hover_data=['Fecha_Turno'],
                             title="Ajuste (Pasa el mouse para ver Fecha y Turno)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Gráfico de error en el tiempo
            fig_err = px.line(st.session_state['audit'].sort_index(), y='Error_Absoluto', 
                              title="Evolución del Error (Inestabilidad por Fecha)")
            st.plotly_chart(fig_err, use_container_width=True)

    with t_audit:
        st.subheader("🚩 ¿Dónde falló más el modelo?")
        st.write("Esta tabla muestra las fechas y turnos donde la predicción fue menos precisa:")
        st.dataframe(st.session_state['audit'], use_container_width=True)

    with t_sim:
        # Simulador (mismo que antes)
        st.subheader("Simulador")
        # ... (código del simulador anterior)
