import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Predictiva", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    return df

st.title("🏭 Modelamiento: Original vs Sin Outliers")

archivo = st.sidebar.file_uploader("Subir dataset (CSV o Excel)", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- SELECCIÓN DE VARIABLES ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        target = st.selectbox("🎯 Objetivo (Y):", num_cols)
        features = st.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])
        factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5)

    if target and features:
        # --- AUDITORÍA DE DATOS ---
        indices_out = set()
        for col in [target] + features:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            indices_out.update(df[(df[col] < q1 - factor_iqr*iqr) | (df[col] > q3 + factor_iqr*iqr)].index)
        
        df_orig = df[[target] + features].dropna()
        df_sin = df_orig.drop(list(indices_out), errors='ignore')

        # --- MOTOR DE ENTRENAMIENTO ---
        def entrenar(data, etiqueta):
            X, y = data[features], data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Modelo idéntico al de Colab
            model = xgb.XGBRegressor(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                n_jobs=-1, 
                tree_method='hist', 
                random_state=42
            )
            
            # CV sobre el set de entrenamiento
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            return {
                "cv": np.mean(cv_scores),
                "test": r2_score(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "importancia": pd.DataFrame({'Var': features, 'Imp': model.feature_importances_}).sort_values('Imp'),
                "plot": pd.DataFrame({'Real': y_test, 'Pred': pred})
            }

        # Ejecución
        res_o = entrenar(df_orig, "ORIGINAL")
        res_s = entrenar(df_sin, "SIN OUTLIERS")

        # --- VISUALIZACIÓN ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📊 Modelo Original")
            st.metric("Estabilidad (CV)", f"{res_o['cv']:.4f}")
            st.metric("Precisión (Test)", f"{res_o['test']:.4f}")
            st.metric("Error (RMSE)", f"{res_o['rmse']:.4f}")

        with c2:
            st.subheader("🧹 Modelo Sin Outliers")
            st.metric("Estabilidad (CV)", f"{res_s['cv']:.4f}")
            st.metric("Precisión (Test)", f"{res_s['test']:.4f}")
            st.metric("Error (RMSE)", f"{res_s['rmse']:.4f}")

        st.divider()
        
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(px.bar(res_s['importancia'], x='Imp', y='Var', orientation='h', title="Variables Críticas (Sin Outliers)"), use_container_width=True)
        with d2:
            st.plotly_chart(px.scatter(res_s['plot'], x='Real', y='Pred', trendline="ols", title="Ajuste: Real vs Predicho"), use_container_width=True)

else:
    st.info("👋 Sube tu archivo CSV o Excel en la barra lateral para comenzar.")
