import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import io

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Metalurgia Pro: Auditoría de Sesgo", layout="wide")
st.title("🏭 Auditoría de Modelos: Impacto de Outliers en el Sesgo")

# --- CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab_limpieza, tab_modelo, tab_analisis = st.tabs(["🧹 1. Auditoría", "🛠️ 2. Entrenamiento e Impacto", "🔍 3. Análisis X-Y"])

    with tab_limpieza:
        st.subheader("⚙️ Identificación de Ruidos")
        cols_auditoria = st.multiselect("Variables a auditar:", num_cols, default=num_cols[:3])
        
        indices_borrar = set()
        for col in cols_auditoria:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            indices_borrar.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
        
        st.session_state['borrar'] = list(indices_borrar)
        st.warning(f"Se han identificado {len(indices_borrar)} datos extremos que podrían causar sobreestimación.")

    with tab_modelo:
        st.subheader("🛠️ Evaluación de Métricas y Sesgo")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])
        
        if st.button("🚀 Ejecutar Comparativa", use_container_width=True):
            with st.spinner('Calculando impacto en la precisión...'):
                df_s = df[[target] + features].dropna()
                df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')

                def entrenar_con_sesgo(data):
                    X, y = data[features], data[target]
                    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
                    m = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05)
                    m.fit(X_t, y_t)
                    p = m.predict(X_v)
                    # El Sesgo (Bias) es la media de (Predicho - Real)
                    sesgo = np.mean(p - y_v)
                    return {
                        'R2': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)),
                        'Sesgo': sesgo, 'n': len(data), 'model': m, 'df_val': X_v.assign(REAL=y_v, PRED=p)
                    }

                res_s = entrenar_con_sesgo(df_s)
                res_l = entrenar_con_sesgo(df_l)

                # TABLA COMPARATIVA
                tabla = pd.DataFrame({
                    "Métrica": ["Precisión (R²)", "Error (RMSE)", "Sesgo (Promedio Error)", "Muestras"],
                    "Modelo Sucio": [f"{res_s['R2']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['Sesgo']:.4f}", res_s['n']],
                    "Modelo Limpio": [f"{res_l['R2']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['Sesgo']:.4f}", res_l['n']],
                    "Mejora": [f"{res_l['R2']-res_s['R2']:.4f}", f"{res_l['RMSE']-res_s['RMSE']:.4f}", f"{abs(res_s['Sesgo'])-abs(res_l['Sesgo']):.4f}", res_l['n']-res_s['n']]
                })
                st.table(tabla)

                # EXPLICACIÓN DEL SESGO
                if abs(res_s['Sesgo']) > abs(res_l['Sesgo']):
                    st.success(f"💡 El modelo limpio redujo el sesgo en {abs(res_s['Sesgo'] - res_l['Sesgo']):.4f}. Es más honesto.")
                else:
                    st.info("💡 El sesgo es similar, pero el modelo limpio tiene menos error aleatorio (RMSE).")

                st.session_state.update({'res_s': res_s, 'res_l': res_l, 'feat': features})

    with tab_analisis:
        if 'res_l' in st.session_state:
            st.subheader("🔍 Comparación Visual: ¿Dónde mentía el modelo?")
            cols_v = st.session_state.res_l['df_val'].columns.tolist()
            ex = st.selectbox("Eje X:", st.session_state.feat)
            ey = st.selectbox("Eje Y:", [c for c in cols_v if c not in st.session_state.feat])

            g1, g2 = st.columns(2)
            # Gráfico con Outliers
            with g1:
                st.plotly_chart(px.scatter(st.session_state.res_s['df_val'], x=ex, y=ey, trendline="ols", 
                                          title="Modelo Sucio: La línea se desvía por los extremos", color_discrete_sequence=["#e74c3c"]), use_container_width=True)
            # Gráfico Limpio
            with g2:
                st.plotly_chart(px.scatter(st.session_state.res_l['df_val'], x=ex, y=ey, trendline="ols", 
                                          title="Modelo Limpio: La línea sigue el proceso real", color_discrete_sequence=["#2ecc71"]), use_container_width=True)
else:
    st.info("Sube un archivo para iniciar el análisis de sesgo.")
