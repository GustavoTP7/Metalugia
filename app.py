import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="App Metalúrgica Pro: Validación Cruzada", layout="wide")
st.title("🏭 Modelamiento con Validación Cruzada (K-Fold)")

# --- CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tabs = st.tabs(["👁️ Vista Previa", "🧹 Auditoría", "🛠️ Entrenamiento K-Fold", "🔍 Análisis X-Y", "🎯 Simulador"])

    # --- TAB 2: AUDITORÍA (Mantenemos tu lógica multivariable) ---
    with tabs[1]:
        st.subheader("⚙️ Filtro Global de Ruidos")
        cols_auditoria = st.multiselect("Variables para limpiar:", num_cols, default=num_cols[:3])
        indices_borrar = set()
        for col in cols_auditoria:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            indices_borrar.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
        st.session_state['borrar'] = list(indices_borrar)
        st.info(f"Filas con ruidos detectadas: {len(indices_borrar)}")

    # --- TAB 3: ENTRENAMIENTO CON VALIDACIÓN CRUZADA ---
    with tabs[2]:
        st.subheader("🛠️ Entrenamiento con K-Fold (5 particiones)")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])

        if st.button("🚀 Iniciar Validación Cruzada", use_container_width=True):
            if not features:
                st.error("Selecciona variables.")
            else:
                with st.spinner('Ejecutando K-Fold Cross Validation...'):
                    df_s = df[[target] + features].dropna()
                    df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')

                    def ejecutar_kfold(data, label):
                        X, y = data[features], data[target]
                        # Definimos 5 cortes aleatorios
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05)
                        
                        # Ejecutamos Cross Validation para el R2
                        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                        
                        # Entrenamiento final para predicciones visuales
                        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
                        model.fit(X_t, y_t)
                        preds = model.predict(X_v)
                        
                        return {
                            'R2_mean': np.mean(cv_scores),
                            'R2_std': np.std(cv_scores),
                            'RMSE': np.sqrt(mean_squared_error(y_v, preds)),
                            'Bias': np.mean(preds - y_v),
                            'model': model,
                            'df_val': X_v.assign(REAL=y_v, PREDICCION=preds),
                            'n': len(data)
                        }

                    res_s = ejecutar_kfold(df_s, "Sucio")
                    res_l = ejecutar_kfold(df_l, "Limpio")

                    # --- REPORTE TÉCNICO ---
                    st.markdown("### 📊 Reporte de Estabilidad del Modelo")
                    
                    met_df = pd.DataFrame({
                        "Métrica": ["R² Promedio (Estabilidad)", "Desviación R² (Incertidumbre)", "Error (RMSE)", "Sesgo (Bias)", "Muestras"],
                        "Modelo Original": [f"{res_s['R2_mean']:.4f}", f"±{res_s['R2_std']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['Bias']:.4f}", res_s['n']],
                        "Modelo Optimizado": [f"{res_l['R2_mean']:.4f}", f"±{res_l['R2_std']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['Bias']:.4f}", res_l['n']],
                    })
                    st.table(met_df)

                    st.info(f"💡 **Interpretación:** El modelo limpio tiene una incertidumbre de ±{res_l['R2_std']:.4f}. Cuanto más bajo es este número, más confiable es el modelo ante cambios en el mineral.")

                    st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'db': df_l, 'res_s': res_s, 'res_l': res_l})
    
    # --- TAB 4 y 5: Mantenemos las visualizaciones ---
    with tabs[3]:
        if 'res_l' in st.session_state:
            ex = st.selectbox("Variable X:", st.session_state.feat)
            g1, g2 = st.columns(2)
            with g1: st.plotly_chart(px.scatter(st.session_state.res_s['df_val'], x=ex, y="REAL", trendline="ols", title="Original"), use_container_width=True)
            with g2: st.plotly_chart(px.scatter(st.session_state.res_l['df_val'], x=ex, y="REAL", trendline="ols", title="Limpio"), use_container_width=True)
    
    with tabs[4]:
        if 'mod' in st.session_state:
            inputs = {f: st.slider(f, float(st.session_state.db[f].min()), float(st.session_state.db[f].max()), float(st.session_state.db[f].mean())) for f in st.session_state.feat}
            pred = st.session_state.mod.predict(pd.DataFrame([inputs]))[0]
            st.metric(f"PREDICCIÓN {st.session_state.targ}", f"{pred:.2f}")

else:
    st.info("Sube un archivo para iniciar.")
