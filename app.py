import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Pro: Partición 70/30", layout="wide")
st.title("🏭 Modelamiento Predictivo: Configuración 70/30 + K-Fold")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Gestión de Datos")
archivo = st.sidebar.file_uploader("Subir dataset (CSV o Excel)", type=["csv", "xlsx"])

if archivo:
    # Reiniciar estado si se sube un archivo nuevo
    if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo.name:
        for key in ['mod', 'res_l', 'res_s', 'borrar']:
            if key in st.session_state: del st.session_state[key]
        st.session_state.ultimo_archivo = archivo.name

    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    t1, t2, t3, t4, t5 = st.tabs([
        "👁️ 1. Vista Previa", 
        "🧹 2. Auditoría de Outliers", 
        "🛠️ 3. Entrenamiento (70/30)", 
        "📊 4. Diagnóstico", 
        "🎯 5. Simulador"
    ])

    with t1:
        st.subheader("Inspección Inicial")
        st.dataframe(df.head(20), use_container_width=True)

    with t2:
        st.subheader("⚙️ Auditoría Multivariable (X e Y)")
        cols_auditoria = st.multiselect("Selecciona variables críticas para limpiar ruidos:", num_cols, default=num_cols[:min(3, len(num_cols))])
        
        indices_out = set()
        if cols_auditoria:
            for col in cols_auditoria:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
            st.session_state['borrar'] = list(indices_out)
            st.warning(f"Filas con anomalías detectadas: {len(indices_out)}")

    with t3:
        st.subheader("🚀 Entrenamiento con Partición 70% Train / 30% Test")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])

        st.divider()
        col_p1, col_p2, col_p3 = st.columns(3)
        n_trees = col_p1.slider("N° Árboles", 50, 500, 150)
        m_depth = col_p2.slider("Profundidad", 3, 10, 5)
        l_rate = col_p3.select_slider("Learning Rate", [0.01, 0.05, 0.1, 0.2], value=0.05)

        if st.button("🔥 Iniciar Entrenamiento Pro", use_container_width=True):
            if not features:
                st.error("Selecciona variables de entrada.")
            else:
                progreso = st.progress(0)
                status = st.empty()
                with st.spinner('Entrenando modelos de alta precisión...'):
                    df_s = df[[target] + features].dropna()
                    df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')
                    
                    def entrenar_modelo(data):
                        X, y = data[features], data[target]
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        model_base = xgb.XGBRegressor(n_estimators=n_trees, max_depth=m_depth, learning_rate=l_rate, random_state=42)
                        cv_scores = cross_val_score(model_base, X, y, cv=kf, scoring='r2')
                        
                        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.30, random_state=42)
                        model_base.fit(X_t, y_t)
                        p = model_base.predict(X_v)
                        
                        return {
                            'R2_CV': np.mean(cv_scores), 'R2_STD': np.std(cv_scores),
                            'R2_Test': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)),
                            'BIAS': np.mean(p - y_v), 'n': len(data), 'model': model_base,
                            'df_val': X_v.assign(REAL=y_v, PRED=p),
                            'importancia': pd.Series(model_base.feature_importances_, index=features).sort_values()
                        }

                    status.text("Procesando modelo original...")
                    res_s = entrenar_modelo(df_s)
                    progreso.progress(50)
                    
                    status.text("Procesando modelo auditado...")
                    res_l = entrenar_modelo(df_l)
                    progreso.progress(100)
                    status.text("¡Completado!")

                    st.markdown("### 📊 Reporte Final (70/30)")
                    res_df = pd.DataFrame({
                        "Métrica": ["R² Estabilidad (CV)", "R² Examen (Test 30%)", "Error (RMSE)", "Sesgo (Bias)", "Muestras"],
                        "Modelo Sucio": [f"{res_s['R2_CV']:.4f}", f"{res_s['R2_Test']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['BIAS']:.4f}", res_s['n']],
                        "Modelo Limpio": [f"{res_l['R2_CV']:.4f}", f"{res_l['R2_Test']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['BIAS']:.4f}", res_l['n']]
                    })
                    st.table(res_df)
                    st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'db': df_l, 'res_s': res_s, 'res_l': res_l})
                    st.toast("Modelo entrenado", icon="✅")

    with t4:
        if 'res_l' in st.session_state:
            st.subheader("🧪 Diagnóstico de Variables")
            d1, d2 = st.columns(2)
            with d1:
                # CORRECCIÓN: Usamos el objeto Series directamente
                st.write("**Importancia de Variables (Impacto):**")
                st.plotly_chart(px.bar(st.session_state.res_l['importancia'], orientation='h', color_discrete_sequence=['#2ecc71']), use_container_width=True)
            with d2:
                var_x = st.selectbox("Eje X para dispersión:", st.session_state.feat)
                st.plotly_chart(px.scatter(st.session_state.res_l['df_val'], x=var_x, y="REAL", trendline="ols", title="Correlación Test (30%)"), use_container_width=True)
        else:
            st.info("💡 Ve a la pestaña '3. Entrenamiento' y presiona el botón para generar este análisis.")

    with t5:
        if 'mod' in st.session_state:
            st.subheader("🎯 Simulador What-If")
            col_in, col_res = st.columns([1, 2])
            with col_in:
                input_data = {f: st.slider(f, float(df[f].min()), float(df[f].max()), float(df[f].mean())) for f in st.session_state.feat}
            with col_res:
                pred_val = st.session_state.mod.predict(pd.DataFrame([input_data]))[0]
                st.metric(f"PREDICCIÓN {st.session_state.targ}", f"{pred_val:.2f}")
                sens = {f: st.session_state.mod.predict(pd.DataFrame([input_data]).assign(**{f: input_data[f]*1.05}))[0] - pred_val for f in st.session_state.feat}
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Sensibilidad Operativa (+5%)"), use_container_width=True)
        else:
            st.info("💡 El simulador se activará automáticamente después de entrenar el modelo.")
else:
    st.info("👋 Sube un archivo Excel o CSV para comenzar el análisis metalúrgico.")
