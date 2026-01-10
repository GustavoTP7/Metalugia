import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Pro: Alta Velocidad", layout="wide")
st.title("🏭 Sistema de Inteligencia Metalúrgica (Optimizado)")
st.markdown("---")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Carga de Información")
archivo = st.sidebar.file_uploader("Subir dataset (CSV o Excel)", type=["csv", "xlsx"])

if archivo:
    if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo.name:
        for key in list(st.session_state.keys()):
            if key != "ultimo_archivo":
                del st.session_state[key]
        st.session_state.ultimo_archivo = archivo.name

    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    t1, t2, t3, t4, t5 = st.tabs(["👁️ 1. Vista Previa", "🧹 2. Auditoría", "🛠️ 3. Entrenamiento (70/30)", "📊 4. Diagnóstico", "🎯 5. Simulador"])

    with t1:
        st.subheader("Inspección de Datos")
        st.dataframe(df.head(15), use_container_width=True)

    with t2:
        st.subheader("⚙️ Gestión de Calidad (Outliers)")
        cols_auditoria = st.multiselect("Variables a auditar (X e Y):", num_cols, default=num_cols[:min(3, len(num_cols))])
        
        indices_out = set()
        if cols_auditoria:
            for col in cols_auditoria:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
            st.session_state['borrar'] = list(indices_out)
            st.warning(f"Se han identificado {len(indices_out)} filas con anomalías.")

    with t3:
        st.subheader("🚀 Entrenamiento de Alta Velocidad")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])

        if st.button("🔥 Iniciar Modelamiento Pro (Parámetros Optimizados)", use_container_width=True):
            if not features:
                st.error("⚠️ Selecciona variables de entrada.")
            else:
                with st.spinner('Calculando...'):
                    df_s = df[[target] + features].dropna()
                    df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')
                    
                    # HIPERPARÁMETROS FIJOS OPTIMIZADOS (Balance Precisión/Velocidad)
                    # n_estimators=100 (Suficiente para convergencia)
                    # max_depth=6 (Captura complejidad sin sobreajustar)
                    # learning_rate=0.1 (Rápido y estable)
                    params = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42}

                    def motor_rapido(data):
                        X, y = data[features], data[target]
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        model = xgb.XGBRegressor(**params)
                        
                        # Validación Cruzada
                        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                        
                        # Partición 70/30
                        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.30, random_state=42)
                        model.fit(X_t, y_t)
                        p = model.predict(X_v)
                        
                        return {
                            'R2_CV': np.mean(cv_scores),
                            'R2_Test': r2_score(y_v, p),
                            'RMSE': np.sqrt(mean_squared_error(y_v, p)),
                            'Bias': np.mean(p - y_v),
                            'n': len(data),
                            'model': model,
                            'df_val': X_v.assign(REAL=y_v, PRED=p),
                            'importancia': pd.DataFrame({'Var': features, 'Imp': model.feature_importances_}).sort_values(by='Imp', ascending=True)
                        }

                    res_s = motor_rapido(df_s)
                    res_l = motor_rapido(df_l)

                    st.markdown("### 📊 Reporte 70/30")
                    res_df = pd.DataFrame({
                        "Métrica": ["R² Promedio (Estabilidad)", "R² Examen (30%)", "Error (RMSE)", "Sesgo (Bias)", "Filas"],
                        "Modelo Original": [f"{res_s['R2_CV']:.4f}", f"{res_s['R2_Test']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['Bias']:.4f}", res_s['n']],
                        "Modelo Limpio": [f"{res_l['R2_CV']:.4f}", f"{res_l['R2_Test']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['Bias']:.4f}", res_l['n']]
                    })
                    st.table(res_df)
                    st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'res_l': res_l, 'db_limpio': df_l})
                    st.success("¡Listo!")

    with t4:
        if 'res_l' in st.session_state:
            st.subheader("🧪 Diagnóstico")
            d1, d2 = st.columns(2)
            with d1:
                fig_imp = px.bar(st.session_state.res_l['importancia'], x='Imp', y='Var', orientation='h', title="Impacto de Variables", color='Imp', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)
            with d2:
                var_x = st.selectbox("Eje X:", st.session_state.feat)
                fig_scat = px.scatter(st.session_state.res_l['df_val'], x=var_x, y="REAL", trendline="ols", title="Test 30% (Real vs Pred)")
                st.plotly_chart(fig_scat, use_container_width=True)

    with t5:
        if 'mod' in st.session_state:
            st.subheader("🎯 Simulador")
            col_in, col_res = st.columns([1, 2])
            with col_in:
                in_data = {f: st.slider(f, float(st.session_state.db_limpio[f].min()), float(st.session_state.db_limpio[f].max()), float(st.session_state.db_limpio[f].mean())) for f in st.session_state.feat}
            with col_res:
                df_sim = pd.DataFrame([in_data])
                pred = st.session_state.mod.predict(df_sim)[0]
                st.metric(f"PREDICCIÓN {st.session_state.targ}", f"{pred:.2f}")
                sens = {f: st.session_state.mod.predict(df_sim.assign(**{f: in_data[f]*1.05}))[0] - pred for f in st.session_state.feat}
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Sensibilidad (+5%)", color=list(sens.values()), color_continuous_scale='RdYlGn'), use_container_width=True)
else:
    st.info("👋 Sube un archivo para comenzar.")
