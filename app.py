import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Pro: Optimización 70/30", layout="wide")
st.title("🏭 Sistema de Inteligencia Metalúrgica")
st.markdown("---")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Carga de Información")
archivo = st.sidebar.file_uploader("Subir dataset (CSV o Excel)", type=["csv", "xlsx"])

if archivo:
    # Reinicio inteligente si el archivo cambia
    if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo.name:
        for key in list(st.session_state.keys()):
            if key != "ultimo_archivo":
                del st.session_state[key]
        st.session_state.ultimo_archivo = archivo.name

    # Lectura de datos
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Definición de pestañas
    t1, t2, t3, t4, t5 = st.tabs([
        "👁️ 1. Vista Previa", 
        "🧹 2. Auditoría", 
        "🛠️ 3. Entrenamiento (70/30)", 
        "📊 4. Diagnóstico", 
        "🎯 5. Simulador"
    ])

    # --- 1. VISTA PREVIA ---
    with t1:
        st.subheader("Inspección de Datos")
        st.dataframe(df.head(15), use_container_width=True)
        st.write("**Resumen Estadístico:**", df.describe())

    # --- 2. AUDITORÍA (Outliers Multivariable) ---
    with t2:
        st.subheader("⚙️ Gestión de Calidad de Datos")
        st.info("Audita entradas (X) y objetivo (Y) para eliminar ruidos de sensores.")
        cols_auditoria = st.multiselect("Variables a auditar:", num_cols, default=num_cols[:min(3, len(num_cols))])
        
        indices_out = set()
        if cols_auditoria:
            for col in cols_auditoria:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                bajo, alto = q1 - 1.5*iqr, q3 + 1.5*iqr
                indices_out.update(df[(df[col] < bajo) | (df[col] > alto)].index)
            
            st.session_state['borrar'] = list(indices_out)
            st.warning(f"Se han identificado {len(indices_out)} filas con anomalías globales.")

    # --- 3. ENTRENAMIENTO (70/30 + K-FOLD) ---
    with t3:
        st.subheader("🚀 Entrenamiento de Alta Precisión")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])

        st.divider()
        st.write("🔧 **Parámetros de Optimización:**")
        col_a, col_b, col_c = st.columns(3)
        n_trees = col_a.slider("Cantidad de Árboles", 50, 500, 150)
        m_depth = col_b.slider("Complejidad (Profundidad)", 3, 10, 5)
        l_rate = col_c.select_slider("Tasa de Aprendizaje", [0.01, 0.05, 0.1, 0.2], value=0.05)

        if st.button("🔥 Ejecutar Modelamiento Pro", use_container_width=True):
            if not features:
                st.error("⚠️ Debes seleccionar al menos una variable de entrada (X).")
            else:
                with st.spinner('Procesando algoritmos y validación cruzada...'):
                    # Preparación de datos
                    df_s = df[[target] + features].dropna()
                    df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')
                    
                    def motor_entrenamiento(data):
                        X, y = data[features], data[target]
                        # K-Fold (5 particiones)
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        model = xgb.XGBRegressor(n_estimators=n_trees, max_depth=m_depth, learning_rate=l_rate, random_state=42)
                        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                        
                        # Partición 70/30 para examen final
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

                    res_s = motor_entrenamiento(df_s)
                    res_l = motor_entrenamiento(df_l)

                    # Mostrar Resultados
                    st.markdown("### 📊 Reporte de Performance (70/30)")
                    res_df = pd.DataFrame({
                        "Métrica": ["R² Promedio (CV)", "R² Examen (Test 30%)", "Error (RMSE)", "Sesgo (Bias)", "Filas Utilizadas"],
                        "Modelo Original": [f"{res_s['R2_CV']:.4f}", f"{res_s['R2_Test']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['Bias']:.4f}", res_s['n']],
                        "Modelo Limpio": [f"{res_l['R2_CV']:.4f}", f"{res_l['R2_Test']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['Bias']:.4f}", res_l['n']]
                    })
                    st.table(res_df)
                    
                    # Guardar en estado
                    st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'res_l': res_l, 'res_s': res_s, 'db_limpio': df_l})
                    st.success("¡Modelo entrenado con éxito!")

    # --- 4. DIAGNÓSTICO ---
    with t4:
        if 'res_l' in st.session_state:
            st.subheader("🧪 Análisis de Sensibilidad y Error")
            d1, d2 = st.columns(2)
            with d1:
                st.write("**Importancia Relativa de Variables:**")
                # Gráfico corregido
                fig_imp = px.bar(st.session_state.res_l['importancia'], x='Imp', y='Var', orientation='h', 
                                 title="Variables Críticas", color='Imp', color_continuous_scale='Viridis')
                st.plotly_chart(fig_imp, use_container_width=True)
            with d2:
                var_x = st.selectbox("Analizar dispersión por:", st.session_state.feat)
                fig_scat = px.scatter(st.session_state.res_l['df_val'], x=var_x, y="REAL", trendline="ols", 
                                     title=f"Precisión en Test (30%): {var_x} vs {target}")
                st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("⚠️ Los diagnósticos aparecerán aquí después de entrenar en la pestaña 3.")

    # --- 5. SIMULADOR ---
    with t5:
        if 'mod' in st.session_state:
            st.subheader("🎯 Simulador de Operación")
            col_in, col_res = st.columns([1, 2])
            with col_in:
                st.write("**Condiciones de Proceso:**")
                in_data = {}
                for f in st.session_state.feat:
                    v_min = float(st.session_state.db_limpio[f].min())
                    v_max = float(st.session_state.db_limpio[f].max())
                    v_mean = float(st.session_state.db_limpio[f].mean())
                    in_data[f] = st.slider(f, v_min, v_max, v_mean)
            
            with col_res:
                df_sim = pd.DataFrame([in_data])
                pred = st.session_state.mod.predict(df_sim)[0]
                st.metric(f"PREDICCIÓN {st.session_state.targ}", f"{pred:.2f}")
                
                # Gráfico de impacto (+5%)
                sens = {}
                for f in st.session_state.feat:
                    df_t = df_sim.copy()
                    df_t[f] = df_t[f] * 1.05
                    sens[f] = st.session_state.mod.predict(df_t)[0] - pred
                
                fig_sens = px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', 
                                  title="Impacto en el resultado ante +5% de cambio",
                                  labels={'x':'Cambio en Predicción', 'y':'Variable'},
                                  color=list(sens.values()), color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("⚠️ El simulador se activará automáticamente al finalizar el entrenamiento.")

else:
    st.info("👋 Por favor, sube un archivo Excel o CSV para comenzar.")
