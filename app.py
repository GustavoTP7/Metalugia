import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="App Metalúrgica Pro", layout="wide")
st.title("🏭 Inteligencia de Procesos: Análisis de Impacto Total")

# --- CARGA DE DATOS ---
st.sidebar.header("📂 Entrada")
archivo = st.sidebar.file_uploader("Subir dataset (Excel o CSV)", type=["csv", "xlsx"])

if archivo:
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab_vista, tab_limpieza, tab_modelo, tab_simulador = st.tabs([
        "👁️ 1. Vista Previa", "🧹 2. Limpieza de Datos", "🛠️ 3. Entrenamiento e Impacto", "🎯 4. Panel de Control"
    ])

    with tab_vista:
        st.subheader("Inspección de Datos Brutos")
        st.dataframe(df.head(20), use_container_width=True)
        st.write(df.describe().T)

    with tab_limpieza:
        st.subheader("⚙️ Configuración de Limpieza")
        modo_limpieza = st.radio("Método de selección:", ["Automático (IQR Global)", "Manual (Selección en tabla)"])
        col_ref = st.selectbox("Variable de referencia para outliers:", num_cols)
        
        Q1, Q3 = df[col_ref].quantile(0.25), df[col_ref].quantile(0.75)
        IQR = Q3 - Q1
        inf, sup = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers_detectados = df[(df[col_ref] < inf) | (df[col_ref] > sup)].copy()

        if modo_limpieza == "Manual (Selección en tabla)":
            outliers_detectados['ELIMINAR'] = True
            edited = st.data_editor(outliers_detectados, use_container_width=True, hide_index=True)
            st.session_state['indices_borrar'] = edited[edited['ELIMINAR'] == True].index
        else:
            st.session_state['indices_borrar'] = outliers_detectados.index
            st.success(f"Modo Automático: {len(outliers_detectados)} filas identificadas para eliminar.")

    # --- TAB 3: ENTRENAMIENTO E IMPACTO (MÉTRICAS + GRÁFICOS) ---
    with tab_modelo:
        st.subheader("🛠️ Evaluación: Con Outliers vs Sin Outliers")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Variable Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Variables de Entrada (X):", [c for c in num_cols if c != target])
        
        if st.button("🚀 Entrenar y Comparar Impacto", use_container_width=True):
            if not features:
                st.error("Selecciona variables de entrada.")
            else:
                # Preparar datos
                df_sucio = df[[target] + features].dropna()
                df_limpio = df_sucio.drop(st.session_state.get('indices_borrar', []), errors='ignore')

                def entrenar_full(data):
                    X, y = data[features], data[target]
                    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
                    m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.06)
                    m.fit(X_t, y_t)
                    p = m.predict(X_v)
                    # Métricas
                    return {
                        'R2': r2_score(y_v, p),
                        'MAE': mean_absolute_error(y_v, p),
                        'RMSE': np.sqrt(mean_squared_error(y_v, p)),
                        'Bias': np.mean(p - y_v),
                        'n': len(data),
                        'y_real': y_v, 'y_pred': p, 'model': m
                    }

                res_s = entrenar_full(df_sucio)
                res_l = entrenar_full(df_limpio)

                # --- 1. CUADRO DE MÉTRICAS COMPARATIVAS ---
                st.markdown("### 📊 Comparativa de Métricas Técnicas")
                
                # Creamos un DataFrame para mostrar la tabla de métricas
                met_df = pd.DataFrame({
                    'Métrica': ['R² Precisión', 'MAE (Error Promedio)', 'RMSE (Riesgo)', 'Bias (Sesgo)', 'Muestras'],
                    'Modelo Sucio (Original)': [
                        f"{res_s['R2']:.4f}", f"{res_s['MAE']:.4f}", f"{res_s['RMSE']:.4f}", f"{res_s['Bias']:.4f}", res_s['n']
                    ],
                    'Modelo Limpio (Filtrado)': [
                        f"{res_l['R2']:.4f}", f"{res_l['MAE']:.4f}", f"{res_l['RMSE']:.4f}", f"{res_l['Bias']:.4f}", res_l['n']
                    ]
                })
                st.table(met_df)

                # --- 2. GRÁFICOS COMPARATIVOS ---
                st.markdown("### 📈 Visualización del Ajuste")
                g1, g2 = st.columns(2)

                def plot_comparativo(res, titulo, color):
                    fig = px.scatter(x=res['y_real'], y=res['y_pred'], labels={'x':'Real', 'y':'Predicho'},
                                     title=titulo, opacity=0.5, color_discrete_sequence=[color], trendline="ols")
                    # Línea de identidad (45 grados)
                    fig.add_shape(type="line", x0=res['y_real'].min(), y0=res['y_real'].min(),
                                  x1=res['y_real'].max(), y1=res['y_real'].max(),
                                  line=dict(color="Red", dash="dash"))
                    return fig

                with g1:
                    st.plotly_chart(plot_comparativo(res_s, "Modelo CON Outliers", "gray"), use_container_width=True)
                with g2:
                    st.plotly_chart(plot_comparativo(res_l, "Modelo SIN Outliers", "#2ecc71"), use_container_width=True)

                st.session_state.update({'mod': res_l['model'], 'feat': features, 'targ': target, 'db': df_limpio})

    # --- TAB 4: PANEL DE CONTROL ---
    with tab_simulador:
        if 'mod' in st.session_state:
            st.subheader("🎯 Simulador de Operación")
            col_sl, col_ga = st.columns([1, 2])
            with col_sl:
                inputs = {f: st.slider(f, float(st.session_state.db[f].min()), float(st.session_state.db[f].max()), float(st.session_state.db[f].mean())) for f in st.session_state.feat}
            
            pred = st.session_state.mod.predict(pd.DataFrame([inputs]))[0]
            with col_ga:
                st.metric(f"PREDICCIÓN DE {st.session_state.targ}", f"{pred:.2f}")
                sens = {}
                for f in st.session_state.feat:
                    df_p = pd.DataFrame([inputs])
                    df_p[f] += (st.session_state.db[f].max() - st.session_state.db[f].min()) * 0.05
                    sens[f] = st.session_state.mod.predict(df_p)[0] - pred
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Sensibilidad Local", color=list(sens.values()), color_continuous_scale="RdYlGn"))
        else:
            st.warning("⚠️ Primero entrena el modelo en la pestaña anterior.")
else:
    st.info("👋 Sube un archivo Excel o CSV para comenzar.")
