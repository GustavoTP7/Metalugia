import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

st.set_page_config(page_title="Metalurgia Control Hub", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    return df

st.title("🏭 Centro de Simulación y Modelamiento")

archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- PESTAÑAS PRINCIPALES ---
    tab_datos, tab_entrenamiento, tab_diagnostico, tab_simulador = st.tabs([
        "👁️ Visor de Datos", "🛠️ Entrenamiento", "📊 Diagnóstico Ejes", "🎯 Simulador"
    ])

    # --- 1. VISOR DE DATOS ---
    with tab_datos:
        st.subheader("Inspección del Dataset")
        st.dataframe(df, use_container_width=True)
        st.write(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    # --- 2. CONFIGURACIÓN Y ENTRENAMIENTO ---
    with tab_entrenamiento:
        st.subheader("Configuración del Modelo")
        c1, c2 = st.columns(2)
        target = c1.selectbox("🎯 Variable Objetivo (Y)", num_cols)
        features = c2.multiselect("🔍 Variables de Entrada (X)", [c for c in num_cols if c != target])
        
        factor_iqr = st.sidebar.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5)

        if st.button("🔥 Entrenar y Comparar Modelos", use_container_width=True):
            if not features:
                st.error("Selecciona variables X.")
            else:
                # Auditoría
                indices_out = set()
                for col in [target] + features:
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    indices_out.update(df[(df[col] < q1 - factor_iqr*iqr) | (df[col] > q3 + factor_iqr*iqr)].index)
                
                df_orig = df[[target] + features].dropna()
                df_sin = df_orig.drop(list(indices_out), errors='ignore')

                def procesar(data, etiqueta):
                    X, y = data[features], data[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, tree_method='hist', random_state=42)
                    
                    # Validación Cruzada
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    return {
                        "cv": np.mean(cv), "test": r2_score(y_test, pred), "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                        "model": model, "df_test": pd.DataFrame({'Real': y_test, 'Pred': pred}),
                        "imp": pd.DataFrame({'Var': features, 'Imp': model.feature_importances_})
                    }

                st.session_state['res_o'] = procesar(df_orig, "ORIGINAL")
                st.session_state['res_s'] = procesar(df_sin, "SIN OUTLIERS")
                st.session_state['features'] = features
                st.session_state['target'] = target
                st.session_state['data_ready'] = df_sin # Usamos la limpia para el simulador

        if 'res_o' in st.session_state:
            m1, m2 = st.columns(2)
            m1.metric("Original (Test R²)", f"{st.session_state.res_o['test']:.4f}", f"RMSE: {st.session_state.res_o['rmse']:.2f}", delta_color="inverse")
            m2.metric("Sin Outliers (Test R²)", f"{st.session_state.res_s['test']:.4f}", f"RMSE: {st.session_state.res_s['rmse']:.2f}", delta_color="inverse")

    # --- 3. DIAGNÓSTICO CON EJES ---
    with tab_diagnostico:
        if 'res_s' in st.session_state:
            st.subheader("Explorador de Correlaciones y Ajuste")
            col_x = st.selectbox("Eje X para visualizar:", st.session_state.features)
            
            # Gráfico de dispersión con variables reales vs predichas
            fig = px.scatter(st.session_state.res_s['df_test'], x='Real', y='Pred', 
                             trendline="ols", title="Ajuste Final del Modelo")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Peso de Variables")
            st.plotly_chart(px.bar(st.session_state.res_s['imp'].sort_values('Imp'), x='Imp', y='Var', orientation='h'), use_container_width=True)
        else:
            st.info("Entrena el modelo primero.")

    # --- 4. SIMULADOR EN TIEMPO REAL ---
    with tab_simulador:
        if 'res_s' in st.session_state:
            st.subheader("🎯 Simulador de Proceso (Escenario What-If)")
            st.write(f"Modifica las variables para predecir **{st.session_state.target}**")
            
            c_input, c_result = st.columns([1, 1])
            
            inputs = {}
            with c_input:
                for f in st.session_state.features:
                    min_val = float(st.session_state.data_ready[f].min())
                    max_val = float(st.session_state.data_ready[f].max())
                    mean_val = float(st.session_state.data_ready[f].mean())
                    inputs[f] = st.slider(f"Ajustar {f}", min_val, max_val, mean_val)
            
            with c_result:
                # Predicción en tiempo real
                input_df = pd.DataFrame([inputs])
                prediction = st.session_state.res_s['model'].predict(input_df)[0]
                
                st.markdown(f"""
                <div style="background-color:#1E1E1E; padding:20px; border-radius:10px; border: 2px solid #00FF00; text-align:center">
                    <h2 style="color:white; margin:0">VALOR ESTIMADO</h2>
                    <h1 style="color:#00FF00; font-size:50px; margin:10px">{prediction:.3f}</h1>
                    <p style="color:gray">{st.session_state.target}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico de impacto local (Sensibilidad)
                st.write("---")
                st.write("🔍 **Análisis de Sensibilidad (+10%)**")
                sensibilidad = []
                for f in st.session_state.features:
                    temp_df = input_df.copy()
                    temp_df[f] = temp_df[f] * 1.10
                    new_pred = st.session_state.res_s['model'].predict(temp_df)[0]
                    sensibilidad.append({'Variable': f, 'Impacto': new_pred - prediction})
                
                fig_sens = px.bar(pd.DataFrame(sensibilidad), x='Impacto', y='Variable', orientation='h', 
                                 color='Impacto', color_continuous_scale='RdYlGn', title="Cambio en Y si aumento X en 10%")
                st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("Entrena el modelo primero para activar el simulador.")

else:
    st.info("👋 Sube un archivo para comenzar.")
