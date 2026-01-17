import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Metalurgia Control Hub Pro", layout="wide")

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df.columns = df.columns.astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return None

st.title("🏭 Centro de Control Metalúrgico: Inteligencia K-Fold")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1️⃣ Gestión de Datos")
    archivo = st.file_uploader("Subir dataset", type=["csv", "xlsx"])
    
    modo_datos = st.radio("Selecciona modo de datos:", 
                         ["Dataset Original", "Sin Outliers (Auditado)"])
    
    factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5, 
                           disabled=(modo_datos == "Dataset Original"))
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            cols_todas = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.header("2️⃣ Configuración")
            col_id = st.selectbox("🔑 ID de Rastreo:", cols_todas)
            target = st.selectbox("🎯 Variable Objetivo (Y)", num_cols)
            features_raw = st.multiselect("🔍 Variables de Entrada (X)", [c for c in cols_todas if c not in [target, col_id]])
            
            st.divider()
            # Botón con instrucción clara de lo que hace internamente
            btn_entrenar = st.button("🚀 INICIAR AUDITORÍA K-FOLD (MULTI-CORE)", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO K-FOLD (PARALELIZADO) ---
if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features_raw:
        st.error("⚠️ Selecciona variables X.")
    else:
        status = st.empty()
        status.info("⏳ Ejecutando K-Fold en paralelo (n_jobs=-1)...")
        
        features = sorted(features_raw)
        df_base = df[[col_id, target] + features].dropna().copy()
        
        # Limpieza Outliers
        if modo_datos == "Sin Outliers (Auditado)":
            indices_out = set()
            for col in [target] + [f for f in features if df_base[f].dtype in [np.float64, np.int64]]:
                q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
            df_final = df_base.drop(list(indices_out), errors='ignore')
        else:
            df_final = df_base

        # Encoding de categorías
        X_encoded = df_final[features].copy()
        mapeos = {}
        for col in features:
            if X_encoded[col].dtype == 'object':
                cats = sorted(X_encoded[col].unique())
                mapeos[col] = cats
                X_encoded[col] = X_encoded[col].map({v: i for i, v in enumerate(cats)})

        # MODELO CON PARALELISMO ACTIVADO (n_jobs=-1)
        model = xgb.XGBRegressor(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            random_state=42, 
            n_jobs=-1 # Usa todos los núcleos disponibles
        )

        # ESTRATEGIA K-FOLD
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Predicción Out-of-Fold (también usa n_jobs del modelo)
        y_pred_kfold = cross_val_predict(model, X_encoded, df_final[target], cv=kf)
        
        # Entrenamos un modelo final para el simulador
        model_final = model.fit(X_encoded, df_final[target])

        # Métricas Globales
        r2_total = r2_score(df_final[target], y_pred_kfold)
        mae_total = mean_absolute_error(df_final[target], y_pred_kfold)
        rmse_total = np.sqrt(mean_squared_error(df_final[target], y_pred_kfold))
        bias_total = np.mean(y_pred_kfold - df_final[target])
        
        status.empty()

        st.session_state['res'] = {
            'r2': r2_total, 'mae': mae_total, 'rmse': rmse_total, 'bias': bias_total,
            'model': model_final, 'target': target, 'features': features, 'mapeos': mapeos,
            'df_work': df_final, 'preds': y_pred_kfold, 'col_id': col_id,
            'df_audit': pd.DataFrame({
                'ID_Turno': df_final[col_id],
                'Real': df_final[target],
                'Pred': y_pred_kfold,
                'Error': np.abs(df_final[target] - y_pred_kfold)
            }).sort_values('Error', ascending=False)
        }

# --- VISUALIZACIÓN ---
if 'res' in st.session_state:
    res = st.session_state['res']
    
    st.markdown(f"### 🛡️ Reporte Técnico K-Fold (N={len(res['df_work'])})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Estabilidad", f"{res['r2']:.4f}")
    m2.metric("MAE (Error Promedio)", f"{res['mae']:.3f}")
    m3.metric("RMSE (Riesgo)", f"{res['rmse']:.3f}")
    m4.metric("Bias (Sesgo)", f"{res['bias']:.4f}", delta_color="inverse")

    tabs = st.tabs(["📊 Gráficos de Correlación", "🚩 Auditoría por Turno", "🎯 Simulador de Planta", "👁️ Datos & Histogramas"])

    with tabs[0]: 
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Ejes de Análisis")
            opc = ["Objetivo Real", "Predicción IA"] + res['features']
            e_x = st.selectbox("Eje X:", opc, index=0)
            e_y = st.selectbox("Eje Y:", opc, index=1)
            
            st.divider()
            imp = pd.DataFrame({'V': res['features'], 'I': res['model'].feature_importances_}).sort_values('I')
            st.plotly_chart(px.bar(imp, x='I', y='V', orientation='h', title="Peso de Variables"), use_container_width=True)
        
        with c2:
            df_p = pd.DataFrame({res['col_id']: res['df_work'][res['col_id']]})
            def get_col(sel):
                if sel == "Objetivo Real": return "Real_Val", res['df_work'][res['target']]
                if sel == "Predicción IA": return "Pred_IA", res['preds']
                return sel, res['df_work'][sel]
            
            nx, dx = get_col(e_x); ny, dy = get_col(e_y)
            if nx == ny: ny += "_dup"
            df_p[nx], df_p[ny] = dx, dy
            
            st.plotly_chart(px.scatter(df_p, x=nx, y=ny, trendline="ols", hover_data=[res['col_id']], 
                             labels={nx: e_x, ny: e_y}, title=f"Correlación {e_x} vs {e_y}"), use_container_width=True)

    with tabs[1]: 
        st.subheader("Auditoría de Desviaciones")
        df_a = res['df_audit'].merge(res['df_work'], on='ID_Turno')
        df_a['Desv_%'] = (df_a['Error'] / df_a['Real']) * 100
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.write("Top 10 Desviaciones:")
            st.table(df_a[['ID_Turno', 'Real', 'Pred', 'Error', 'Desv_%']].head(10))
            csv = df_a.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Auditoría", csv, "auditoria_kfold.csv", "text/csv")
        with col_t2:
            st.plotly_chart(px.scatter(df_a, x='Real', y='Error', color='Error'), use_container_width=True)

    with tabs[2]: 
        st.subheader("Simulador What-If")
        c_in, c_out = st.columns(2)
        inputs = {}
        with c_in:
            for f in res['features']:
                if f in res['mapeos']:
                    sel = st.selectbox(f"Ajustar {f}", res['mapeos'][f])
                    inputs[f] = res['mapeos'][f].index(sel)
                else:
                    v_avg = float(res['df_work'][f].mean())
                    inputs[f] = st.slider(f, float(res['df_work'][f].min()), float(res['df_work'][f].max()), v_avg)
        with c_out:
            p_val = res['model'].predict(pd.DataFrame([inputs])[res['features']])[0]
            st.markdown(f"""
                <div style='background-color:#0E1117; padding:50px; border-radius:20px; border: 2px solid #00FF00; text-align:center'>
                    <h2 style='color:white'>PREDICCIÓN ESTIMADA</h2>
                    <h1 style='color:#00FF00; font-size:80px'>{p_val:.3f}</h1>
                    <p style='color:white; font-size:20px'>Confianza K-Fold (±{res['mae']:.2f})</p>
                </div>
            """, unsafe_allow_html=True)

    with tabs[3]: 
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.subheader("Dataset de Trabajo")
            st.dataframe(res['df_work'], use_container_width=True)
        with c_d2:
            st.subheader("Análisis de Distribución")
            v_h = st.selectbox("Variable:", [res['target']] + res['features'])
            st.plotly_chart(px.histogram(res['df_work'], x=v_h, marginal="box"), use_container_width=True)

elif archivo:
    st.info("👈 Configura los parámetros y presiona 'INICIAR'.")
