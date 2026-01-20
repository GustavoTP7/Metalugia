import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

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

st.title("🏭 Centro de Control Metalúrgico: Inteligencia en Tiempo Real")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1️⃣ Gestión de Datos")
    archivo = st.file_uploader("Subir dataset (CSV o XLSX)", type=["csv", "xlsx"])
    
    modo_datos = st.radio("Filtro de Ruido:", ["Dataset Original", "Sin Outliers (Auditado)"])
    factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5, disabled=(modo_datos == "Dataset Original"))
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            cols_todas = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.header("2️⃣ Configuración")
            col_id = st.selectbox("🔑 ID de Rastreo (Turno/Fecha):", cols_todas)
            target = st.selectbox("🎯 Variable Objetivo (Y)", num_cols)
            features_raw = st.multiselect("🔍 Variables de Entrada (X)", [c for c in cols_todas if c not in [target, col_id]])
            
            st.divider()
            btn_entrenar = st.button("🚀 INICIAR MONITOREO Y AUDITORÍA", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO K-FOLD + PARALELISMO ---
if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features_raw:
        st.error("⚠️ Debes seleccionar al menos una variable de entrada (X).")
    else:
        status = st.empty()
        status.info("⏳ Procesando validación cruzada en paralelo (n_jobs=-1)...")
        
        features = sorted(features_raw)
        df_base = df[[col_id, target] + features].dropna().copy()
        
        if modo_datos == "Sin Outliers (Auditado)":
            indices_out = set()
            for col in [target] + [f for f in features if df_base[f].dtype in [np.float64, np.int64]]:
                q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
            df_final = df_base.drop(list(indices_out), errors='ignore')
        else:
            df_final = df_base

        X_encoded = df_final[features].copy()
        mapeos = {}
        for col in features:
            if X_encoded[col].dtype == 'object':
                cats = sorted(X_encoded[col].unique())
                mapeos[col] = cats
                X_encoded[col] = X_encoded[col].map({v: i for i, v in enumerate(cats)})

        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_kfold = cross_val_predict(model, X_encoded, df_final[target], cv=kf)
        model_final = model.fit(X_encoded, df_final[target])

        idx_ultimo = df_final.index[-1]
        val_id_ultimo = df_final.loc[idx_ultimo, col_id]
        val_real_ultimo = df_final.loc[idx_ultimo, target]
        val_pred_ultimo = y_pred_kfold[-1]
        fila_ultimo_raw = df_final.iloc[-1]

        status.empty()

        st.session_state['res'] = {
            'r2': r2_score(df_final[target], y_pred_kfold),
            'mae': mean_absolute_error(df_final[target], y_pred_kfold),
            'rmse': np.sqrt(mean_squared_error(df_final[target], y_pred_kfold)),
            'bias': np.mean(y_pred_kfold - df_final[target]),
            'model': model_final, 'target': target, 'features': features, 'mapeos': mapeos,
            'df_work': df_final, 'preds': y_pred_kfold, 'col_id': col_id,
            'ultimo': {'id': val_id_ultimo, 'real': val_real_ultimo, 'pred': val_pred_ultimo, 'fila': fila_ultimo_raw}
        }

# --- INTERFAZ DE RESULTADOS ---
if 'res' in st.session_state:
    res = st.session_state['res']
    
    error_actual = abs(res['ultimo']['real'] - res['ultimo']['pred'])
    color_alerta = "#FF4B4B" if error_actual > res['mae'] else "#00FF00"
    
    st.markdown(f"""
        <div style="background-color:#1E1E1E; padding:25px; border-radius:15px; border-left: 10px solid {color_alerta}; margin-bottom:20px">
            <h2 style="color:white; margin:0">⚡ AUDITORÍA TURNO ACTUAL: <span style="color:{color_alerta}">{res['ultimo']['id']}</span></h2>
            <div style="display: flex; justify-content: space-around; padding-top:20px">
                <div style="text-align:center"><p style="color:#BBB; margin:0">Valor Preliminar</p><h1 style="color:white; margin:0">{res['ultimo']['real']:.2f}</h1></div>
                <div style="text-align:center"><p style="color:#BBB; margin:0">Predicción IA</p><h1 style="color:#00FF00; margin:0">{res['ultimo']['pred']:.2f}</h1></div>
                <div style="text-align:center"><p style="color:#BBB; margin:0">Desviación</p><h1 style="color:{color_alerta}; margin:0">{error_actual:.2f}</h1></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Estabilidad (K-Fold)", f"{res['r2']:.4f}")
    m2.metric("Error Promedio (MAE)", f"{res['mae']:.3f}")
    m3.metric("Riesgo (RMSE)", f"{res['rmse']:.3f}")
    m4.metric("Sesgo (Bias)", f"{res['bias']:.4f}", delta_color="inverse")

    tabs = st.tabs(["📊 Análisis de Tendencias", "🚩 Auditoría por Fila", "🎯 Simulador Proactivo", "👁️ Base de Datos"])

    with tabs[0]:
        col_ctrl, col_graph = st.columns([1, 2])
        with col_ctrl:
            opc_ejes = ["Objetivo Real", "Predicción IA"] + res['features']
            e_x = st.selectbox("Eje Horizontal (X):", opc_ejes, index=0)
            e_y = st.selectbox("Eje Vertical (Y):", opc_ejes, index=1)
            imp = pd.DataFrame({'Var': res['features'], 'Imp': res['model'].feature_importances_}).sort_values('Imp')
            st.plotly_chart(px.bar(imp, x='Imp', y='Var', orientation='h', title="Peso de Variables"), use_container_width=True)

        with col_graph:
            df_plot = pd.DataFrame({res['col_id']: res['df_work'][res['col_id']]})
            def get_data(sel):
                if sel == "Objetivo Real": return res['df_work'][res['target']]
                if sel == "Predicción IA": return res['preds']
                return res['df_work'][sel]
            
            df_plot['X'] = get_data(e_x)
            df_plot['Y'] = get_data(e_y)
            
            # --- SOLUCIÓN: LÍNEA DE TENDENCIA RESTAURADA ---
            # Usamos px.scatter con trendline sobre el dataset completo
            fig = px.scatter(df_plot, x='X', y='Y', hover_data=[res['col_id']], 
                             labels={'X':e_x, 'Y':e_y}, template="plotly_dark",
                             trendline="ols", trendline_color_override="cyan")
            
            # Superponemos el círculo del último turno para que no lo tape la línea
            fig.add_trace(go.Scatter(
                x=[df_plot['X'].iloc[-1]], 
                y=[df_plot['Y'].iloc[-1]],
                mode='markers', 
                marker=dict(color='Lime', size=12, symbol='circle', line=dict(width=2, color='white')),
                name='Turno Actual',
                showlegend=True
            ))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        df_audit = pd.DataFrame({
            'ID_Turno': res['df_work'][res['col_id']],
            'Real': res['df_work'][res['target']],
            'IA_Pred': res['preds'],
            'Error_Abs': np.abs(res['df_work'][res['target']] - res['preds'])
        }).sort_values('Error_Abs', ascending=False)
        st.dataframe(df_audit, use_container_width=True)
        st.download_button("📥 Descargar Reporte", df_audit.to_csv(index=False), "auditoria.csv")

    with tabs[2]:
        st.info(f"📍 Turno base: {res['ultimo']['id']}")
        cin, cout = st.columns(2)
        user_inputs = {}
        fila_actual = res['ultimo']['fila']
        with cin:
            for f in res['features']:
                if f in res['mapeos']:
                    opciones = res['mapeos'][f]
                    idx_ini = opciones.index(fila_actual[f])
                    sel = st.selectbox(f, opciones, index=idx_ini)
                    user_inputs[f] = opciones.index(sel)
                else:
                    v_min, v_max = float(res['df_work'][f].min()), float(res['df_work'][f].max())
                    v_ini = float(fila_actual[f])
                    user_inputs[f] = st.slider(f, v_min, v_max, v_ini)
        with cout:
            pred_escenario = res['model'].predict(pd.DataFrame([user_inputs])[res['features']])[0]
            impacto = pred_escenario - res['ultimo']['pred']
            st.markdown(f"""
                <div style="background-color:#0E1117; padding:60px; border-radius:20px; border: 3px solid #00FF00; text-align:center; margin-top:20px">
                    <h3 style="color:white">PROYECCIÓN</h3>
                    <h1 style="color:#00FF00; font-size:90px; margin:0">{pred_escenario:.3f}</h1>
                    <h3 style="color:{'#00FF00' if impacto >=0 else '#FF4B4B'}">Impacto: {impacto:+.3f}</h3>
                </div>
            """, unsafe_allow_html=True)

    with tabs[3]:
        st.dataframe(res['df_work'], use_container_width=True)
