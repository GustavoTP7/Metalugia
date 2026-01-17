import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

st.title("🏭 Centro de Control Metalúrgico: Auditoría & IA")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("1️⃣ Gestión de Datos")
    archivo = st.file_uploader("Subir dataset", type=["csv", "xlsx"])
    
    modo_datos = st.radio("Selecciona modo de datos:", 
                         ["Dataset Original", "Sin Outliers (Auditado)"],
                         help="El modo 'Sin Outliers' elimina ruido usando el método IQR.")
    
    factor_iqr = st.slider("Sensibilidad Outliers (IQR)", 1.0, 3.0, 1.5, 
                           disabled=(modo_datos == "Dataset Original"))
    
    st.divider()
    
    if archivo:
        df = cargar_datos(archivo)
        if df is not None:
            cols_todas = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            st.header("2️⃣ Configuración")
            col_id = st.selectbox("🔑 ID de Rastreo (Fecha_Turno):", cols_todas)
            target = st.selectbox("🎯 Variable Objetivo (Y)", num_cols)
            features_raw = st.multiselect("🔍 Variables de Entrada (X)", [c for c in cols_todas if c not in [target, col_id]])
            
            st.divider()
            btn_entrenar = st.button("🚀 ENTRENAR Y AUDITAR", use_container_width=True, type="primary")

# --- LÓGICA DE PROCESAMIENTO ---
if archivo and 'btn_entrenar' in locals() and btn_entrenar:
    if not features_raw:
        st.error("⚠️ Selecciona variables X.")
    else:
        with st.spinner('Procesando inteligencia de planta...'):
            # 1. ORDENAMIENTO ALFABÉTICO (Blindaje)
            features = sorted(features_raw)
            
            df_base = df[[col_id, target] + features].dropna().copy()
            
            # 2. LIMPIEZA DE OUTLIERS
            if modo_datos == "Sin Outliers (Auditado)":
                indices_out = set()
                for col in [target] + [f for f in features if df_base[f].dtype in [np.float64, np.int64]]:
                    q1, q3 = df_base[col].quantile(0.25), df_base[col].quantile(0.75)
                    iqr = q3 - q1
                    indices_out.update(df_base[(df_base[col] < q1 - factor_iqr*iqr) | (df_base[col] > q3 + factor_iqr*iqr)].index)
                df_final = df_base.drop(list(indices_out), errors='ignore')
                etiqueta = "AUDITADO (SIN OUTLIERS)"
            else:
                df_final = df_base
                etiqueta = "ORIGINAL (CON RUIDO)"

            # 3. ENCODING (Variables Categóricas)
            X_encoded = df_final[features].copy()
            mapeos = {}
            for col in features:
                if X_encoded[col].dtype == 'object':
                    cats = sorted(X_encoded[col].unique())
                    mapeos[col] = cats
                    X_encoded[col] = X_encoded[col].map({v: i for i, v in enumerate(cats)})

            # 4. ENTRENAMIENTO
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, df_final[target], test_size=0.3, random_state=42)
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            
            test_preds = model.predict(X_test)
            all_preds = model.predict(X_encoded)

            st.session_state['res'] = {
                'test': r2_score(y_test, test_preds), 
                'rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
                'model': model, 'target': target, 'features': features, 'mapeos': mapeos,
                'df_work': df_final, 'modo': etiqueta, 'col_id': col_id,
                'all_preds': all_preds,
                'df_audit': pd.DataFrame({
                    'ID_Turno': df_final.loc[y_test.index, col_id],
                    'Real': y_test.values, 'Pred': test_preds, 
                    'Error': np.abs(y_test.values - test_preds)
                }).sort_values('Error', ascending=False)
            }

# --- VISUALIZACIÓN ---
if 'res' in st.session_state:
    res = st.session_state['res']
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión (R²)", f"{res['test']:.4f}")
    m2.metric("Error (RMSE)", f"{res['rmse']:.3f}")
    m3.metric("Modo", res['modo'])
    m4.metric("Datos Usados", len(res['df_work']))

    tabs = st.tabs(["📊 Diagnóstico Ejes", "🚩 Auditoría 360°", "🎯 Simulador Pro", "👁️ Datos & Histogramas"])

    with tabs[0]:
        cx1, cx2 = st.columns([1, 2])
        with cx1:
            st.subheader("Configuración")
            opciones_ejes = ["Objetivo Real", "Predicción IA"] + res['features']
            eje_x = st.selectbox("Eje X:", opciones_ejes, index=0)
            eje_y = st.selectbox("Eje Y:", opciones_ejes, index=1)
            
            st.divider()
            st.subheader("Sensibilidad")
            imp_df = pd.DataFrame({'Var': res['features'], 'Imp': res['model'].feature_importances_}).sort_values('Imp')
            st.plotly_chart(px.bar(imp_df, x='Imp', y='Var', orientation='h', height=300), use_container_width=True)

        with cx2:
            # PROTECCIÓN CONTRA DUPLICADOS (MISMA VARIABLE X e Y)
            df_plot = pd.DataFrame(index=res['df_work'].index)
            df_plot[res['col_id']] = res['df_work'][res['col_id']]
            
            def get_data(sel):
                if sel == "Objetivo Real": return "Real_Target", res['df_work'][res['target']]
                if sel == "Predicción IA": return "IA_Pred", res['all_preds']
                return sel, res['df_work'][sel]

            nx, dx = get_data(eje_x)
            ny, dy = get_data(eje_y)

            if nx == ny: ny = f"{ny}_dup" # Evita DuplicateError

            df_plot[nx] = dx
            df_plot[ny] = dy

            fig_sc = px.scatter(df_plot, x=nx, y=ny, hover_data=[res['col_id']], trendline="ols",
                                labels={nx: eje_x, ny: eje_y}, title=f"Análisis: {eje_x} vs {eje_y}")
            st.plotly_chart(fig_sc, use_container_width=True)

    with tabs[1]:
        st.subheader("🚩 Radar de Desviaciones Críticas")
        df_audit_full = res['df_audit'].merge(res['df_work'], left_on='ID_Turno', right_on=res['col_id'], how='left')
        df_audit_full['Desviación_%'] = (df_audit_full['Error'] / df_audit_full['Real']) * 100
        
        c1, c2 = st.columns(2)
        with c1:
            st.table(df_audit_full[['ID_Turno', 'Real', 'Pred', 'Error', 'Desviación_%']].head(10))
            csv = df_audit_full.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Reporte", csv, "auditoria.csv", "text/csv")
        with c2:
            st.plotly_chart(px.scatter(df_audit_full, x='Real', y='Error', size='Error', color='Error'), use_container_width=True)
        
        st.divider()
        v_err = st.selectbox("Error vs Variable:", res['features'])
        st.plotly_chart(px.scatter(df_audit_full, x=v_err, y='Error', trendline="ols", color_discrete_sequence=['#FF4B4B']), use_container_width=True)

    with tabs[2]:
        st.subheader("Simulador What-If")
        cin, cout = st.columns(2)
        inputs = {}
        with cin:
            for f in res['features']:
                if f in res['mapeos']:
                    opc = res['mapeos'][f]
                    sel = st.selectbox(f"Seleccionar {f}", opc)
                    inputs[f] = opc.index(sel)
                else:
                    v_min, v_max = float(res['df_work'][f].min()), float(res['df_work'][f].max())
                    inputs[f] = st.slider(f, v_min, v_max, float(res['df_work'][f].mean()))
        with cout:
            df_input = pd.DataFrame([inputs])[res['features']] # Blindaje de orden
            pred_val = res['model'].predict(df_input)[0]
            st.markdown(f"<div style='background-color:#0E1117; padding:40px; border-radius:15px; border: 2px solid #00FF00; text-align:center'><h2 style='color:white'>PREDICCIÓN {res['target']}</h2><h1 style='color:#00FF00; font-size:70px'>{pred_val:.3f}</h1><p style='color:white'>±{res['rmse']:.2f}</p></div>", unsafe_allow_html=True)

    with tabs[3]:
        ct1, ct2 = st.columns(2)
        with ct1:
            st.subheader("Visor de Datos")
            st.dataframe(res['df_work'], use_container_width=True)
        with ct2:
            st.subheader("Histogramas")
            v_h = st.selectbox("Histograma de:", [res['target']] + res['features'])
            st.plotly_chart(px.histogram(res['df_work'], x=v_h, nbins=35, marginal="box"), use_container_width=True)

elif archivo:
    st.info("👈 Selecciona variables y entrena.")
