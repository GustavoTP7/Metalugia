import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Metalurgia Pro: Control Total", layout="wide")
st.title("🏭 Modelamiento por Demanda (70/30 + K-Fold)")

# --- CARGA DE DATOS ---
archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo.name:
        for key in list(st.session_state.keys()):
            if key != "ultimo_archivo": del st.session_state[key]
        st.session_state.ultimo_archivo = archivo.name

    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    t1, t2, t3, t4, t5 = st.tabs(["👁️ Vista Previa", "🧹 Auditoría", "🛠️ Entrenamiento", "📊 Diagnóstico", "🎯 Simulador"])

    # --- TAB 1: VISTA PREVIA ---
    with t1:
        st.dataframe(df.head(10), use_container_width=True)

    # --- TAB 2: AUDITORÍA (X e Y) ---
    with t2:
        st.subheader("⚙️ Configuración de Auditoría")
        cols_auditoria = st.multiselect("Variables a auditar (Entradas y Objetivo):", num_cols, default=num_cols[:min(3, len(num_cols))])
        indices_out = set()
        if cols_auditoria:
            for col in cols_auditoria:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
        st.session_state['borrar'] = list(indices_out)
        st.info(f"Datos detectados como ruido: {len(indices_out)}")

    # --- TAB 3: ENTRENAMIENTO (70/30 + K-FOLD) ---
    with t3:
        st.subheader("🚀 Control de Entrenamiento")
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])
        tipo_modelo = c3.radio("📂 Dataset:", ["Limpio (Auditado)", "Sucio (Original)"])

        if st.button(f"🔥 Entrenar Modelo {tipo_modelo}", use_container_width=True):
            if not features:
                st.error("Selecciona variables X.")
            else:
                with st.spinner(f'Calculando {tipo_modelo}...'):
                    data_full = df[[target] + features].dropna()
                    if tipo_modelo == "Limpio (Auditado)":
                        data_final = data_full.drop(st.session_state.get('borrar', []), errors='ignore')
                    else:
                        data_final = data_full

                    X, y = data_final[features], data_final[target]
                    
                    # K-Fold 5 (Standard Industrial)
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                    
                    # Split 70/30 (Examen Riguroso)
                    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.30, random_state=42)
                    model.fit(X_t, y_t)
                    p = model.predict(X_v)

                    st.session_state.update({
                        'mod': model, 'feat': features, 'targ': target, 'db_entrenada': data_final,
                        'res': {
                            'R2_CV': np.mean(cv_scores), 'R2_STD': np.std(cv_scores),
                            'R2_Test': r2_score(y_v, p), 'RMSE': np.sqrt(mean_squared_error(y_v, p)),
                            'Bias': np.mean(p - y_v), 'n': len(data_final),
                            'df_val': X_v.assign(REAL=y_v, PRED=p),
                            'importancia': pd.DataFrame({'Var': features, 'Imp': model.feature_importances_}).sort_values(by='Imp', ascending=True)
                        },
                        'tipo_actual': tipo_modelo
                    })

        if 'res' in st.session_state:
            st.success(f"✅ Modelo {st.session_state.tipo_actual} finalizado.")
            m = st.session_state.res
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Estabilidad (CV)", f"{m['R2_CV']:.4f}", f"±{m['R2_STD']:.3f}")
            col_m2.metric("Precisión Examen (30%)", f"{m['R2_Test']:.4f}")
            col_m3.metric("Error RMSE", f"{m['RMSE']:.4f}")
            col_m4.metric("Sesgo (Bias)", f"{m['Bias']:.4f}")

    # --- TAB 4: DIAGNÓSTICO ---
    with t4:
        if 'res' in st.session_state:
            st.subheader(f"🧪 Diagnóstico: Modelo {st.session_state.tipo_actual}")
            d1, d2 = st.columns(2)
            with d1:
                st.plotly_chart(px.bar(st.session_state.res['importancia'], x='Imp', y='Var', orientation='h', 
                                      title="Impacto de Variables", color='Imp', color_continuous_scale='Viridis'), use_container_width=True)
                
            with d2:
                var_x = st.selectbox("Cruzar variable con resultado:", st.session_state.feat)
                st.plotly_chart(px.scatter(st.session_state.res['df_val'], x=var_x, y="REAL", trendline="ols", 
                                          title="Validación 30% (Predicho vs Real)"), use_container_width=True)
                

[Image of a scatter plot with a regression line]

        else:
            st.info("⚠️ Entrena un modelo para activar el diagnóstico.")

    # --- TAB 5: SIMULADOR ---
    with t5:
        if 'mod' in st.session_state:
            st.subheader(f"🎯 Simulador Operativo ({st.session_state.tipo_actual})")
            c_in, c_re = st.columns([1, 2])
            with c_in:
                in_d = {f: st.slider(f, float(st.session_state.db_entrenada[f].min()), 
                                     float(st.session_state.db_entrenada[f].max()), 
                                     float(st.session_state.db_entrenada[f].mean())) for f in st.session_state.feat}
            with c_re:
                pred = st.session_state.mod.predict(pd.DataFrame([in_d]))[0]
                st.metric(f"PREDICCIÓN {st.session_state.targ}", f"{pred:.2f}")
                sens = {f: st.session_state.mod.predict(pd.DataFrame([in_d]).assign(**{f: in_d[f]*1.05}))[0] - pred for f in st.session_state.feat}
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', 
                                      title="Sensibilidad Operativa (+5% impacto)", color=list(sens.values()), 
                                      color_continuous_scale='RdYlGn'), use_container_width=True)
                
else:
    st.info("👋 Sube un archivo para comenzar.")
