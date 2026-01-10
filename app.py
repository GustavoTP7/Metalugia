import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

st.set_page_config(page_title="Metalurgia Pro: Turbo Edition", layout="wide")
st.title("🏭 Modelamiento Ultra-Rápido")

# Carga de datos optimizada con cache
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    return df

archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    df = cargar_datos(archivo)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    t1, t2, t3, t4, t5 = st.tabs(["👁️ Vista", "🧹 Auditoría", "🛠️ Entrenamiento", "📊 Diagnóstico", "🎯 Simulador"])

    with t2:
        st.subheader("🧹 Auditoría veloz")
        cols_auditoria = st.multiselect("Variables a auditar:", num_cols, default=num_cols[:min(3, len(num_cols))])
        indices_out = set()
        if cols_auditoria:
            for col in cols_auditoria:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
        st.session_state['borrar'] = list(indices_out)
        st.write(f"Filas marcadas para limpieza: {len(indices_out)}")

    with t3:
        st.subheader("🚀 Entrenamiento Optimizado")
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])
        tipo_modelo = c3.radio("Dataset:", ["Limpio", "Sucio"])

        if st.button(f"🔥 Entrenar ahora", use_container_width=True):
            if not features:
                st.error("Faltan variables.")
            else:
                with st.spinner('Entrenando a máxima velocidad...'):
                    data_full = df[[target] + features].dropna()
                    if tipo_modelo == "Limpio":
                        data_final = data_full.drop(st.session_state.get('borrar', []), errors='ignore')
                    else:
                        data_final = data_full

                    X, y = data_final[features], data_final[target]
                    
                    # CONFIGURACIÓN TURBO: n_jobs=-1 y tree_method='hist'
                    model = xgb.XGBRegressor(
                        n_estimators=100, 
                        max_depth=6, 
                        learning_rate=0.1, 
                        n_jobs=-1,           # USA TODOS LOS NÚCLEOS
                        tree_method='hist',  # MÉTODO MÁS RÁPIDO
                        random_state=42
                    )

                    # Validación Cruzada
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                    
                    # 70/30 Split
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
            st.success(f"Modelo {st.session_state.tipo_actual} entrenado.")
            m = st.session_state.res
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Estabilidad (CV)", f"{m['R2_CV']:.4f}")
            col_m2.metric("Precisión (30%)", f"{m['R2_Test']:.4f}")
            col_m3.metric("Error RMSE", f"{m['RMSE']:.4f}")
            col_m4.metric("Bias", f"{m['Bias']:.4f}")

    # (El resto de pestañas se mantienen igual que el código anterior)
    with t4:
        if 'res' in st.session_state:
            st.plotly_chart(px.bar(st.session_state.res['importancia'], x='Imp', y='Var', orientation='h', title="Importancia"), use_container_width=True)
            var_x = st.selectbox("Eje X:", st.session_state.feat)
            st.plotly_chart(px.scatter(st.session_state.res['df_val'], x=var_x, y="REAL", trendline="ols"), use_container_width=True)

    with t5:
        if 'mod' in st.session_state:
            in_d = {f: st.slider(f, float(st.session_state.db_entrenada[f].min()), float(st.session_state.db_entrenada[f].max()), float(st.session_state.db_entrenada[f].mean())) for f in st.session_state.feat}
            pred = st.session_state.mod.predict(pd.DataFrame([in_d]))[0]
            st.metric(f"PREDICCIÓN", f"{pred:.2f}")

else:
    st.info("Sube un archivo.")
