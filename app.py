import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Metalurgia Pro: Consistencia Total", layout="wide")
st.title("🏭 Modelamiento Predictivo: Auditoría y Validación")

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
    df.columns = df.columns.astype(str).str.strip()
    return df

archivo = st.sidebar.file_uploader("Subir dataset", type=["csv", "xlsx"])

if archivo:
    if "ultimo_archivo" not in st.session_state or st.session_state.ultimo_archivo != archivo.name:
        st.session_state.clear() # Limpieza total al cambiar archivo
        st.session_state.ultimo_archivo = archivo.name

    df = cargar_datos(archivo)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    t1, t2, t3, t4, t5 = st.tabs(["👁️ Vista", "🧹 Auditoría", "🛠️ Entrenamiento", "📊 Diagnóstico", "🎯 Simulador"])

    with t2:
        st.subheader("🧹 Auditoría de Outliers (IQR Method)")
        cols_auditoria = st.multiselect("Variables críticas:", num_cols, default=num_cols[:min(3, len(num_cols))])
        indices_out = set()
        if cols_auditoria:
            for col in cols_auditoria:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                indices_out.update(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)].index)
        st.session_state['borrar'] = list(indices_out)
        st.info(f"Puntos anómalos detectados: {len(indices_out)}")

    with t3:
        st.subheader("🚀 Entrenamiento con Validación 70/30")
        c1, c2, c3 = st.columns([2, 2, 1])
        target = c1.selectbox("🎯 Objetivo (Y):", num_cols)
        features = c2.multiselect("🔍 Entradas (X):", [c for c in num_cols if c != target])
        tipo_modelo = c3.radio("Configuración:", ["Limpio (Auditado)", "Sucio (Original)"])

        if st.button(f"🔥 Ejecutar Modelo {tipo_modelo}", use_container_width=True):
            if not features:
                st.error("Selecciona variables X.")
            else:
                with st.spinner('Entrenando...'):
                    # 1. Preparación de Datos
                    df_base = df[[target] + features].dropna()
                    if tipo_modelo == "Limpio (Auditado)":
                        data_final = df_base.drop(st.session_state.get('borrar', []), errors='ignore')
                    else:
                        data_final = df_base

                    X, y = data_final[features], data_final[target]
                    
                    # 2. Configuración del Modelo (N-Jobs -1 para velocidad)
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, tree_method='hist', random_state=42)

                    # 3. K-Fold Cross Validation (Sobre el 100% del dataset elegido)
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
                    
                    # 4. Hold-out Split (70% Train / 30% Test)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # 5. Guardar todo sincronizado
                    st.session_state.update({
                        'mod': model, 'feat': features, 'targ': target, 'db_entrenada': data_final,
                        'res': {
                            'R2_CV': np.mean(cv_scores), 'R2_STD': np.std(cv_scores),
                            'R2_Test': r2_score(y_test, y_pred), 
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'n': len(data_final),
                            'df_test': X_test.assign(REAL=y_test, PRED=y_pred),
                            'importancia': pd.DataFrame({'Var': features, 'Imp': model.feature_importances_}).sort_values(by='Imp', ascending=True)
                        },
                        'modelo_ok': tipo_modelo
                    })

        if 'res' in st.session_state:
            st.success(f"Resultados para: {st.session_state.modelo_ok}")
            m = st.session_state.res
            c1, c2, c3 = st.columns(3)
            c1.metric("Estabilidad (CV R²)", f"{m['R2_CV']:.4f}", f"±{m['R2_STD']:.3f}")
            c2.metric("Precisión (Test 30% R²)", f"{m['R2_Test']:.4f}")
            c3.metric("Error (RMSE)", f"{m['RMSE']:.4f}")

    with t4:
        if 'res' in st.session_state:
            st.subheader(f"📊 Diagnóstico: {st.session_state.modelo_ok}")
            d1, d2 = st.columns(2)
            with d1:
                st.plotly_chart(px.bar(st.session_state.res['importancia'], x='Imp', y='Var', orientation='h', title="Peso de las Variables"), use_container_width=True)
            with d2:
                st.plotly_chart(px.scatter(st.session_state.res['df_test'], x="REAL", y="PRED", labels={'REAL':'Valor Real', 'PRED':'Predicción'}, title="Ajuste Real vs Predicho (Datos de Test)"), use_container_width=True)
        else:
            st.info("Entrena el modelo para ver diagnósticos.")

    with t5:
        if 'mod' in st.session_state:
            st.subheader(f"🎯 Simulador: {st.session_state.modelo_ok}")
            c_in, c_re = st.columns([1, 2])
            with c_in:
                # Sliders con valores dinámicos
                in_d = {f: st.slider(f, float(st.session_state.db_entrenada[f].min()), float(st.session_state.db_entrenada[f].max()), float(st.session_state.db_entrenada[f].mean())) for f in st.session_state.feat}
            with c_re:
                pred = st.session_state.mod.predict(pd.DataFrame([in_d]))[0]
                st.metric(f"RESULTADO ESTIMADO ({st.session_state.targ})", f"{pred:.2f}")
                
                # Gráfico de Sensibilidad (+5%)
                sens = {}
                for f in st.session_state.feat:
                    df_temp = pd.DataFrame([in_d])
                    df_temp[f] *= 1.05
                    sens[f] = st.session_state.mod.predict(df_temp)[0] - pred
                st.plotly_chart(px.bar(x=list(sens.values()), y=list(sens.keys()), orientation='h', title="Impacto por incremento del 5%", color=list(sens.values()), color_continuous_scale='RdYlGn'), use_container_width=True)
else:
    st.info("👋 Sube tu archivo CSV o Excel para empezar.")
