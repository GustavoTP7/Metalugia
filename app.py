# --- TAB 3: ENTRENAMIENTO PRO CON SLIDER DE PARTICIÓN ---
with tab3:
    st.subheader("🚀 Configuración del Motor de Inteligencia")
    c1, c2 = st.columns(2)
    target = c1.selectbox("🎯 Variable Objetivo (Y):", num_cols)
    features = c2.multiselect("🔍 Variables de Entrada (X):", [c for c in num_cols if c != target])

    st.write("🔧 **Ajuste de Validación y Red:**")
    col_t1, col_t2, col_t3 = st.columns(3)
    
    # NUEVO: Slider para elegir el tamaño del Test (examen)
    test_size_pct = col_t1.slider("Porcentaje de Test (%):", 10, 40, 20)
    m_depth = col_t2.slider("Profundidad del Árbol:", 3, 10, 5)
    l_rate = col_t3.select_slider("Tasa de Aprendizaje:", [0.01, 0.05, 0.1, 0.2], value=0.05)

    if st.button("🔥 Entrenar Modelo Optimizado", use_container_width=True):
        if not features:
            st.error("Selecciona variables de entrada.")
        else:
            with st.spinner('Entrenando...'):
                df_s = df[[target] + features].dropna()
                df_l = df_s.drop(st.session_state.get('borrar', []), errors='ignore')

                def entrenar_pro(data, t_size):
                    X, y = data[features], data[target]
                    
                    # VALIDACIÓN CRUZADA (K-Fold) de respaldo
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    model_cv = xgb.XGBRegressor(n_estimators=150, max_depth=m_depth, learning_rate=l_rate)
                    cv_scores = cross_val_score(model_cv, X, y, cv=kf, scoring='r2')
                    
                    # ENTRENAMIENTO FINAL con el % elegido por el usuario
                    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=t_size/100, random_state=42)
                    model_final = xgb.XGBRegressor(n_estimators=150, max_depth=m_depth, learning_rate=l_rate)
                    model_final.fit(X_t, y_t)
                    p = model_final.predict(X_v)
                    
                    return {
                        'R2_CV': np.mean(cv_scores), 'R2_test': r2_score(y_v, p),
                        'RMSE': np.sqrt(mean_squared_error(y_v, p)), 'Bias': np.mean(p - y_v),
                        'model': model_final, 'df_val': X_v.assign(REAL=y_v, PRED=p), 'n': len(data),
                        'importancia': pd.Series(model_final.feature_importances_, index=features).sort_values()
                    }

                res_l = entrenar_pro(df_l, test_size_pct)
                # (Aquí iría el resto del código para mostrar métricas y guardar en session_state)
                st.success(f"Modelo entrenado con un {100-test_size_pct}% de entrenamiento y {test_size_pct}% de test.")
                st.metric("Precisión en Test", f"{res_l['R2_test']:.4f}")
