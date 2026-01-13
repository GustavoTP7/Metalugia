with tabs[1]:
        st.subheader("🚩 Radar de Desviaciones y Diagnóstico de Fallas")
        
        # 1. Preparación de datos de auditoría
        df_audit = st.session_state['res']['df_audit'].copy()
        # Traemos las variables X de vuelta para analizar por qué falló
        df_audit = df_audit.merge(st.session_state['res']['df_work'], left_on='ID_Turno', right_on=st.session_state['res']['col_id'], how='left')
        df_audit['Desviación_%'] = (df_audit['Error'] / df_audit['Real']) * 100

        col_a1, col_a2 = st.columns([1, 1])

        with col_a1:
            st.markdown("### 🏆 Top 10 Turnos Críticos")
            st.table(df_audit[['ID_Turno', 'Real', 'Pred', 'Error', 'Desviación_%']].head(10))
            
        with col_a2:
            st.markdown("### 🔍 ¿En qué rango falla más el modelo?")
            # Este gráfico te dice si el modelo falla más cuando la recuperación es baja o alta
            fig_error_rango = px.scatter(df_audit, x='Real', y='Error', 
                                         color='Error', size='Error',
                                         title="Error Absoluto vs Valor Real",
                                         labels={'Real': f"Valor Real de {st.session_state['res']['target']}"})
            st.plotly_chart(fig_error_rango, use_container_width=True)

        st.divider()

        col_a3, col_a4 = st.columns([1, 1])
        
        with col_a3:
            st.markdown("### 🌡️ Correlación del Error con Sensores")
            # Aquí elegimos una variable X para ver si el error crece cuando esa variable sube
            var_analisis = st.selectbox("Analizar error contra variable:", st.session_state['res']['features'])
            fig_correl_err = px.scatter(df_audit, x=var_analisis, y='Error', 
                                        trendline="ols", title=f"¿El error depende de {var_analisis}?",
                                        color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_correl_err, use_container_width=True)
            st.info("💡 Si la línea de tendencia sube, significa que el sensor está mal calibrado o el modelo necesita más datos en rangos altos de esa variable.")

        with col_a4:
            st.markdown("### 📊 Resumen Estadístico de la Falla")
            avg_err = df_audit['Error'].mean()
            max_err = df_audit['Error'].max()
            st.metric("Error Promedio del Sistema", f"{avg_err:.3f}")
            st.metric("Desviación Máxima Detectada", f"{max_err:.3f}")
            
            # Alerta de confianza
            if avg_err > (st.session_state['res']['df_work'][st.session_state['res']['target']].std() * 0.5):
                st.warning("⚠️ El error promedio es alto comparado con la variabilidad natural. Revisar sensores.")
            else:
                st.success("✅ El modelo mantiene una desviación aceptable para la operación.")

        st.subheader("📋 Auditoría Maestra de Filas")
        st.dataframe(df_audit, use_container_width=True)
