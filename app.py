import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

NOMBRE_ALUMNO = "Sandro"
LINK_COLAB = "https://colab.research.google.com/drive/11rEb4ggFfEU0LX6tu9M4F8SOb4_CzMx-?usp=sharing"

# Configuración inicial de la página web
st.set_page_config(
    page_title="Predicción de Viviendas - Boston", 
    page_icon="🏡", 
    layout="centered"
)

# REQUERIMIENTO OBLIGATORIO: Contenido de la Barra Lateral
st.sidebar.title("Información del Alumno")
st.sidebar.markdown(f"**Nombre:** \n{NOMBRE_ALUMNO}")
st.sidebar.markdown("---")
st.sidebar.subheader("Enlaces del Proyecto")

# Enlace al cuaderno de Colab en modo lector
st.sidebar.markdown(f"🔗 [Ver Cuaderno de Código en Google COLAB]({LINK_COLAB})")

# Cuerpo Principal de la Aplicación Web
st.title("🏡 Clasificador del Valor de Viviendas en Boston")
st.markdown("""
Esta aplicación web utiliza un modelo avanzado de **Machine Learning (Random Forest)** para estimar si una propiedad inmobiliaria pertenece a la categoría de **Alto Valor** o **Bajo/Medio Valor**, analizando variables del entorno urbano y características estructurales de la edificación.
""")

st.subheader("📊 Panel de Ingreso de Datos")
st.write("Modifique las características de la vivienda para calcular la predicción:")

# Diseño de la interfaz en dos columnas simétricas
col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("Tasa de Criminalidad per cápita por zona (crim):", min_value=0.0, max_value=100.0, value=0.1, step=0.01)
    rm = st.slider("Número promedio de habitaciones por vivienda (rm):", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
    age = st.slider("Proporción de casas construidas antes de 1940 (age):", min_value=0.0, max_value=100.0, value=65.0, step=1.0)

with col2:
    dis = st.number_input("Distancia ponderada a centros de empleo (dis):", min_value=0.5, max_value=15.0, value=3.8, step=0.1)
    tax = st.number_input("Tasa de impuesto a la propiedad por cada $10k (tax):", min_value=150.0, max_value=800.0, value=400.0, step=10.0)

st.markdown("---")

# Botón para ejecutar el procesamiento y la predicción del modelo
if st.button("🔮 Calcular Clasificación de Valor", use_container_width=True):
    try:
        # Ruta apuntando a tu carpeta en GitHub
        ruta_modelo = os.path.join('modelos', 'modelo_boston_rf.pkl')
        
        if not os.path.exists(ruta_modelo):
            st.error("Error Crítico: El archivo 'modelo_boston_rf.pkl' no se encuentra dentro de la carpeta 'modelos/'.")
        else:
            pipeline = joblib.load(ruta_modelo)
            
            # Reconstrucción del DataFrame inicial con las 5 variables de los sliders
            datos_entrada = pd.DataFrame([[crim, rm, age, dis, tax]], 
                                         columns=['crim', 'rm', 'age', 'dis', 'tax'])
            
            # VALORES PROMEDIO REALES para evitar el sobreajuste por ceros (0.0)
            valores_por_defecto = {
                'zn': 11.36, 'indus': 11.13, 'chas': 0.069, 'nox': 0.55, 
                'rad': 9.54, 'ptratio': 18.45, 'b': 356.67, 'lstat': 12.65
            }
            
            # Rellenar las columnas faltantes usando los promedios del dataset original
            columnas_entrenamiento = pipeline.feature_names_in_ if hasattr(pipeline, 'feature_names_in_') else ['crim', 'rm', 'age', 'dis', 'tax']
            for col in columnas_entrenamiento:
                if col not in datos_entrada.columns:
                    # Si la columna no está en el slider, le asignamos su promedio real o 0.0 si no está en el diccionario
                    datos_entrada[col] = valores_por_defecto.get(col, 0.0)
                    
            # Alineamos las columnas estrictamente al orden requerido por tu modelo
            datos_entrada = datos_entrada[columnas_entrenamiento]
            
            # Inferencia del modelo
            prediccion = pipeline.predict(datos_entrada)[0]
            
            # Presentación de resultados visuales
            st.subheader("🎯 Resultado del Modelo:")
            if prediccion == 1:
                st.success("🎉 **VIVIENDA DE ALTO VALOR:** El inmueble se clasifica en el segmento de precios superiores del mercado.")
            else:
                st.warning("📉 **VIVIENDA DE BAJO / MEDIO VALOR:** El inmueble se clasifica en el segmento estándar o económico del mercado.")
                
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar el modelo: {str(e)}")
