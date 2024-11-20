import streamlit as st
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Función para cargar los datos al dataset
@st.cache_data
def cargarDatos():
    return pd.read_csv('./datos/datos-limpios.csv', encoding='latin-1')

# Título principal de la aplicación
st.title("Análisis Predictivo de Incendios Forestales en México")

# Mostrar datasets originales y limpios
st.header("Dataset original:")
st.dataframe(data=pd.read_csv('./datos/datos-originales.csv', encoding='latin-1'))

df = cargarDatos()
st.header("Dataset limpio:")
st.dataframe(data=df)

# Función para normalizar las columnas
def normalizarColumna(df, columna):
    scaler = MinMaxScaler()
    df[f'{columna}-normalizado'] = scaler.fit_transform(df[[columna]])
    return df

# Función para normalizar causa
def normalizarCausa(df):
    df['causa'] = df['causa'].str.lower()
    df['causa'] = df['causa'].str.strip()
    return df

# Aplicar normalizaciones
df = normalizarColumna(df, "duracion-dias")
df = normalizarColumna(df, "tamanio-m2")
df = normalizarCausa(df)

st.header("Dataset con columnas normalizadas:")
st.dataframe(data=df)

# Sección 1: Análisis de Distribución de Datos
st.header("Análisis de Distribución de Datos")

# Función para crear histogramas de comparación
def crearHistogramas(df, columnaOriginal, columnaNormalizada):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(df[columnaOriginal], bins=30, edgecolor='black')
    ax1.set_title(f'Distribución de {columnaOriginal}')
    ax1.set_xlabel(columnaOriginal)
    ax1.set_ylabel('Frecuencia')
    
    ax2.hist(df[columnaNormalizada], bins=30, edgecolor='black')
    ax2.set_title(f'Distribución de {columnaNormalizada}')
    ax2.set_xlabel(columnaNormalizada)
    ax2.set_ylabel('Frecuencia')
    
    plt.tight_layout()
    return fig

# Mostrar histogramas
st.subheader("Histogramas de Tamaño")
figTamanio = crearHistogramas(df, 'tamanio-m2', 'tamanio-m2-normalizado')
st.pyplot(figTamanio)

st.subheader("Histogramas de Duración")
figDuracion = crearHistogramas(df, 'duracion-dias', 'duracion-dias-normalizado')
st.pyplot(figDuracion)

# Distribución logarítmica del tamaño
df['log_tamanio_m2'] = np.log(df['tamanio-m2'] + 1)
fig2 = plt.figure(figsize=(10, 6))
sns.histplot(df['log_tamanio_m2'], bins=30, kde=True)
plt.title('Distribución Logarítmica del Tamaño de los Incendios')
plt.xlabel('Log(Tamaño en Metros Cuadrados)')
plt.ylabel('Frecuencia')
st.pyplot(fig2)
plt.clf()

# Gráfico de causas de incendios
fig3 = plt.figure(figsize=(12, 6))
sns.countplot(y='causa', data=df, order=df['causa'].value_counts().index)
plt.title('Frecuencia de las Causas de los Incendios')
plt.xlabel('Número de Incendios')
plt.ylabel('Causa')
st.pyplot(fig3)
plt.clf()

# Distribución geográfica
fig4 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitud', y='latitud', hue='duracion-dias', size='tamanio-m2', data=df)
plt.title('Distribución Geográfica de Incendios')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
st.pyplot(fig4)
plt.clf()

# Sección 2: Análisis de Correlación
st.header("Análisis de Correlación")

# Convertir variables categóricas a numéricas
categorical_cols = ['causa', 'tipo-vegetacion', 'estado', 'region', 'tipo-incendio', 'regimen-de-fuego', 'tipo-impacto']
df_encoded = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Matriz de correlación
corr_variables = ['duracion-dias', 'tamanio-m2', 'latitud', 'longitud', 'anio'] + categorical_cols
corr_matrix = df_encoded[corr_variables].corr()

# Mostrar matriz de correlación
fig_corr_ext, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
plt.title('Heatmap de Correlación')
st.pyplot(fig_corr_ext)
plt.clf()

# Sección 3: Clustering K-Means
st.header("Análisis de Clustering K-Means")

# Preparar datos para clustering
numeric_features = ['duracion-dias-normalizado', 'log_tamanio_m2', 'latitud', 'longitud']
categorical_features = ['causa', 'tipo-vegetacion', 'region']

# Estandarización
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

# Codificación one-hot
encoder = OneHotEncoder(sparse_output=False)
encoded_cats = encoder.fit_transform(df_scaled[categorical_features])
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))

# Datos para clustering
df_cluster = pd.concat([df_scaled[numeric_features], encoded_cats_df], axis=1)

# Determinación del número óptimo de clusters
st.subheader("Determinación del Número Óptimo de Clusters")

inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_cluster)
    inertia.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(df_cluster, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Gráficos de evaluación de clusters
fig_elbow = plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
st.pyplot(fig_elbow)
plt.clf()

fig_silhouette = plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='red')
plt.title('Silhouette Score para Diferentes k')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
st.pyplot(fig_silhouette)
plt.clf()

# Selección de K óptimo
best_k = k_range[np.argmax(silhouette_scores)]
st.write(f"**El número óptimo de clusters sugerido es {best_k}, basado en el máximo Silhouette Score de {max(silhouette_scores):.2f}.**")

# Interfaz para selección de K
selected_k = st.slider('Seleccione K', min_value=2, max_value=10, value=best_k)

if st.button("Generar Clusters"):
    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    clusters = kmeans.fit_predict(df_cluster)
    df['cluster'] = clusters

    # Métricas de evaluación
    silhouette_avg = silhouette_score(df_cluster, clusters)
    ch_score = calinski_harabasz_score(df_cluster, clusters)
    
    st.subheader("Evaluación del Modelo de Clustering")
    st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")
    st.write(f"**Calinski-Harabasz Index:** {ch_score:.2f}")

    # Visualización PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_cluster)
    
    fig_clusters = plt.figure(figsize=(8, 5))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Clusters Visualizados con PCA (K={selected_k})')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    st.pyplot(fig_clusters)
    plt.clf()

    # Análisis de clusters
    st.subheader("Resultados de Clustering")
    st.dataframe(df[['duracion-dias', 'tamanio-m2', 'causa', 'cluster']].head(50))

    st.subheader("Conteo de Registros por Cluster")
    cluster_counts = df['cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("Análisis Detallado de Clusters")
    for i in range(selected_k):
        st.write(f"**Cluster {i}:**")
        cluster_data = df[df['cluster'] == i]
        st.write(f"- Número de registros: {len(cluster_data)}")
        st.write(f"- Duración promedio de incendios: {cluster_data['duracion-dias'].mean():.2f} días")
        st.write(f"- Tamaño promedio de incendios: {cluster_data['tamanio-m2'].mean():.2f} m²")
        st.write(f"- Causa más común: {cluster_data['causa'].mode()[0]}")
        st.write("---")

# Sección 4: Modelo Predictivo
st.header("Modelo Predictivo de Duración de Incendios")

# Preparar datos para el modelo
X = df[['latitud', 'longitud', 'total-hectareas', 'tamanio-m2-normalizado', 'duracion-dias-normalizado']].values
y = df['duracion-dias'].values

# División train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y compilar modelo
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar modelo
st.write("Entrenando el modelo...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluación del modelo
loss = model.evaluate(X_test, y_test)
st.write(f"Error cuadrático medio (MSE) en el conjunto de prueba: {loss}")

# Predicciones
predicciones = model.predict(X_test)

# Métricas de evaluación
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

st.subheader("Métricas de Evaluación del Modelo")
st.write(f"*Error Cuadrático Medio (MSE):* {mse:.2f}")
st.write(f"*Coeficiente de Determinación (R²):* {r2:.2f}")

# Visualizaciones del modelo
st.subheader("Visualización del Rendimiento del Modelo")

# Gráfico de evolución del error
fig_loss = plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida (train)')
plt.plot(history.history['val_loss'], label='Pérdida (test)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
st.pyplot(fig_loss)
plt.clf()

# Predicciones vs valores reales
fig_pred = plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicciones, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Línea Ideal")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Valores Reales vs Predicciones")
plt.legend()
st.pyplot(fig_pred)
plt.clf()

# Histograma de errores
errores = y_test - predicciones.flatten()
fig_err = plt.figure(figsize=(10, 6))
plt.hist(errores, bins=20, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label="Sin Error")
plt.xlabel("Error (Valor Real - Predicción)")
plt.ylabel("Frecuencia")
plt.title("Histograma de los Errores")
plt.legend()
st.pyplot(fig_err)
plt.clf()

# Interfaz para predicciones personalizadas
st.subheader("Predicción Personalizada")
latitud = st.number_input('Latitud:', min_value=-90.0, max_value=90.0, value=0.0)
longitud = st.number_input('Longitud:', min_value=-180.0, max_value=180.0, value=0.0)
hectareas = st.number_input('Total de Hectáreas:', min_value=0.0, value=100.0)
tamano_m2 = st.number_input('Tamaño (m2):', min_value=0.0, value=1000.0)

# Normalizar las entradas del usuario
if st.button('Realizar Predicción'):
    # Crear y normalizar datos de entrada
    input_data = np.array([[latitud, longitud, hectareas, tamano_m2, tamano_m2]])
    
    # Normalizar columnas relevantes usando MinMaxScaler
    scaler_tamano = MinMaxScaler()
    scaler_tamano.fit(df[['tamanio-m2']])
    input_data[:, 3] = scaler_tamano.transform([[tamano_m2]]).flatten()
    input_data[:, 4] = scaler_tamano.transform([[tamano_m2]]).flatten()
    
    # Realizar predicción
    pred = model.predict(input_data)
    
    # Mostrar resultado
    st.write("### Resultado de la Predicción")
    st.write(f"La duración estimada del incendio es de: **{pred[0][0]:.2f} días**")
    
    # Información adicional
    st.write("### Contexto de la Predicción")
    avg_duration = df['duracion-dias'].mean()
    median_duration = df['duracion-dias'].median()
    
    st.write(f"""
    Contexto estadístico:
    - Duración media de incendios en el dataset: {avg_duration:.2f} días
    - Duración mediana de incendios en el dataset: {median_duration:.2f} días
    """)
    
    # Clasificación relativa
    if pred[0][0] < median_duration:
        st.write("Esta predicción está por debajo de la mediana de duración.")
    else:
        st.write("Esta predicción está por encima de la mediana de duración.")

# Resumen y métricas del dataset
st.header("Resumen del Dataset")
st.write("### Estadísticas Generales")
summary_stats = df[['duracion-dias', 'tamanio-m2', 'total-hectareas']].describe()
st.dataframe(summary_stats)

# Información adicional sobre el modelo
st.header("Información del Modelo")
st.write("""
### Características del Modelo
- Tipo: Red Neuronal Feed-Forward
- Capas: 3 (entrada: 64 neuronas, oculta: 32 neuronas, salida: 1 neurona)
- Función de activación: ReLU (capas ocultas), Lineal (capa de salida)
- Optimizador: Adam
- Métrica de pérdida: Error Cuadrático Medio (MSE)
""")

# Notas y consideraciones
st.header("Notas y Consideraciones")
st.write("""
- Las predicciones se basan en patrones históricos del dataset.
- Los resultados pueden variar según las condiciones específicas no capturadas en el modelo.
- Se recomienda usar esta herramienta como una guía complementaria junto con otros métodos de evaluación.
- La precisión del modelo depende de la calidad y representatividad de los datos de entrenamiento.
""")

# Footer
st.markdown("---")
st.markdown("*Desarrollado para análisis predictivo de incendios forestales en México*")
