import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Función para cargar los datos al dataset
@st.cache_data
def cargarDatos():
    return pd.read_csv('./datos/datos-limpios.csv', encoding='latin-1')

# Título de la aplicación
st.title("Análisis Predictivo de Incendios Forestales en México")

# Cargar y mostrar el dataset
df = cargarDatos()
st.header("Dataset Limpio")
st.dataframe(data=df)

# Función para normalizar las columnas
def normalizarColumna(df, columna):
    scaler = MinMaxScaler()
    df[f'{columna}-normalizado'] = scaler.fit_transform(df[[columna]])
    return df

# Función para convertir todas las entradas de causa a minúsculas
def normalizarCausa(df):
    df['causa'] = df['causa'].str.lower()
    df['causa'] = df['causa'].str.strip()
    return df

# Aplicar normalizaciones
df = normalizarColumna(df, "duracion-dias")
df = normalizarColumna(df, "tamanio-m2")
df = normalizarCausa(df)

# Sección de Distribución de los Datos
st.header("Distribución de los Datos")

# Histograma de la duración de los incendios
fig1 = plt.figure(figsize=(10, 6))
sns.histplot(df['duracion-dias'], bins=30, kde=True)
plt.title('Distribución de la Duración de Incendios')
plt.xlabel('Duración en Días')
plt.ylabel('Frecuencia')
st.pyplot(fig1)
plt.clf()

# Histograma del tamaño de los incendios (log transformado)
fig2 = plt.figure(figsize=(10, 6))
df['log_tamanio_m2'] = np.log(df['tamanio-m2'] + 1)  # +1 para evitar log(0)
sns.histplot(df['log_tamanio_m2'], bins=30, kde=True)
plt.title('Distribución Logarítmica del Tamaño de los Incendios')
plt.xlabel('Log(Tamaño en Metros Cuadrados)')
plt.ylabel('Frecuencia')
st.pyplot(fig2)
plt.clf()

# Gráfico de barras de las causas de los incendios
fig3 = plt.figure(figsize=(12, 6))
sns.countplot(y='causa', data=df, order=df['causa'].value_counts().index)
plt.title('Frecuencia de las Causas de los Incendios')
plt.xlabel('Número de Incendios')
plt.ylabel('Causa')
st.pyplot(fig3)
plt.clf()

# Gráfico de dispersión geográfico
fig4 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitud', y='latitud', hue='duracion-dias', size='tamanio-m2', data=df, alpha=0.6, palette='viridis')
plt.title('Distribución Geográfica de Incendios')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig4)
plt.clf()

# Analizar correlación entre variables
st.header("Correlación entre Variables")

# Convertir variables categóricas a numéricas usando Label Encoding
categorical_cols = ['causa', 'tipo-vegetacion', 'estado', 'region', 'tipo-incendio', 'regimen-de-fuego', 'tipo-impacto']

# Crear una copia para evitar advertencias
df_encoded = df.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le  # Guardar el encoder para interpretación futura

# Seleccionar variables para la correlación
corr_variables = ['duracion-dias', 'tamanio-m2', 'latitud', 'longitud', 'anio'] + categorical_cols

# Calcular la matriz de correlación
corr_matrix = df_encoded[corr_variables].corr()

# Mostrar la matriz de correlación
st.subheader("Matriz de Correlación Ampliada")
fig_corr_ext, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
plt.title('Heatmap de Correlación Ampliada')
st.pyplot(fig_corr_ext)
plt.clf()

# Sección para seleccionar el modelo
st.header("Modelos de Clustering K-Means")

# Opciones de modelos
st.subheader("Seleccione el Modelo que Desea Generar")

model_option = st.selectbox(
    'Seleccione el modelo:',
    ('Modelo 1: Tamaño, Duración, Región y Causa',
     'Modelo 2: Tipo de Vegetación, Región, Régimen de Fuego y Causa')
)

if model_option == 'Modelo 1: Tamaño, Duración, Región y Causa':
    # Modelo 1
    st.write("**Variables utilizadas en el Modelo 1:**")
    st.write("- `tamanio-m2`")
    st.write("- `duracion-dias`")
    st.write("- `region`")
    st.write("- `causa`")
    
    # Variables seleccionadas para clustering
    numeric_features = ['tamanio-m2', 'duracion-dias']
    categorical_features = ['region', 'causa']
    
    # Estandarización de variables numéricas
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])
    
    # Codificación de variables categóricas con OneHotEncoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df_scaled[categorical_features])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
    
    # Concatenar variables numéricas escaladas y categóricas codificadas
    df_cluster = pd.concat([df_scaled[numeric_features], encoded_cats_df], axis=1)
    
    # Variable para identificar el modelo
    model_id = 1

elif model_option == 'Modelo 2: Tipo de Vegetación, Región, Régimen de Fuego y Causa':
    # Modelo 2
    st.write("**Variables utilizadas en el Modelo 2:**")
    st.write("- `tipo-vegetacion`")
    st.write("- `region`")
    st.write("- `regimen-de-fuego`")
    st.write("- `causa`")
    
    # Variables seleccionadas para clustering
    numeric_features = []  # No hay variables numéricas seleccionadas en este modelo
    categorical_features = ['tipo-vegetacion', 'region', 'regimen-de-fuego', 'causa']
    
    # No hay variables numéricas a estandarizar en este modelo, pero si hubiera:
    # scaler = StandardScaler()
    # df_scaled = df.copy()
    # df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])
    
    # Si no hay variables numéricas, simplemente copiamos el dataframe
    df_scaled = df.copy()
    
    # Codificación de variables categóricas con OneHotEncoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df_scaled[categorical_features])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
    
    # Concatenar variables categóricas codificadas (no hay numéricas en este modelo)
    df_cluster = encoded_cats_df.copy()
    
    # Variable para identificar el modelo
    model_id = 2

# Determinación del número óptimo de clusters con el método del codo y silhouette score
st.subheader("Determinación del Número Óptimo de Clusters")

# Cálculo de inercia y silhouette score para diferentes valores de K
inertia = []
silhouette_scores = []
k_range = range(2, 21)  # Ahora de 2 a 20

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_cluster)
    inertia.append(kmeans.inertia_)
    cluster_labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_cluster, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Gráfico del Método del Codo
fig_elbow = plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
st.pyplot(fig_elbow)
plt.clf()

# Gráfico de Silhouette Score
fig_silhouette = plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='red')
plt.title('Silhouette Score para Diferentes k')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
st.pyplot(fig_silhouette)
plt.clf()

# Recomendación de K basado en el máximo Silhouette Score
best_k = k_range[np.argmax(silhouette_scores)]
st.write(f"**El número óptimo de clusters sugerido es {best_k}, basado en el máximo Silhouette Score de {max(silhouette_scores):.2f}.**")

# Entrada del usuario para seleccionar K
st.subheader("Seleccione el Número de Clusters (K)")
selected_k = st.slider('Seleccione K', min_value=2, max_value=20, value=best_k)

if st.button("Generar Clusters y Predecir Causas para Incendios Desconocidos"):
    # Entrenamiento del modelo K-Means con el K seleccionado
    kmeans = KMeans(n_clusters=selected_k, random_state=42)
    clusters = kmeans.fit_predict(df_cluster)

    # Agregar los clusters al DataFrame original
    df['cluster'] = clusters

    # Evaluación del modelo
    silhouette_avg = silhouette_score(df_cluster, clusters)
    ch_score = calinski_harabasz_score(df_cluster, clusters)

    st.subheader("Evaluación del Modelo de Clustering")
    st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")
    st.write(f"**Calinski-Harabasz Index:** {ch_score:.2f}")

    # Visualización con PCA
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

    # Mostrar tabla con resultados
    st.subheader("Resultados de Clustering")
    st.dataframe(df[['duracion-dias', 'tamanio-m2', 'causa', 'tipo-vegetacion', 'regimen-de-fuego', 'cluster']].head(50))

    # Mostrar conteo de registros por cluster
    st.subheader("Conteo de Registros por Cluster")
    cluster_counts = df['cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # Análisis de cada cluster
    st.subheader("Análisis de Clusters")
    cluster_cause_mapping = {}  # Para almacenar la causa más común de cada cluster

    for i in range(selected_k):
        st.write(f"**Cluster {i}:**")
        cluster_data = df[df['cluster'] == i]
        st.write(f"- Número de registros: {len(cluster_data)}")
        st.write(f"- Duración promedio de incendios: {cluster_data['duracion-dias'].mean():.2f} días")
        st.write(f"- Tamaño promedio de incendios: {cluster_data['tamanio-m2'].mean():.2f} m²")
        
        # Determinar la causa más común excluyendo 'desconocido'
        if 'causa' in categorical_features:
            known_causes = cluster_data[cluster_data['causa'] != 'desconocido']
            if not known_causes.empty:
                most_common_cause = known_causes['causa'].mode()[0]
                st.write(f"- Causa más común: {most_common_cause}")
                cluster_cause_mapping[i] = most_common_cause
            else:
                st.write("- Causa más común: No disponible")
        
        # Determinar el tipo de vegetación más común si está en el modelo
        if 'tipo-vegetacion' in categorical_features:
            most_common_veg = cluster_data['tipo-vegetacion'].mode()[0]
            st.write(f"- Tipo de vegetación más común: {most_common_veg}")
        
        # Determinar el régimen de fuego más común si está en el modelo
        if 'regimen-de-fuego' in categorical_features:
            most_common_regimen = cluster_data['regimen-de-fuego'].mode()[0]
            st.write(f"- Régimen de fuego más común: {most_common_regimen}")
        
        st.write("---")

    # Predicción de causas para incendios con causa desconocida
    st.subheader("Predicción de Causas para Incendios con Causa Desconocida")

    # Filtrar los incendios con causa desconocida
    unknown_cause_df = df[df['causa'] == 'desconocido']
    if not unknown_cause_df.empty and 'causa' in categorical_features:
        # Asignar la causa más común del cluster correspondiente
        df.loc[df['causa'] == 'desconocido', 'causa_predicha'] = df[df['causa'] == 'desconocido']['cluster'].map(cluster_cause_mapping)

        # Mostrar los resultados
        st.write("Se han predicho las causas para los incendios con causa desconocida basándose en los clusters.")
        st.dataframe(df[df['causa'] == 'desconocido'][['duracion-dias', 'tamanio-m2', 'causa', 'causa_predicha', 'cluster']].head(50))

        # Guardar el nuevo dataset con las causas predichas
        output_file = './datos/datos_con_causas_predichas.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Crear la carpeta si no existe
        df.to_csv(output_file, index=False)

        st.write(f"El nuevo dataset con las causas predichas ha sido guardado en: `{output_file}`")

    else:
        st.write("No hay incendios con causa desconocida o la causa no es una variable considerada en este modelo.")

