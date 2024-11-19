import streamlit as st
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PSA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans

# Funcion para cargar los datos al dataset
@st.cache_data
def cargarDatos():
    return pd.read_csv('./datos/datos-limpios.csv', encoding='latin-1')

st.title("Dataset original:")
st.dataframe(data=pd.read_csv('./datos/datos-originales.csv', encoding='latin-1'))

df = cargarDatos()
st.title("Dataset limpio:")
st.dataframe(data=df)

# Funcion para normalizar las columans
def normalizarColumna(df, columna):
    scaler = MinMaxScaler()
    df[f'{columna}-normalizado'] = scaler.fit_transform(df[[columna]])
    return df

# Funcion para convertir todas las entradas de causa a minusculas
def normalizarCausa(df):
    # Convertir todas las entradas a minúsculas
    df['causa'] = df['causa'].str.lower()

    # Eliminar espacios en blanco al inicio y al final
    df['causa'] = df['causa'].str.strip()

    return df

df = normalizarColumna(df, "duracion-dias")
df = normalizarColumna(df, "tamanio-m2")
df = normalizarCausa(df)

st.title("Distribucion de los datos")

# Histograma de la duración de los incendios
plt.figure(figsize=(10, 6))
sns.histplot(df['duracion-dias'], bins=30, kde=True)
plt.title('Distribución de la Duración de Incendios')
plt.xlabel('Duración en Dias')
plt.ylabel('Frecuencia')
st.pyplot(plt)

# Histograma del tamanio de los incendios
fig2 = plt.figure(figsize=(10, 6))
df['log_tamanio_m2'] = np.log(df['tamanio-m2'] + 1)  # +1 para evitar log(0)
sns.histplot(df['log_tamanio_m2'], bins=30, kde=True)
plt.title('Distribucion Logaritmica del Tamanio de los Incendios')
plt.xlabel('Log(Tamaioo en Metros Cuadrados)')
plt.ylabel('Frecuencia')
st.pyplot(fig2)
plt.clf()

# Grafico de barras de las causas de los incendios
plt.figure(figsize=(12, 6))
sns.countplot(y='causa', data=df, order=df['causa'].value_counts().index)
plt.title('Frecuencia de las Causas de los Incendios')
plt.xlabel('Numero de Incendios')
plt.ylabel('Causa')
st.pyplot(plt)

# Grafico de dispersion geografico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='longitud', y='latitud', hue='duracion-dias', size='tamanio-m2', data=df)
plt.title('Distribucion Geografica de Incendios')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
st.pyplot(plt)

# Analizar correlacion entre variables
st.title("Correlacion entre variables")

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

# Variables seleccionadas para clustering
numeric_features = ['duracion-dias-normalizado', 'log_tamanio_m2', 'latitud', 'longitud']
categorical_features = ['causa', 'tipo-vegetacion', 'region']

# Estandarización de variables numéricas
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

# Codificación de variables categóricas con OneHotEncoding
encoder = OneHotEncoder(sparse_output=False)
encoded_cats = encoder.fit_transform(df_scaled[categorical_features])
encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))

# Concatenar variables numéricas escaladas y categóricas codificadas
df_cluster = pd.concat([df_scaled[numeric_features], encoded_cats_df], axis=1)

# Determinación del número óptimo de clusters con el método del codo
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.show()

# Usar silhouette score para evaluar k
best_k = 3  # Cambiar basado en resultados del codo o silhouette
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(df_cluster)

# Agregar los clusters al DataFrame original
df['cluster'] = clusters

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_cluster)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('Clusters visualizados con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

if st.button("Generar Clusters"):
    st.title("Resultados de Clustering")
    st.dataframe(data=df[['duracion-dias', 'tamanio-m2', 'causa', 'cluster']])
    st.pyplot(plt)  # Donde plt corresponde al gráfico PCA generado.
