import streamlit as st
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats

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

st.title("Dataset con nuevas columnas normalizadas:")
st.dataframe(data=df)

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


