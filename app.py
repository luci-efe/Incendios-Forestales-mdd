import streamlit as st
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Funcion para cargar los datos al dataset
@st.cache_data
def cargarDatos():
    return pd.read_csv('./datos/datos-limpios.csv', encoding='latin-1')

df = cargarDatos()
st.title("Dataset original:")
st.dataframe(data=df)

# Funcion para normalizar las columans
def normalizarColumna(df, columna):
    scaler = MinMaxScaler()
    df[f'{columna}-normalizado'] = scaler.fit_transform(df[[columna]])
    return df

df = normalizarColumna(df, "duracion-dias")
df = normalizarColumna(df, "tamanio-m2")

st.title("Dataset con nuevas columnas normalizadas:")
st.dataframe(data=df)

# Función para crear histogramas
def crearHistogramas(df, columnaOriginal, columnaNormalizada):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma de datos originales
    ax1.hist(df[columnaOriginal], bins=30, edgecolor='black')
    ax1.set_title(f'Distribución de {columnaOriginal}')
    ax1.set_xlabel(columnaOriginal)
    ax1.set_ylabel('Frecuencia')

    # Histograma de datos normalizados
    ax2.hist(df[columnaNormalizada], bins=30, edgecolor='black')
    ax2.set_title(f'Distribución de {columnaNormalizada}')
    ax2.set_xlabel(columnaNormalizada)
    ax2.set_ylabel('Frecuencia')

    plt.tight_layout()
    return fig

# Crear y mostrar histogramas para tamaño
st.write("### Histogramas de Tamaño")
figTamanio = crearHistogramas(df, 'tamanio-m2', 'tamanio-m2-normalizado')
st.pyplot(figTamanio)

# Crear y mostrar histogramas para duración
st.write("### Histogramas de Duración")
figDuracion = crearHistogramas(df, 'duracion-dias', 'duracion-dias-normalizado')
st.pyplot(figDuracion)

def crearGraficosDispersion(df, columnaOriginal, columnaNormalizada):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df[columnaOriginal], df[columnaNormalizada], alpha=0.5)
    ax.set_xlabel(columnaOriginal)
    ax.set_ylabel(columnaNormalizada)
    ax.set_title(f'Relación entre {columnaOriginal} y {columnaNormalizada}')

    # Añadir una línea de referencia
    ax.plot([df[columnaOriginal].min(), df[columnaOriginal].max()],
            [0, 1], 'r--', lw=2, alpha=0.7)

    return fig

# Crear y mostrar gráfico de dispersión para tamaño
st.write("### Gráfico de Dispersión: Tamaño")
figTamanio = crearGraficosDispersion(df, 'tamanio-m2', 'tamanio-m2-normalizado')
st.pyplot(figTamanio)

# Crear y mostrar gráfico de dispersión para duración
st.write("### Gráfico de Dispersión: Duración")
figDuracion = crearGraficosDispersion(df, 'duracion-dias', 'duracion-dias-normalizado')
st.pyplot(figDuracion)
