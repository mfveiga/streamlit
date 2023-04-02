import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

st.set_page_config(layout="wide")

st.title('Eddy Current')

DATA_URL = ('RAW_Coletados.parquet')

@st.cache_data
def load_data(data_path):
    df = pd.read_parquet(data_path)
    #df = pd.read_parquet('Filtrado.parquet')
    return df

@st.cache_data
def load_data_excel(data_path):
    df = pd.read_excel(data_path)
    #df = pd.read_parquet('Filtrado.parquet')
    return df

st.sidebar.header('Configuration')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input PARQUET file", type=["xlsx"])
if uploaded_file is not None:
    data = load_data_excel(uploaded_file)
else:
    data = load_data(DATA_URL)

# frequency_Hz = st.sidebar.slider('Frequency (Hz)', 0, 15000, 10000)

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(DATA_URL)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache_data)")

st.subheader('Graph')

sns.set_theme(style="dark")
sns.set(rc={'axes.facecolor':'black', 
    'figure.facecolor':'black',
    "grid.color": ".6",
    "grid.linestyle": ":",
    'axes.facecolor': 'black',
    'xtick.color': 'white',
    'ytick.color': 'white',})

#Tamanho (comprimento)
N = len(data.index)

#Frequência
fs = 10000

#Vector tempo
T = 1/fs
t = np.arange(0,N/fs,T)

#FFT
f = np.fft.fftfreq(N,T)
transf = np.fft.fft(data.Column1) #transformada
transf = np.abs(transf)

sig1 = data.Column1
sig2 = data.Column2


b1 = st.sidebar.slider('The order of the filter', 0, 10, 1)
b2 = st.sidebar.slider('The critical frequency or frequencies', 1, 10, 1)

sos = signal.butter(b1, b2, 'lowpass', fs=10000, output='sos')
filtered = signal.sosfilt(sos, sig1)
filtered2 = signal.sosfilt(sos, sig2)

fig = plt.figure(layout="constrained",figsize=(14,6))
ax_dict = fig.subplot_mosaic(
    [
        ["x", "out"],
        ["y", "out"],
    ],
)
delay = 0.01

sns.scatterplot(x=t, y=data.Column1, s=1, color='.15', ax=ax_dict["x"])
sns.scatterplot(x=t-delay, y=filtered, s=1, color='yellow', ax=ax_dict["x"])

sns.scatterplot(x=t, y=data.Column2, s=1, color='.15', ax=ax_dict["y"])
sns.scatterplot(x=t-delay, y=filtered2, s=1, color='yellow', ax=ax_dict["y"])

sns.scatterplot(x=data.Column1, y=data.Column2, s=1, color='.15', ax=ax_dict["out"])
sns.scatterplot(x=filtered, y=filtered2, s=1, color='yellow', ax=ax_dict["out"])

st.pyplot(fig)

st.subheader('Raw data')
st.write(data)