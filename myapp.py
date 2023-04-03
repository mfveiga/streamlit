import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import numpy as np

st.set_page_config(layout="wide")

st.title('Eddy Current')

# DATA_URL = ('RAW_Coletados.parquet')

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
    st.subheader(uploaded_file.name)
    data = pd.read_excel(uploaded_file)
    # data = load_data_excel(uploaded_file)
else:
    st.subheader('RAW_Coletados')
    data = pd.read_parquet('RAW_Coletados.parquet')
    # data = load_data(DATA_URL)

# frequency_Hz = st.sidebar.slider('Frequency (Hz)', 0, 15000, 10000)
frequency_Hz = 10000

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(DATA_URL)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache_data)")

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
fs = frequency_Hz

#Vector tempo
T = 1/fs
t = np.arange(0,N/fs,T)

#FFT
f = np.fft.fftfreq(N,T)
transf = np.fft.fft(data.Column1) #transformada
transf = np.abs(transf)


x_axis = st.sidebar.selectbox('X axis',('Column1','Column2','Column3','Column4','Column5'),index=0)
y_axis = st.sidebar.selectbox('Y axis',('Column1','Column2','Column3','Column4','Column5'),index=1)

sig1 = data[x_axis]
sig2 = data[y_axis]

# # Graph -- signal.butter -- begin

# st.subheader('Graph -- signal.butter')

# bt = st.sidebar.selectbox('The type of filter',('lowpass', 'highpass', 'bandpass', 'bandstop'),index=0)
# b1 = st.sidebar.slider('The order of the filter', 0, 10, 1)
# b2 = st.sidebar.slider('The critical frequency or frequencies', 1, 100, 1)

# sos = signal.butter(b1, b2, bt, fs=frequency_Hz, output='sos')
# filtered = signal.sosfilt(sos, sig1)
# filtered2 = signal.sosfilt(sos, sig2)

# fig = plt.figure(layout="constrained",figsize=(14,6))
# ax_dict = fig.subplot_mosaic(
#     [
#         ["x", "out"],
#         ["y", "out"],
#     ],
# )

# sns.scatterplot(x=t, y=sig1, s=1, color='.15', ax=ax_dict["x"])
# sns.scatterplot(x=t, y=filtered, s=1, color='yellow', ax=ax_dict["x"])

# sns.scatterplot(x=t, y=sig2, s=1, color='.15', ax=ax_dict["y"])
# sns.scatterplot(x=t, y=filtered2, s=1, color='yellow', ax=ax_dict["y"])

# sns.scatterplot(x=sig1, y=sig2, s=1, color='.15', ax=ax_dict["out"])
# sns.scatterplot(x=filtered, y=filtered2, s=1, color='yellow', ax=ax_dict["out"])

# st.pyplot(fig)

# # Graph -- signal.butter -- end


# Graph -- gaussian_filter1d -- begin

st.subheader('Graph -- gaussian_filter1d')

sig = st.sidebar.slider('standard deviation for Gaussian kernel', min_value=0.0, max_value=100.0, value=50.0, step=5.0)
if sig==0:
    sig=0.1

ps = st.sidebar.slider('marker size in points', min_value=0.0, max_value=2.0, value=1.0, step=0.25)

p1 = ndimage.gaussian_filter1d(sig1, sigma=sig)
p2 = ndimage.gaussian_filter1d(sig2, sigma=sig)

# fig2 = plt.figure(figsize=(14,6))

fig2 = plt.figure(layout="constrained",figsize=(14,6))
ax_dict2 = fig2.subplot_mosaic(
    [
        ["a", "c"],
        ["b", "c"],
    ],
)

sns.scatterplot(x=t, y=sig1, s=ps, color='.15', ax=ax_dict2["a"])
sns.scatterplot(x=t, y=p1, s=ps, color='gold', ax=ax_dict2["a"])

sns.scatterplot(x=t, y=sig2, s=ps, color='.15', ax=ax_dict2["b"])
sns.scatterplot(x=t, y=p2, s=ps, color='gold', ax=ax_dict2["b"])

sns.scatterplot(x=sig1, y=sig2, s=ps, color='.15', ax=ax_dict2["c"])
sns.scatterplot(x=p1, y=p2, s=ps, color='gold', ax=ax_dict2["c"])

st.pyplot(fig2)

# Graph -- gaussian_filter1d -- end

st.subheader('Raw data')

data['filter1'] = sig1
data['filter2'] = sig2

st.write(data)