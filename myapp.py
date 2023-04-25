import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal, ndimage
import numpy as np
from numpy import sin, cos, pi, linspace

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

st.sidebar.title('Configuration')
st.sidebar.header('Data')

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


x_axis = st.sidebar.selectbox('X axis',('Column1','Column2','Column3','Column4','Column5'),index=2)
y_axis = st.sidebar.selectbox('Y axis',('Column1','Column2','Column3','Column4','Column5'),index=3)

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

sig = st.sidebar.slider('standard deviation for Gaussian kernel', min_value=0.0, max_value=100.0, value=20.0, step=5.0)
if sig==0:
    sig=0.1

ps = st.sidebar.slider('marker size in points', min_value=0.0, max_value=2.0, value=1.0, step=0.25)

# st.sidebar.header('Circle')
# red_cicle = st.sidebar.slider('define radius angle', min_value=0.0, max_value=360.0, value=25.0, step=5.0)
# start_ang = st.sidebar.slider('define start angle', min_value=0.0, max_value=360.0, value=0.0, step=5.0)
# end_ang = st.sidebar.slider('define end angle', min_value=0.0, max_value=360.0, value=90.0, step=5.0)


st.sidebar.header('Donut')
donut_pos = st.sidebar.slider('position angle', min_value=0.0, max_value=180.0, value=73.0, step=1.0)
donut_ang = st.sidebar.slider('width', min_value=0.0, max_value=180.0, value=15.0, step=1.0)
donut_inner = st.sidebar.slider('inner radius', min_value=0.0, max_value=400.0, value=100.0, step=1.0)
donut_outer = st.sidebar.slider('outer radius', min_value=0.0, max_value=400.0, value=300.0, step=1.0)
# red_cicle = st.sidebar.slider('define radius angle', min_value=0.0, max_value=360.0, value=25.0, step=5.0)
# start_ang = st.sidebar.slider('define start angle', min_value=0.0, max_value=360.0, value=0.0, step=5.0)
# end_ang = st.sidebar.slider('define end angle', min_value=0.0, max_value=360.0, value=90.0, step=5.0)

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

sns.scatterplot(x=t, y=sig1, s=ps, color='.25', ax=ax_dict2["a"], edgecolor="none")
sns.scatterplot(x=t, y=p1, s=ps, color='gold', ax=ax_dict2["a"], edgecolor="none")

sns.scatterplot(x=t, y=sig2, s=ps, color='.25', ax=ax_dict2["b"], edgecolor="none")
sns.scatterplot(x=t, y=p2, s=ps, color='gold', ax=ax_dict2["b"], edgecolor="none")

sns.scatterplot(x=sig1, y=sig2, s=ps, color='.25', ax=ax_dict2["c"], edgecolor="none")
sns.scatterplot(x=p1, y=p2, s=ps, color='gold', ax=ax_dict2["c"], edgecolor="none")

# Define the parameters of the partial donuts
center = [0, 0]
inner_radius = donut_inner
outer_radius = donut_outer
width = outer_radius - inner_radius
angle1_start = donut_pos-(donut_ang/2)
angle1_end = donut_pos+(donut_ang/2)
angle2_start = (donut_pos+180)-(donut_ang/2)
angle2_end = (donut_pos+180)+(donut_ang/2)

# Create the two sectors of the partial donuts
# sector1 = patches.Wedge(center, outer_radius, angle1_start, angle1_end, width=width, edgecolor='blue', color='blue', alpha=0.15)
# sector2 = patches.Wedge(center, outer_radius, angle2_start, angle2_end, width=width, edgecolor='red', color='red', alpha=0.15)
sector1 = patches.Wedge(center, outer_radius, angle1_start, angle1_end, width=width, edgecolor='blue', lw=2, fill=False)
sector2 = patches.Wedge(center, outer_radius, angle2_start, angle2_end, width=width, edgecolor='red', lw=2, fill=False)

# Plot the scatter plot and the partial donuts
ax_dict2["c"].add_artist(sector1)
ax_dict2["c"].add_artist(sector2)

# #draw a circle
# angles = linspace((start_ang*pi)/180, (end_ang*pi)/180, 200 )
# xs = red_cicle * cos(angles)
# ys = red_cicle * sin(angles)
# sns.scatterplot(x=xs, y=ys, s=5, color='red', ax=ax_dict2["c"], edgecolor="none")

# circle1 = plt.Circle(xy=(0, 0), radius=red_cicle, color='red', fill=False)
# ax_dict2["c"].add_patch(circle1)
ax_dict2["c"].set(xlim=(-400, 400))
ax_dict2["c"].set(ylim=(-400, 400))


st.pyplot(fig2)

# Graph -- gaussian_filter1d -- end

st.subheader('Raw data')

data['filter1'] = sig1
data['filter2'] = sig2

st.write(data)