import streamlit as st
import pandas as pd

st.title('Eddy Current')

DATA_URL = ('https://github.com/mfveiga/streamlit/blob/9610eb040305a6a4c0bc3ed625749a5f6071ef0b/RAW_Coletados.parquet')

@st.cache_data
def load_data(data_path):
    df = pd.read_parquet(DATA_URL)
    #df = pd.read_parquet('Filtrado.parquet')
    return df

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(DATA_URL)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")

st.subheader('Raw data')
st.write(data)