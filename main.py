# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:02:37 2023

@author: madhu
"""

import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecast App")

stocks = ('GOOG', 'AAPL')

selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:',1,4)
period = n_years*365

@st.cache(persist=True, allow_output_mutation=True)

def load_data(ticker):
    data = yf.download(ticker,start =START,end = TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Loading data....")
try:
    data = load_data(selected_stock)
    data_load_state.text("Loading data... done!")
except Exception as e:
    st.error(f"Error occurred while loading data: {str(e)}")

st.subheader("Raw data")
st.write(data.tail())

#Plotting raw data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y = data["Open"],name = "stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y = data["Close"],name = "stock_close"))
    fig.layout.update(title_text = "Time Series data with Rangeslider", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

#Predict forecast with Prophet
df_train = data[["Date","Close"]]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecaste = m.predict(future)

#Show and plot forecast
st.subheader("Forecast Data")
st.write(fotrcast.tail())

st.write(f'Forecast plot for{n_years} years')
fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)








