# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:02:37 2023

@author: madhu
"""

import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecast App")

stocks = ( 'AAPL','MSFT','FBgrx','amzn','twtr')

selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:',1,5)
period = n_years*365

forecast_method = st.selectbox("Select forecasting method", ["Prophet", "LSTM"])

@st.cache_data(persist=True)

def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
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


if forecast_method == "LSTM":
    df_train_lstm = data[["Date", "Close"]]
    df_train_lstm = df_train_lstm.dropna().reset_index(drop=True)
    df_train_lstm = df_train_lstm.rename(columns={"Date": "ds", "Close": "y"})

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_lstm["y_scaled"] = scaler.fit_transform(df_train_lstm[["y"]])
    df_train_lstm["y_scaled"] = df_train_lstm["y_scaled"].astype(float)
    # Preparing the data for LSTM input
    X = df_train_lstm[["ds", "y_scaled"]].copy()
    X["ds"] = pd.to_datetime(X["ds"])  # Convert the "ds" column to datetime
   # X["y_scaled"] = X["y_scaled"].astype(float)  # Convert "y_scaled" to float
    X = X.values.reshape(-1, 2, 1)

     # Building and training the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(2, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    y_scaled = df_train_lstm["y_scaled"].values.astype('float32')  # Convert y_scaled to float32

    model.fit(X, y_scaled, epochs=10, batch_size=16, verbose=0)
    # Predicting with the LSTM model
    forecast_scaled = model.predict(X)
    forecast = scaler.inverse_transform(forecast_scaled)

    # Plotting LSTM forecast
    st.subheader("LSTM Forecast")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_train_lstm["ds"], y=df_train_lstm["y"], name="Actual"))
    fig3.add_trace(go.Scatter(x=df_train_lstm["ds"], y=forecast[:, 0], name="LSTM Forecast"))
    fig3.layout.update(title_text="LSTM Forecast", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)
else:

    #Predict forecast with Prophet
    df_train = data[["Date","Close"]]
    df_train.dropna(inplace=True)
    df_train = df_train.rename(columns={"Date":"ds","Close":"y"})
    
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods = period)
    forecast = m.predict(future)
    
    #Show and plot forecast
    st.subheader("Forecast Data")
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for{n_years} years')
    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)
    
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    







