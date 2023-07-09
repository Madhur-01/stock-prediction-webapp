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
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Forecast App")

stocks = ( 'AAPL','MSFT','FBgrx','amzn','twtr')

selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:',1,5)
period = n_years*365

forecast_method = st.selectbox("Select forecasting method", ["Prophet","ARIMA","LSTM"])

if forecast_method=="ARIMA" :
    p = st.number_input('Enter value of p')
    d = st.number_input('Enter value of d')
    q = st.number_input('Enter value of q')

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
    df_train_lstm.dropna(inplace=True)
    df_train_lstm.reset_index(drop=True, inplace=True)
    df_train_lstm = df_train_lstm.rename(columns={"Date": "ds", "Close": "y"})

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_lstm["y_scaled"] = scaler.fit_transform(df_train_lstm[["y"]])

    # Preparing the data for LSTM input
    X = df_train_lstm["y_scaled"].values
    X = np.reshape(X, (X.shape[0], 1, 1))

    # Building and training the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, 1)))
    model.add(Dense(10,'relu'))
    model.add(Dense(5,'relu'))
    model.add(Dense(1,'linear'))
    model.compile(loss="mean_squared_error", optimizer="adam")

    y_scaled = df_train_lstm["y_scaled"].values.astype('float32')  # Convert y_scaled to float32

    model.fit(X, y_scaled, epochs=50, batch_size=20, verbose=0)

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
    
    
elif forecast_method == "Prophet":

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
    
else :
    df_train_arima = data[["Date", "Close"]]
    df_train_arima.dropna(inplace=True)
    df_train_arima.reset_index(drop=True, inplace=True)
    df_train_arima = df_train_arima.rename(columns={"Date": "ds", "Close": "y"})

    model = ARIMA(df_train_arima["y"], order=(p, d, q))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(df_train_arima))
    st.subheader("Forecast Data")
    st.write(forecast.tail())

    # Plotting ARIMA forecast
    st.subheader("ARIMA Forecast")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_train_arima["ds"], y=df_train_arima["y"], name="Actual"))
    fig1.add_trace(go.Scatter(x=pd.date_range(start=df_train_arima["ds"].max(), periods=period, freq='D'), y=forecast,
                              name="ARIMA Forecast"))
    fig1.layout.update(title_text="ARIMA Forecast", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # Residuals plot
    residuals = model_fit.resid
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_train_arima["ds"], y=residuals, name="Residuals"))
    fig2.layout.update(title_text="ARIMA Residuals", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

    # Decompose components
    decomposed = sm.tsa.seasonal_decompose(df_train_arima["y"], model='additive', period=period)

    # Plot trend component
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_train_arima["ds"], y=decomposed.trend, name="Trend"))
    fig_trend.layout.update(title_text="Trend Component", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_trend)

    # Plot seasonal component
    st.subheader("Seasonal Component")
    fig_seasonal = go.Figure()
    fig_seasonal.add_trace(go.Scatter(x=df_train_arima["ds"], y=decomposed.seasonal, name="Seasonal"))
    fig_seasonal.update_layout(xaxis_tickformat="%b %d")  # Format x-axis as month and day
    fig_seasonal.layout.update(title_text="Seasonal Component", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_seasonal)

    # Plot yearly component
    st.subheader("Yearly Component")
    fig_yearly = go.Figure()
    fig_yearly.add_trace(go.Scatter(x=df_train_arima["ds"], y=decomposed.resid, name="Yearly"))
    fig_yearly.update_layout(xaxis_tickformat="%b")  # Format x-axis as month
    fig_yearly.layout.update(title_text="Yearly Component", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_yearly)