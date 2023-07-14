# Stock-Prediction-Webapp
https://stock-prediction-webapp.streamlit.app/

In this project, I designed a web app in streamlit cloud to accurately predict stock prices of different firms. A stock represents ownership certificates of a company, and its price undergoes changes over time, forming a time series. Time series refers to an ordered sequence of data points distributed across a specific period.

In this project, I used yfinance library which provides real time stock dataset of firms like google, amazon, microsoft etc.It allows us to easily download financial data from Yahoo Finance. It provides a simple and convenient way to access a wide range of financial data for a given stock symbol, including historical price data, financial statements, and other information.

There are many different techniques designed specifically for dealig with time series. Such techniques range from simple visualization tools that show trends evolving or repeating over time to advanced machine learning models that utilize the specific structure of time series. Most common used models are LSTM, ARIMA, SARIMA FBProphet, NeuralProphet. In this project I used FBProphet as its time complexity is less and have built in functions to visualise predictions.

Technologies used in this project-

1. Python
2. Numpy and Pandas
3. plotly for data visualization
4. prophet for forecasting
5. Google Colab, visual studio code and pycharm as IDE
6. Python flask for http server
7. Streamlit Cloud for UI
