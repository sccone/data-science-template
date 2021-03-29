from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
from numpy import concatenate
import numpy as np
import pandas as pd
import math
from scipy.ndimage.interpolation import shift
import plotly.graph_objs as go
import plotly.express as px

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, label_col='index',dropnan=True, include_self=True):
    """data: raw datafram
        n_in: number of previous time step used for modeling, lagging
        n_out: number of forecast time step, sequence
        label_col: the target column
        include_self: check if itself is included as part of X
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    df_noself=df.copy()
    df_noself=df_noself.drop([label_col],axis=1)
    n_vars_noself = 1 if type(df_noself) is list else df_noself.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    if include_self:
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [(df.columns[j]+'(t-%d)' % (i) ) for j in range(n_vars)]
    else:
        for i in range(n_in, 0, -1):
            cols.append(df_noself.shift(i))
            names += [(df_noself.columns[j]+'(t-%d)' % (i) ) for j in range(n_vars_noself)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[label_col].shift(-i))
        if i == 0:
            names += [(label_col+'(t)')]
        else:
            names += [(label_col+'(t+%d)' %  (i))]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, r_test, n_lag, n_seq, label_col,include_self=True):
    """series: the raw dataframe
        r_test: percentage of test sample
        n_lag: number of previous time step used for modeling, lagging
        n_seq:number of forecast time step, sequence
        label_col: the target column
        include_self: check if itself is included as part of X
    """
    # extract raw values
    n_test=math.floor(r_test*series.shape[0])
    _input_num=n_lag*series.shape[1]
    train_index, test_index = series.index[0:-n_test], series.index[-n_test:]
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values_ndarry=scaler.fit_transform(series)
    scaled_values = DataFrame(scaled_values_ndarry)
    scaled_values.columns=series.columns
    scaled_values_shape = scaled_values.copy()
    if not include_self:
        scaled_values_shape=scaled_values_shape.drop([label_col],axis=1)
        _input_num=n_lag*scaled_values_shape.shape[1]
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq, label_col,include_self=include_self)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    train_raw, test_raw = scaled_values[0:-n_test], scaled_values[-n_test:]
    train_x,train_y=train[:, :_input_num], train[:, _input_num:]
    test_x, test_y = test[:, :_input_num], test[:, _input_num:]
    # train_x=train_x.drop([label_col],axis=1)
    # test_x=test_x.drop([label_col],axis=1)
    # reshape training into [samples, timesteps, features]
    train_x=train_x.reshape(train_x.shape[0], n_lag, scaled_values_shape.shape[1])
    test_x=test_x.reshape(test_x.shape[0], n_lag, scaled_values_shape.shape[1])
    return scaler, train_raw, test_raw,train_x,train_y,test_x,test_y,train_index, test_index

# fit an LSTM network to training data
def fit_lstm(train_x,train_y, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_x.shape[1], train_x.shape[2]), stateful=True))
	model.add(Dense(train_y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=nb_epoch, batch_size=n_batch, verbose=0, shuffle=False)
	return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return forecast

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, label_col, n_seq):
    inverted = list()
    for i in range(n_seq):
        # create array from forecast
        series[label_col] = forecasts[:,i]
        # invert scaling
        inv_scale = DataFrame(scaler.inverse_transform(series))
        inv_scale.columns=series.columns
        inv_forecast=inv_scale[label_col].to_numpy()
        inv_forecast = inv_forecast.reshape(len(inv_forecast), 1)
        # store
        inverted.append(inv_forecast)
    return concatenate(inverted,axis=1)

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# plot the forecasts in the context of the original dataset
def plot_forecasts(date, actual, forecast):
    fig=px.scatter(
        # x=test_index,
        # y=actual[:,0],
        width=750,
        height=500,
        # log_y=True
    ).update_traces(mode='lines')
    fig.add_trace(
        go.Scatter(
        mode='lines',
        x=date,
        y=actual[:,0],
        marker=dict(
            color="#2CA02C",
            size=2,
            ),
        name="actual"
        ) 
    ),
    fig.add_trace(
        go.Scatter(
        mode='lines',
        x=date,
        y=shift(actual[:,0],1),
        marker=dict(
            color="#00CC96",
            size=2,
            ),
        name="actual shift"
        ) 
    ),
    fig.add_trace(
        go.Scatter(
        x=date,
        y=forecasts[:,0],
        marker=dict(
            color="#FF7F0E",
            size=2,
            ),
        name="lstm"
    ))
    fig.show()
    return

def getdata(ticker,rollingdays=30,delta_days=1):
    """ticker can be 'qqq.us', or 'spy.us'"""
    address = "https://stooq.com/q/d/l/?s=" + ticker + "&i=d"
    df=pd.read_csv(address)
    df['Delta']=(df['Close'].shift(-delta_days)-df['Close'])/df['Close']
    df['Delta_rev']=(df['Close']-df['Close'].shift(delta_days))/df['Close'].shift(delta_days)
    df['Delta_1day']=(df['Close']-df['Close'].shift(1))/df['Close'].shift(1)
    df['Vol_rolling']=df['Delta_1day'].rolling(rollingdays).std()
    df['Date']=pd.to_datetime(df['Date'])
    df.index=df['Date']
    return df

df_raw = getdata('qqq.us',30,10)
df_raw=df_raw[df_raw['Date']>datetime.datetime(2018,1,1)]
df=df_raw.copy().sort_index()
df=df.drop(['Date','Open','High','Low','Delta_1day'],axis=1)

# configure
n_lag = 2
n_seq = 1
r_test = 0.3
n_epochs = 50
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test, train_x, train_y, test_x, test_y,train_index,test_index = prepare_data(df, r_test, n_lag, n_seq,label_col='Delta',include_self=True)
# fit model
model = fit_lstm(train_x,train_y, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecast=model.predict(test_x,n_batch)
# inverse transform forecasts and test
forecasts = inverse_transform(test, forecast, scaler, label_col='Delta', n_seq=n_seq)
actual = inverse_transform(test, test_y, scaler, label_col='Delta', n_seq=n_seq)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
evaluate_forecasts(actual, shift(actual,1), n_lag, n_seq)

plot_forecasts(test_index, actual, forecast)