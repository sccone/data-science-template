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

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, label_col='index',dropnan=True):
    """data: raw datafram
        n_in: number of previous time step used for modeling, lagging
        n_out: number of forecast time step, sequence
        label_col: the target column
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(df.columns[j]+'(t-%d)' % (i) ) for j in range(n_vars)]
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
def prepare_data(series, r_test, n_lag, n_seq, label_col):
    """series: the raw dataframe
        r_test: percentage of test sample
        n_lag: number of previous time step used for modeling, lagging
        n_seq:number of forecast time step, sequence
        label_col: the target column
    """
    # extract raw values
    n_test=math.floor(r_test*series.shape[0])
    _input_num=n_lag*series.shape[1]
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values_ndarry=scaler.fit_transform(series)
    scaled_values = DataFrame(scaled_values_ndarry)
    scaled_values.columns=series.columns
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq, label_col)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    train_raw, test_raw = scaled_values[0:-n_test], scaled_values[-n_test:]
    train_x,train_y=train[:, :_input_num], train[:, _input_num:]
    test_x, test_y = test[:, :_input_num], test[:, _input_num:]
    # reshape training into [samples, timesteps, features]
    train_x=train_x.reshape(train_x.shape[0], n_lag, series.shape[1])
    test_x=test_x.reshape(test_x.shape[0], n_lag, series.shape[1])
    return scaler, train_raw, test_raw,train_x,train_y,test_x,test_y

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
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

def getdata(ticker,rollingdays=30):
    """ticker can be 'qqq.us', or 'spy.us'"""
    address = "https://stooq.com/q/d/l/?s=" + ticker + "&i=d"
    df=pd.read_csv(address)
    df['Delta']=df['Close'].diff()/df['Close']
    df['Vol_rolling']=df['Delta'].rolling(rollingdays).std()
    df['Date']=pd.to_datetime(df['Date'])
    df.index=df['Date']
    return df

df_raw = getdata('qqq.us',30)
df_raw=df_raw[df_raw['Date']>datetime.datetime(2018,1,1)]
df=df_raw.copy().sort_index()
df=df.drop(['Date','Open','High','Low'],axis=1)

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
# integer encode direction
encoder = LabelEncoder()
dataset['wnd_dir'] = encoder.fit_transform(dataset['wnd_dir'])
# ensure all data is float
dataset = dataset.astype('float32')
# configure
n_lag = 2
n_seq = 3
r_test = 0.3
n_epochs = 1
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test, train_x, train_y, test_x, test_y = prepare_data(df, r_test, n_lag, n_seq,label_col='Delta')
# fit model
model = fit_lstm(train_x,train_y, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecast=model.predict(test_x,n_batch)
# inverse transform forecasts and test
forecasts = inverse_transform(test, forecast, scaler, label_col='Delta', n_seq=3)
actual = inverse_transform(test, test_y, scaler, label_col='Delta', n_seq=3)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test=2)