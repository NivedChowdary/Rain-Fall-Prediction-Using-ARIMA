# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:14:47 2019

@author: G.GAUTAM AND K.RITESH
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading the dataset
#dataset = pd.read_csv('annual.csv')
#dataset = pd.read_csv('myData.csv')
dataset = pd.read_csv('annual.csv',parse_dates = ['YEAR'])

#parse string to datetime type
dataset['YEAR'] = pd.date_range(start='1901', end='2017',freq = 'AS')
#dataset['YEAR'] = pd.date_range('1901-01-01', periods=117,  freq='M')

indexedDataset = dataset.set_index(['YEAR'])

#Plot a Graph
plt.xlabel("Date")
plt.ylabel("Rainfall")
plt.plot(indexedDataset)

#Determining rolling statistics
rolmean =  indexedDataset.rolling(window=1).mean()
rolstd =  indexedDataset.rolling(window=1).std()
print(rolmean,rolstd)

#Plot rolling statistics
orig = plt.plot(indexedDataset,color = 'blue',label = 'Original')
mean = plt.plot(rolmean ,color = 'red',label = 'rolling Mean')
std  = plt.plot(rolstd , color ='black' ,label = 'rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling mean vs Standard deviation')
plt.show(block = False)

#Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller

print('Results of Dickey-Fuller Test: ')
dftest = adfuller(indexedDataset['ANNUAL'],autolag='AIC')
dfoutput = pd.Series(dftest[0:4],index = ['Test Statistics','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical value (%s)'%key] = value
print(dfoutput)

#Estimating Trend
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)

#Determining rolling statistics
movingAverage =  indexedDataset_logScale.rolling(window=1).mean()
movingSTD =  indexedDataset_logScale.rolling(window=1).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color = 'red')

#now we are going to logscale. This may vary with the time series you
#may either myltiply,take log etc...
datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove Nan values
datasetLogScaleMinusMovingAverage.dropna(inplace = True)
datasetLogScaleMinusMovingAverage.head(10)

#ADCF function (Augumenter Dicker Fuller  )
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determining rolling statistics
    rolmean =  timeseries.rolling(window=12).mean()
    rolstd =  timeseries.rolling(window=12).std()
    print(rolmean,rolstd)
    #Plot rolling statistics
    orig = plt.plot(timeseries,color = 'blue',label = 'Original')
    mean = plt.plot(rolmean ,color = 'red',label = 'rolling Mean')
    std = plt.plot(rolstd , color ='black' ,label = 'rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling mean vs Standard deviation')
    plt.show(block = False)
    ##Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test: ')
    dftest = adfuller(timeseries['ANNUAL'],autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],index = ['Test Statistics','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)'%key] = value
    print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage)

exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife = 12,min_periods = 0 ,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage,color='red')

datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

datasetLogDiffShifting = indexedDataset_logScale-indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)    
#till here we have made timeseries stationary

#components of timeseries
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale,label='Original')
plt.legend(loc = 'best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc = 'best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc = 'best')
plt.subplot(414)
plt.plot(residual,label='residual')
plt.legend(loc = 'best')
plt.tight_layout()

decomposeLogData = residual
decomposeLogData.dropna(inplace=True)
test_stationarity(decomposeLogData)

#Now to get P and q we need to get ACF and PACF plots:
from statsmodels.tsa.stattools import acf,pacf
lag_acf = acf(datasetLogDiffShifting,nlags = 20)
lag_pacf= pacf(datasetLogDiffShifting,nlags=20, method = 'ols')

#PLOT ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--',color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--',color = 'gray')
plt.title('Autocorrelation Function')
#i guess q = 1

plt.subplot(121)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color = 'gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--',color = 'gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle = '--',color = 'gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#i guess p = 1

from statsmodels.tsa.arima_model import ARIMA

#AR Model
model = ARIMA(indexedDataset_logScale,order=(1,1,1))
results_AR1 = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR1.fittedvalues,color = 'red')
plt.title('RSS: %.4f'% sum((results_AR1.fittedvalues - datasetLogDiffShifting['ANNUAL'])**2))
print('Plotting AR model')

#MA Model
model = ARIMA(indexedDataset_logScale,order=(2,1,0))
results_MA1 = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA1.fittedvalues,color = 'red')
plt.title('RSS: %.4f'% sum((results_MA1.fittedvalues - datasetLogDiffShifting['ANNUAL'])**2))
print('Plotting AR model')

model = ARIMA(indexedDataset_logScale,order=(1,1,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color = 'red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['ANNUAL'])**2))
print('Plotting AR model')

#Fitting in a combined model called Arima
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy = True)
print(predictions_ARIMA_diff.head(30))

#Convert to cumulative sum
predictions_ARIMA_diff_cumsum =predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head(30))

prediction_ARIMA_log = pd.Series(indexedDataset_logScale['ANNUAL'].ix[0],index=indexedDataset_logScale.index)
prediction_ARIMA_log =prediction_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()

#to get the original form we take exponential 
prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(prediction_ARIMA)

indexedDataset_logScale

results_ARIMA.plot_predict(1,207)
X=results_ARIMA.forecast(steps = 90)