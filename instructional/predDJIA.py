# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:26:59 2017

@author: collin.rooney
"""

import pandas as pd
from htsprophet.hts import hts
from htsprophet.htsPlot import plotNode
from lastprophet.last import lastF
import matplotlib.pyplot as plt
from fbprophet import Prophet

#%% Read in the data

dJIA = pd.read_csv("DowComponents.csv")
dJIA = dJIA.drop("Unnamed: 0", axis = 1)  # Get rid of unnecessary index column
dJIA = dJIA.rename(columns = {dJIA.columns[1] : 'Total'}) #We do this to appease the Plot Child function, which needs to know which column is the total
nodes = [[30]]   # Tell the function the structure of the hierarchy, here we have 30 components to one total node

#%% Forecast to find Stock Market Holidays
data = dJIA.copy()
pred = hts(data.iloc[:550, :], 470, nodes, method = "BU", yearly_seasonality = True, freq = "B")
realDates = pd.DatetimeIndex(dJIA.Date)
leftOvers = [d for d in pred["Total"].ds if d not in realDates]
leftOvers = [d for d in leftOvers if d < pd.datetime(2017,8,29)]    # Only look at dates that are within the range of the original dataset
extraPred = len(leftOvers)

#%% fit data with hts and forecast a year into the future 
pred = hts(dJIA, h = 260+extraPred, nodes = nodes, method = "WLSS", freq = "B")  # We set freq = B because we have a business day frequency and not the default daily frequency
for key in pred.keys():
    pred[key] = pred[key][~pred[key].ds.isin(leftOvers)]

#%% plot the model
plotNode(pred, "GS", 365, 'Date', 'Weighted Price')

#%% Test for reduction in error
import random
random.seed(1)
maeD1 = []
maeD5 = []
maeD20 = []
maeD65 = []
maeD130 = []
maeD195 = []
maeD260 = []
dat = dJIA
for j in range(550, 750, 20):  # Rolling Forecast Origin
    print(j)
    train_index = [i for i in range(j)]
    test_index = [i for i in range(j, len(dJIA))]
    testLen = len(test_index)
    if testLen < 260 or len(train_index) < 522:    # If training data is less than two years (business day years) or test data is less than 1 year, skip it
        continue
    pred = hts(dat.iloc[train_index, :], testLen+extraPred, nodes, method = "WLSS", freq = "B", yearly_seasonality = True)   # Fit the model
    for key in pred.keys():
        pred[key] = pred[key][~pred[key].ds.isin(leftOvers)]
        pred[key] = pred[key].reset_index(drop = True)              #Remove the holidays
    i = 0
    ##
    # Because HTS outputs all Time series at once, all MAE values will be calculated at once
    ##
    for column in dJIA.columns:
        if i == 0 or i == 1:          # Skipping the Total and Date Columns
            i += 1
            continue
        maeD1.append(abs(pred[column].yhat[j] - dat.iloc[test_index[0],2+i-2])/dat.iloc[test_index[0],2+i-2])
        maeD5.append(abs(pred[column].yhat[j+4] - dat.iloc[test_index[4],2+i-2])/dat.iloc[test_index[4],2+i-2])
        maeD20.append(abs(pred[column].yhat[j+19] - dat.iloc[test_index[19],2+i-2])/dat.iloc[test_index[19],2+i-2])
        maeD65.append(abs(pred[column].yhat[j+64] - dat.iloc[test_index[64],2+i-2])/dat.iloc[test_index[64],2+i-2])
        maeD130.append(abs(pred[column].yhat[j+129] - dat.iloc[test_index[129],2+i-2])/dat.iloc[test_index[129],2+i-2])
        maeD195.append(abs(pred[column].yhat[j+194] - dat.iloc[test_index[194],2+i-2])/dat.iloc[test_index[194],2+i-2])
        maeD260.append(abs(pred[column].yhat[j+259] - dat.iloc[test_index[259],2+i-2])/dat.iloc[test_index[259],2+i-2])
        i += 1
#%% Get Base Forecast Errors
import random
random.seed(1)
maeD1 = []
maeD5 = []
maeD20 = []
maeD65 = []
maeD130 = []
maeD195 = []
maeD260 = []
for i in range(30):     # Change to 1 if you want to run on just DJIA total
    dat = dJIA.iloc[:, [0, 2+i]]    # Change 2+i to 1 if you want to run on just DJIA total
    dat = dat.rename(columns = {dat.columns[0] : 'ds'})
    dat = dat.rename(columns = {dat.columns[1] : 'y'})
    for j in range(550, 750, 20):                         # Change 20 to 5 if you want to run on just DJIA total
        print(j)
        train_index = [i for i in range(j)]
        test_index = [i for i in range(j, len(dJIA))]
        testLen = len(test_index)
        if testLen < 260 or len(train_index) < 522:
            continue
        ##
        # Run Through prophet and calculate MAE
        ##
        m = Prophet()
        m.fit(dat.iloc[train_index, :])
        future = m.make_future_dataframe(periods = testLen+extraPred, freq = "B")
        pred = m.predict(future)
        pred = pred[~pred.ds.isin(leftOvers)]
        pred = pred.reset_index(drop = True)
        maeD1.append(abs(pred.yhat[j] - dat.iloc[test_index[0],1]))#/dat.iloc[test_index[0],1])
        maeD5.append(abs(pred.yhat[j+4] - dat.iloc[test_index[4],1]))#/dat.iloc[test_index[4],1])
        maeD20.append(abs(pred.yhat[j+19] - dat.iloc[test_index[19],1]))#/dat.iloc[test_index[19],1])
        maeD65.append(abs(pred.yhat[j+64] - dat.iloc[test_index[64],1]))#/dat.iloc[test_index[64],1])
        maeD130.append(abs(pred.yhat[j+129] - dat.iloc[test_index[129],1]))#/dat.iloc[test_index[129],1])
        maeD195.append(abs(pred.yhat[j+194] - dat.iloc[test_index[194],1]))#/dat.iloc[test_index[194],1])
        maeD260.append(abs(pred.yhat[j+259] - dat.iloc[test_index[259],1]))#/dat.iloc[test_index[259],1])
#%% HTS for DJIA Total
import random
random.seed(1)
maeD1 = []
maeD5 = []
maeD20 = []
maeD65 = []
maeD130 = []
maeD195 = []
maeD260 = []
dat = dJIA
for j in range(550, 750, 5):
    print(j)
    train_index = [i for i in range(j)]
    test_index = [i for i in range(j, len(dJIA))]
    testLen = len(test_index)
    if testLen < 260 or len(train_index) < 522:
        continue
    pred = hts(dat.iloc[train_index, :], testLen+extraPred, nodes, method = "WLSS", freq = "B", yearly_seasonality = True)
    for key in pred.keys():
        pred[key] = pred[key][~pred[key].ds.isin(leftOvers)]
        pred[key] = pred[key].reset_index(drop = True)
    i = 0
    ##
    # Because HTS outputs all Time series at once, all MAE values will be calculated at once
    ##
    maeD1.append(abs(pred["Total"].yhat[j] - dat.iloc[test_index[0],1]))
    maeD5.append(abs(pred["Total"].yhat[j+4] - dat.iloc[test_index[4],1]))
    maeD20.append(abs(pred["Total"].yhat[j+19] - dat.iloc[test_index[19],1]))
    maeD65.append(abs(pred["Total"].yhat[j+64] - dat.iloc[test_index[64],1]))
    maeD130.append(abs(pred["Total"].yhat[j+129] - dat.iloc[test_index[129],1]))
    maeD195.append(abs(pred["Total"].yhat[j+194] - dat.iloc[test_index[194],1]))
    maeD260.append(abs(pred["Total"].yhat[j+259] - dat.iloc[test_index[259],1]))
#%% PLots
#%% Make Plots
##
# Base Forecasts
##
dat = dJIA.iloc[:, [0, 15]]
dat = dat.rename(columns = {dat.columns[0] : 'ds'})
dat = dat.rename(columns = {dat.columns[1] : 'y'})
s = Prophet()
s.fit(dat.iloc[:719, :])
future = s.make_future_dataframe(periods = 281, freq = "B")
pred = s.predict(future)
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(pd.DatetimeIndex(dJIA.iloc[:,0].values), dJIA.iloc[:,15], label = "True Values")
ax.plot(pred.ds.values, pred.yhat.values, label = "Base Forecast")
fig.suptitle('Prophet Forecasts vs Weighted JNJ Price', fontsize=20)
ax.set_xlabel("Day", fontsize=14)
ax.set_ylabel("Weigthed Price", fontsize=14)
ax.legend(fontsize = 12)
#%% HTS Forecasts
data = dJIA.copy()
pred1 = hts(data.iloc[:719, :], 281, nodes, method = "PHA", yearly_seasonality = True, freq = "B")
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(pd.DatetimeIndex(dJIA.iloc[:,0].values), dJIA.iloc[:,3], label = "True Values")
ax.plot(pred1["GS"].ds.values, pred1["GS"].yhat.values, label = "HTS Forecast")
ax.set_xlabel("Day", fontsize=14)
ax.set_ylabel("Weigthed Price", fontsize=14)
ax.legend()

#%% Get Reconciled Forecast Errors
import random
random.seed(1)
maeD1 = []
maeD5 = []
maeD20 = []
maeD65 = []
maeD130 = []
maeD195 = []
maeD260 = []
for i in range(30):          # Change to 1 if you want to run on just DJIA total
    dat = dJIA.iloc[:, [0, 2+i]]            # Change 2+i to 1 if you want to run on just DJIA total
    dat = dat.rename(columns = {dat.columns[0] : 'ds'})
    dat = dat.rename(columns = {dat.columns[1] : 'y'})
    for j in range(550, 750, 20):                       # Change 20 to 5 if you want to run on just DJIA total
        print(j)
        train_index = [i for i in range(j)]
        test_index = [i for i in range(j, len(dJIA))]
        testLen = len(test_index)
        if testLen < 260 or len(train_index) < 522:
            continue
        ##
        # Run Through lastF (with the newDict line commented out and return line changed to "return forecastsDict, boxcoxT"), and calculate MAE
        ##
        pred = lastF(dat.iloc[train_index, :], m = 252, h = testLen+9, comb = "WLSS", aggList = [1,252])
        pred[252] = pred[252][~pred[252].ds.isin(leftOvers)]
        pred[252] = pred[252].reset_index(drop = True)
        lenPred = len(pred[252])
        maeD1.append(abs(pred[252].yhat[lenPred-testLen] - dat.iloc[test_index[0],1]))#/dat.iloc[test_index[0],1])
        maeD5.append(abs(pred[252].yhat[lenPred-testLen+4] - dat.iloc[test_index[4],1]))#/dat.iloc[test_index[4],1])
        maeD20.append(abs(pred[252].yhat[lenPred-testLen+19] - dat.iloc[test_index[19],1]))#/dat.iloc[test_index[19],1])
        maeD65.append(abs(pred[252].yhat[lenPred-testLen+64] - dat.iloc[test_index[64],1]))#/dat.iloc[test_index[64],1])
        maeD130.append(abs(pred[252].yhat[lenPred-testLen+129] - dat.iloc[test_index[129],1]))#/dat.iloc[test_index[129],1])
        maeD195.append(abs(pred[252].yhat[lenPred-testLen+194] - dat.iloc[test_index[194],1]))#/dat.iloc[test_index[194],1])
        maeD260.append(abs(pred[252].yhat[lenPred-testLen+259] - dat.iloc[test_index[259],1]))#/dat.iloc[test_index[259],1])
#%% Make Plots
##
# Reconciled Forecasts
##
dat = dJIA.iloc[:, [0, 15]]
pred1 = lastF(dat.iloc[:719, :], m = 252, h = 281, comb = "WLSS", aggList = [1,252])
lenPred = len(pred1[252])
fig = plt.figure(facecolor='w', figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(pd.DatetimeIndex(dJIA.iloc[:,0].values), dJIA.iloc[:,15], label = "True Values")
ax.plot(pred1[252].ds.values, pred1[252].yhat.values, label = "LAST Forecast")
ax.set_xlabel("Day", fontsize=14)
ax.set_ylabel("Weigthed Price", fontsize=14)
ax.legend(fontsize = 12)
