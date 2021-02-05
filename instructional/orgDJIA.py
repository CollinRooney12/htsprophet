# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:58:33 2017

@author: collin.rooney
"""

import pandas as pd
import numpy as np

#%% Get the data
tickers = ["AAPL","T","GS","NKE","V","UNH","CSCO","TRV","CVX","PFE","VZ","HD","INTC","MSFT","JNJ","WMT","CAT","JPM","DIS","BA","KO","MCD","AXP","IBM","MRK","MMM","UTX","DD","PG","XOM","GE"]
##
# Read in all the data I downloaded as CSVs from yahoo
##
dowJones = pd.read_csv("DJIA.csv")
dowJones = dowJones.iloc[:,[0,5]]
for ticker in tickers:
    ##
    # Deal with T -> AAPL
    ##
    if ticker == "AAPL":
        apple = pd.read_csv(ticker+".csv")
        apple = apple.iloc[:,5]
        continue
    if ticker == "T":
        aTT = pd.read_csv(ticker+".csv")
        aTT = aTT.iloc[:,5]
        a = np.hstack((aTT,apple))
        dowJones["AAPL+T"] = a
        continue
    temp = pd.read_csv(ticker+".csv")
    dowJones[ticker] = temp.iloc[:,5]

#%% Scale all non total time series by DJ divisor
sumComponents = np.sum(dowJones.iloc[:,2:], axis = 1)
divisor = sumComponents/dowJones.iloc[:,1]
dowJones.iloc[:,2:] = dowJones.iloc[:,2:].divide(divisor, axis = 0)

#%% Write DJIA to file
dowJones.to_csv("DowComponents.csv")
