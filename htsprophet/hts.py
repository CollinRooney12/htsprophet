# -*- coding: utf-8 -*-
"""
Name: hts.py
Author: Collin Rooney
Last Updated: 7/18/2017

This script will contain functions for all types of hierarchical modeling approaches.
It will use the prophet package as a forecasting tool.
The general idea of it is very similar to the hts package in R, but it is a little
more specific with how the dataframe is put together.

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import TimeSeriesSplit
import htsprophet.fitForecast as fitForecast

#%%
def hts(y, h = 1, nodes = [[2]], method='OC', freq = 'D', include_history = True, cap = None, capF = None, changepoints = None, \
        n_changepoints = 25, yearly_seasonality = 'auto', weekly_seasonality = 'auto', holidays = None, seasonality_prior_scale = 10.0, \
        holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05, mcmc_samples = 0, interval_width = 0.80, uncertainty_samples = 0):
    '''
    Parameters
    ----------------
     y - dataframe of time-series data
               Layout:
                   0th Col - Time instances
                   1st Col - Total of TS
                   2nd Col - One of the children of the Total TS
                   3rd Col - The other child of the Total TS
                   ...
                   ... Rest of the 1st layer
                   ...
                   Xth Col - First Child of the 2nd Col
                   ...
                   ... All of the 2nd Col's Children
                   ...
                   X+Yth Col - First Child of the 3rd Col
                   ...
                   ..
                   .   And so on...
    
     h - number of step ahead forecasts to make (int)
    
     nodes - a list or list of lists of the number of child nodes at each level
     Ex. if the hierarchy is one total with two child nodes that comprise it, the nodes input would be [2]
     
     method - String  the type of hierarchical forecasting method that the user wants to use. 
                Options:
                "OC" - optimal combination (Default), 
                "FP" - forcasted proportions (top-down)
                "PHA" - proportions of historical averages (top-down)
                "AHP" - average historical proportions (top-down)
                "BU" - bottom-up (simple addition)
                "CVselect" - select which method is best for you based on 3-fold Cross validation (longer run time)
     
     freq - (Time Frequency) input for the forecasting function of Prophet 
     
     include_history - (Boolean) input for the forecasting function of Prophet
                
     cap - (Dataframe or Constant) carrying capacity of the input time series.  If it is a dataframe, then
                                   the number of columns must equal len(y.columns) - 1
                                   
     capF - (Dataframe or Constant) carrying capacity of the future time series.  If it is a dataframe, then
                                    the number of columns must equal len(y.columns) - 1
     
     changepoints - (DataFrame or List) changepoints for the model to consider fitting. If it is a dataframe, then
                                        the number of columns must equal len(y.columns) - 1
     
     n_changepoints - (constant or list) changepoints for the model to consider fitting. If it is a list, then
                                         the number of items must equal len(y.columns) - 1
                                         
     All other inputs - see Prophet
     
    Returns
    -----------------
     ynew - a dictionary of DataFrames with predictions, seasonalities and trends that can all be plotted
    
    '''
    # Function Definitions
    ##
    #  "Creating the summing matrix" funciton
    ##
    def SummingMat(nodes):
        '''
         This function creates a summing matrix for the bottom up and optimal combination approaches
         All the inputs are the same as above
         The output is a summing matrix, see Rob Hyndman's "Forecasting: principles and practice" Section 9.4
        '''
        numAtLev = list(map(sum, nodes))
        numLevs = len(numAtLev)
        top = np.ones(numAtLev[-1])       #Create top row, which is just all ones
        blMat = np.identity(numAtLev[-1]) #Create Identity Matrix for Bottom level Nodes
        finalMat = blMat
        ##
        # These two loops build the matrix from bottom to top 
        ##
        for lev in range(numLevs-1):
            summing = nodes[-(lev + 1)]
            count = 0
            a = 0
            num2sumInd = 0
            B = np.zeros([numAtLev[-1]])
            for num2sum in summing:
                num2sumInd += num2sum
                a = blMat[count:num2sumInd, :]
                count += num2sum
                if np.all(B == 0):
                    B = a.sum(axis = 0)
                else:
                    B = np.vstack((B, a.sum(axis = 0)))
            finalMat = np.vstack((B, finalMat))
            blMat = B
        ##
        # Append the Top array to the Matrix and then return it
        ##
        finalMat = np.vstack((top, finalMat))
        return finalMat
    ##
    # Error Handling
    ##
    if h < 1:
        sys.exit('you must set h (number of step-ahead forecasts) to a positive number')
    if method not in ['OC','FP','PHA','AHP','BU','cvSelect']:
        sys.exit("not a valid method input, must be one of the following: 'OC','FP','PHA','AHP','BU','cvSelect'")
    if len(nodes) < 1:
        sys.exit("nodes input should at least be of length 1")
    if sum(list(map(sum, nodes))) != len(y.columns)-2:
        sys.exit("The sum of the nodes list does not equal the number of columns - 2, dataframe should contain a time column in the 0th pos. Double check node input")
    if not isinstance(cap, int) and not isinstance(cap, pd.DataFrame) and not isinstance(cap, float) and not cap is None:
        sys.exit("cap should be a constant (float or int) or a DataFrame, or not specified")
    if not isinstance(capF, int) and not isinstance(capF, pd.DataFrame) and not isinstance(capF, float) and not capF is None:
        sys.exit("capF should be a constant (float or int) or a DataFrame, or not specified")
    if isinstance(cap, pd.DataFrame):
        if len(cap.columns) != len(y.columns)-1:
            sys.exit("If cap is a DataFrame, it should have a number of columns equal to the input Dataframe - 1")
    if isinstance(capF, pd.DataFrame):
        if len(capF.columns) != len(y.columns)-1:
            sys.exit("If capF is a DataFrame, it should have a number of columns equal to the input Dataframe - 1")
    ##
    # Run specified approach if desired
    ##
    if method == 'OC':
        sumMat = SummingMat(nodes)
        ynew = fitForecast.optimalComb(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                            yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                            changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
    if method == 'FP':
        sumMat = SummingMat(nodes)
        ynew = fitForecast.forecastProp(y, h, nodes, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                             yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                             changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
    if method == 'PHA':
        sumMat = SummingMat(nodes)
        ynew = fitForecast.propHistAvg(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                            yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                            changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
    if method == 'AHP':
        sumMat = SummingMat(nodes)
        ynew = fitForecast.averageHistProp(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                                yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                                changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
    if method == 'BU':
        sumMat = SummingMat(nodes)
        ynew = fitForecast.bottomUp(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                         yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                         changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
    if method == 'cvSelect':
        ##
        # Run all of the Methods and let 3 fold CV chose which is best for you
        ##
        sumMat = SummingMat(nodes)
        tscv = TimeSeriesSplit(n_splits=3)
        MASE1 = []
        MASE2 = []
        MASE3 = []
        MASE4 = []
        MASE5 = []
        ##
        # Split into train and test, using time series split, and predict the test set
        ##
        for trainIndex, testIndex in tscv.split(y.iloc[:,0]):
            ynew1 = fitForecast.optimalComb(y.iloc[trainIndex, :], len(testIndex), sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            ynew2 = fitForecast.forecastProp(y.iloc[trainIndex, :], len(testIndex), nodes, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                 yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                 changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            ynew3 = fitForecast.propHistAvg(y.iloc[trainIndex, :], len(testIndex), sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            ynew4 = fitForecast.averageHistProp(y.iloc[trainIndex, :], len(testIndex), sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            ynew5 = fitForecast.bottomUp(y.iloc[trainIndex, :], len(testIndex), sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                             yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                             changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            for key in ynew1.keys():
                MASE1.append(sum(abs(ynew1[key].yhat[-len(testIndex):].values - y.iloc[testIndex, key+1].values))/((len(testIndex)/(len(testIndex)-1))*sum(abs(y.iloc[testIndex[1:], key + 1].values - y.iloc[testIndex[:-1], key + 1].values))))
                MASE2.append(sum(abs(ynew2[key].yhat[-len(testIndex):].values - y.iloc[testIndex, key+1].values))/((len(testIndex)/(len(testIndex)-1))*sum(abs(y.iloc[testIndex[1:], key + 1].values - y.iloc[testIndex[:-1], key + 1].values))))
                MASE3.append(sum(abs(ynew3[key].yhat[-len(testIndex):].values - y.iloc[testIndex, key+1].values))/((len(testIndex)/(len(testIndex)-1))*sum(abs(y.iloc[testIndex[1:], key + 1].values - y.iloc[testIndex[:-1], key + 1].values))))
                MASE4.append(sum(abs(ynew4[key].yhat[-len(testIndex):].values - y.iloc[testIndex, key+1].values))/((len(testIndex)/(len(testIndex)-1))*sum(abs(y.iloc[testIndex[1:], key + 1].values - y.iloc[testIndex[:-1], key + 1].values))))
                MASE5.append(sum(abs(ynew5[key].yhat[-len(testIndex):].values - y.iloc[testIndex, key+1].values))/((len(testIndex)/(len(testIndex)-1))*sum(abs(y.iloc[testIndex[1:], key + 1].values - y.iloc[testIndex[:-1], key + 1].values))))
        ##
        # If the method has the minimum Average MASE, use it on all of the data
        ##
        choice = min([np.mean(MASE1), np.mean(MASE2), np.mean(MASE3), np.mean(MASE4), np.mean(MASE5)])
        
        if choice == np.mean(MASE1):
            ynew = fitForecast.optimalComb(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                               yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                               changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            print("OC")
        if choice == np.mean(MASE2):
            ynew = fitForecast.forecastProp(y, h, nodes, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            print("FP")
        if choice == np.mean(MASE3):
            ynew = fitForecast.propHistAvg(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                               yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                               changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            print("PHA")
        if choice == np.mean(MASE4):
            ynew = fitForecast.averageHistProp(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                                   yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                   changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            print("AHP")
        if choice == np.mean(MASE5):
            ynew = fitForecast.bottomUp(y, h, sumMat, freq, include_history, cap, capF, changepoints, n_changepoints, \
                            yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                            changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            print("BU")

    return ynew
