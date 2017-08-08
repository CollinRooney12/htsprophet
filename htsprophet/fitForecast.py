# -*- coding: utf-8 -*-
"""
Name: fitForecast.py
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
from fbprophet import Prophet
import contextlib, os
from scipy.special import inv_boxcox
from scipy.stats import boxcox

#%%
def fitForecast(y, h, sumMat, nodes, method, freq, include_history, cap, capF, changepoints, n_changepoints, \
                yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT):
    
    forecastsDict = {}
    nForecasts = sumMat.shape[0]
    if method == 'FP':
        nForecasts = sum(list(map(sum, nodes)))+1
    
    for node in range(nForecasts):
        nodeToForecast = pd.concat([y.iloc[:, [0]], y.iloc[:, node+1]], axis = 1)
        if isinstance(cap, pd.DataFrame):
            cap1 = cap.iloc[:, node]
        else:
            cap1 = cap
        if isinstance(capF, pd.DataFrame):    
            cap2 = capF.iloc[:, node]
        else:
            cap2 = capF
        if isinstance(changepoints, pd.DataFrame):
            changepoints1 = changepoints[:, node]
        else:
            changepoints1 = changepoints
        if isinstance(n_changepoints, list):
            n_changepoints1 = n_changepoints[node]
        else:
            n_changepoints1 = n_changepoints
        ##
        # Put the forecasts into a dictionary of dataframes
        ##
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            # Prophet related stuff
            nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
            nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
            if capF is None:
                growth = 'linear'
                m = Prophet(growth, changepoints1, n_changepoints1, yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, \
                            holidays_prior_scale, changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            else:
                growth = 'logistic'
                m = Prophet(growth, changepoints, n_changepoints, yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, \
                            holidays_prior_scale, changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
                nodeToForecast['cap'] = cap1
            m.fit(nodeToForecast)
            future = m.make_future_dataframe(periods = h, freq = freq, include_history = include_history)
            if capF is not None:
                future['cap'] = cap2
            ##
            # Base Forecasts
            ##
            forecastsDict[node] = m.predict(future)
            ##
            # If logistic use exponential function, so that values can be added correctly
            ##
            if capF is not None:
                forecastsDict[node].yhat = np.exp(forecastsDict[node].yhat)
            if boxcoxT is True:
                lmbda = []*nForecasts
                forecastsDict[node].yhat, lmbda[node] = inv_boxcox(forecastsDict[node].yhat)
    ##
    # Now, Revise them
    ##
    if method == 'BU' or method == 'AHP' or method == 'PHA':
        nCols = len(list(forecastsDict.keys()))+1
        if method == 'BU':
            '''
             Pros:
               No information lost due to aggregation
             Cons:
               Bottom level data can be noisy and more challenging to model and forecast
            '''
            hatMat = np.zeros([len(forecastsDict[0].yhat),1]) 
            for key in range(nCols-sumMat.shape[1]-1, nCols-1):
                f1 = np.array(forecastsDict[key].yhat)
                f2 = f1[:, np.newaxis]
                if np.all(hatMat == 0):
                    hatMat = f2
                else:
                    hatMat = np.concatenate((hatMat, f2), axis = 1)
            
        if method == 'AHP':
            '''
             Pros:
               Creates reliable aggregate forecasts, and good for low count data
             Cons:
               Unable to capture individual series dynamics
            '''
            ##
            # Find Proportions
            ##
            fcst = forecastsDict[0].yhat
            fcst = fcst[:, np.newaxis]
            numBTS = sumMat.shape[1]
            btsDat = pd.DataFrame(y.iloc[:,nCols-numBTS:nCols])
            divs = np.divide(np.transpose(np.array(btsDat)),np.array(y.iloc[:,1]))
            props = divs.mean(1)
            props = props[:, np.newaxis]
            hatMat = np.dot(np.array(fcst),np.transpose(props))
            
        if method == 'PHA':
            '''
             Pros:
               Creates reliable aggregate forecasts, and good for low count data
             Cons:
               Unable to capture individual series dynamics
            '''
            ##
            # Find Proportions
            ##
            fcst = forecastsDict[0].yhat
            fcst = fcst[:, np.newaxis]
            numBTS = sumMat.shape[1]
            btsDat = pd.DataFrame(y.iloc[:,nCols-numBTS:nCols])
            btsSum = btsDat.sum(0)
            topSum = sum(y.iloc[:,1])
            props = btsSum/topSum
            props = props[:, np.newaxis]
            hatMat = np.dot(np.array(fcst),np.transpose(props))
        
        newMat = np.empty([hatMat.shape[0],sumMat.shape[0]])
        for i in range(hatMat.shape[0]):
            newMat[i,:] = np.dot(sumMat, np.transpose(hatMat[i,:]))

    if method == 'FP':
        newMat = forecastProp(forecastsDict, nodes)
    if method == 'OC':
        newMat = optimalComb(forecastsDict, sumMat)
    
    for key in forecastsDict.keys():
        values = forecastsDict[key].yhat.values
        values = newMat[:,key]
        forecastsDict[key].yhat = values
        ##
        # If Logistic fit values with natural log function to revert back to format of input
        ##
        if capF is not None:
            forecastsDict[key].yhat = np.log(forecastsDict[key].yhat)
        if boxcoxT is True:
            forecastsDict[key].yhat = boxcox(forecastsDict[key].yhat, lmbda[key])
        
    return forecastsDict
    
#%%    
def forecastProp(forecastsDict, nodes):
    '''
     Cons:
       Produces biased revised forecasts even if base forecasts are unbiased
    '''
    nCols = len(list(forecastsDict.keys()))+1
    ##
    # Find proportions of forecast at each step ahead, and then alter forecasts
    ##
    levels = len(nodes)
    column = 0
    firstNode = 1
    newMat = np.empty([len(forecastsDict[0].yhat),nCols - 1])
    newMat[:,0] = forecastsDict[0].yhat
    lst = [x for x in range(nCols-1)]
    for level in range(levels):
        nodesInLevel = len(nodes[level])
        foreSum = 0
        for node in range(nodesInLevel):
            numChild = nodes[level][node]
            lastNode = firstNode + numChild
            lst = [x for x in range(firstNode, lastNode)]
            baseFcst = np.array([forecastsDict[k].yhat[:] for k in lst])
            foreSum = np.sum(baseFcst, axis = 0)
            foreSum = foreSum[:, np.newaxis]
            if column == 0:
                revTop = np.array(forecastsDict[column].yhat)
                revTop = revTop[:, np.newaxis]
            else:
                revTop = np.array(newMat[:,column])
                revTop = revTop[:, np.newaxis]
            newMat[:,firstNode:lastNode] = np.divide(np.multiply(np.transpose(baseFcst), revTop), foreSum)
            column += 1       
            firstNode += numChild    
    
    return newMat

#%%    
def optimalComb(forecastsDict, sumMat):

    hatMat = np.zeros([len(forecastsDict[0].yhat),1]) 
    for key in forecastsDict.keys():
        f1 = np.array(forecastsDict[key].yhat)
        f2 = f1[:, np.newaxis]
        if np.all(hatMat == 0):
            hatMat = f2
        else:
            hatMat = np.concatenate((hatMat, f2), axis = 1)
    ##
    # Multiply the Summing Matrix Together S*inv(S'S)*S'
    ##
    optiMat = np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.transpose(sumMat), sumMat))),np.transpose(sumMat))
    
    newMat = np.empty([hatMat.shape[0],sumMat.shape[0]])
    for i in range(hatMat.shape[0]):
        newMat[i,:] = np.dot(optiMat, np.transpose(hatMat[i,:]))
        
    return newMat
