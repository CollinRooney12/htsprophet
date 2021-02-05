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
from htsprophet.fitForecast import fitForecast
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from multiprocessing.dummy import Pool as ThreadPool


#%%
def hts(y, h = 1, nodes = [[2]], method='OLS', freq = 'D', transform = None, include_history = True, cap = None, capF = None, changepoints = None, \
        n_changepoints = 25, yearly_seasonality = 'auto', weekly_seasonality = 'auto', daily_seasonality = 'auto', holidays = None, seasonality_prior_scale = 10.0, \
        holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05, mcmc_samples = 0, interval_width = 0.80, uncertainty_samples = 0, skipFitting = False, numThreads = 0):
    '''
    Parameters
    ----------------
     y - dataframe of time-series data, or if you want to skip fitting, a dictionary of prophet base forecast dataframes
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
                "OLS" - optimal combination by Original Least Squares (Default), 
                "WLSS" - optimal combination by Structurally Weighted Least Squares
                "WLSV" - optimal combination by Error Variance Weighted Least Squares
                "FP" - forcasted proportions (top-down)
                "PHA" - proportions of historical averages (top-down)
                "AHP" - average historical proportions (top-down)
                "BU" - bottom-up (simple addition)
                "CVselect" - select which method is best for you based on 3-fold Cross validation (longer run time)
     
     freq - (Time Frequency) input for the forecasting function of Prophet 
     
     transform - (None or "BoxCox") Do you want to transform your data before fitting the prophet function? If yes, type "BoxCox"
     
     include_history - (Boolean) input for the forecasting function of Prophet
                
     cap - (Dataframe or Constant) carrying capacity of the input time series.  If it is a dataframe, then
                                   the number of columns must equal len(y.columns) - 1
                                   
     capF - (Dataframe or Constant) carrying capacity of the future time series.  If it is a dataframe, then
                                    the number of columns must equal len(y.columns) - 1
     
     changepoints - (DataFrame or List) changepoints for the model to consider fitting. If it is a dataframe, then
                                        the number of columns must equal len(y.columns) - 1
     
     n_changepoints - (constant or list) changepoints for the model to consider fitting. If it is a list, then
                                         the number of items must equal len(y.columns) - 1
                                         
     skipFitting - (Boolean) if y is already a dictionary of dataframes, set this to True, and DO NOT run with method = "cvSelect" or transform = "BoxCox"
     
     numThreads - (int) number of threads you want to use when running cvSelect. Note: 14 has shown to decrease runtime by 10 percent 
                                 
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
    if method not in ['OLS','WLSS','WLSV','FP','PHA','AHP','BU','cvSelect']:
        sys.exit("not a valid method input, must be one of the following: 'OLS','WLSS','WLSV','FP','PHA','AHP','BU','cvSelect'")
    if len(nodes) < 1:
        sys.exit("nodes input should at least be of length 1")
    if not isinstance(cap, int) and not isinstance(cap, pd.DataFrame) and not isinstance(cap, float) and not cap is None:
        sys.exit("cap should be a constant (float or int) or a DataFrame, or not specified")
    if not isinstance(capF, int) and not isinstance(capF, pd.DataFrame) and not isinstance(capF, float) and not capF is None:
        sys.exit("capF should be a constant (float or int) or a DataFrame, or not specified")
    if not isinstance(y, dict):
        if sum(list(map(sum, nodes))) != len(y.columns)-2:
            sys.exit("The sum of the nodes list does not equal the number of columns - 2, dataframe should contain a time column in the 0th pos. Double check node input")
        if isinstance(cap, pd.DataFrame):
            if len(cap.columns) != len(y.columns)-1:
                sys.exit("If cap is a DataFrame, it should have a number of columns equal to the input Dataframe - 1")
        if isinstance(capF, pd.DataFrame):
            if len(capF.columns) != len(y.columns)-1:
                sys.exit("If capF is a DataFrame, it should have a number of columns equal to the input Dataframe - 1")
    if cap is not None and method not in ["BU","FP","AHP","PHA"]:
        print("Consider using BU, FP, AHP, or PHA.  The other methods can create negatives which would cause problems for the log() function")
    ##
    # Transform Variables
    ##
    if transform is not None:
        if transform == 'BoxCox':
            y2 = y.copy()
            import warnings
            warnings.simplefilter("error", RuntimeWarning)
            boxcoxT = [None]*(len(y.columns.tolist())-1)
            try:
                for column in range(len(y.columns.tolist())-1):
                    y2.iloc[:,column+1], boxcoxT[column] = boxcox(y2.iloc[:, column+1])
                y = y2
            ##
            # Does a Natural Log Transform if scipy's boxcox cant deal
            ##
            except RuntimeWarning:
                print("It looks like scipy's boxcox function couldn't deal with your data. Proceeding with Natural Log Transform")
                for column in range(len(y.columns.tolist())-1):
                    y.iloc[:,column+1] = boxcox(y.iloc[:, column+1], lmbda = 0)
                    boxcoxT[column] = 0
        else:
            print("Nothing will be transformed because the input was not = to 'BoxCox'")
    else:
        boxcoxT = None
    ##
    # Run specified approach
    ##
    if method == 'cvSelect':
        ##
        # Run all of the Methods and let 3 fold CV chose which is best for you
        ##
        methodList = ['WLSV','WLSS','OLS','FP','PHA','AHP','BU']
        sumMat = SummingMat(nodes)
        tscv = TimeSeriesSplit(n_splits=3)
        MASE1 = []
        MASE2 = []
        MASE3 = []
        MASE4 = []
        MASE5 = []
        MASE6 = []
        MASE7 = []
        ##
        # Split into train and test, using time series split, and predict the test set
        ##
        y1 = y.copy()
        if boxcoxT is not None:
                for column in range(len(y.columns.tolist())-1):
                    y1.iloc[:,column+1] = inv_boxcox(y1.iloc[:, column+1], boxcoxT[column])
                    
        for trainIndex, testIndex in tscv.split(y.iloc[:,0]):
            if numThreads != 0:
                pool = ThreadPool(numThreads)
                results = pool.starmap(fitForecast, zip([y.iloc[trainIndex, :]]*7, [len(testIndex)]*7, [sumMat]*7, [nodes]*7, methodList, [freq]*7, [include_history]*7, [cap]*7, [capF]*7, [changepoints]*7, [n_changepoints]*7, \
                                    [yearly_seasonality]*7, [weekly_seasonality]*7, [daily_seasonality]*7, [holidays]*7, [seasonality_prior_scale]*7, [holidays_prior_scale]*7,\
                                    [changepoint_prior_scale]*7, [mcmc_samples]*7, [interval_width]*7, [uncertainty_samples]*7,  [boxcoxT]*7, [skipFitting]*7))
                pool.close()
                pool.join()
                ynew1, ynew2, ynew3, ynew4, ynew5, ynew6, ynew7 = results
            else:
                ynew1 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[0], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
                ynew2 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[1], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
                ynew3 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[2], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
                ynew4 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[3], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
                ynew5 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[4], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
                ynew6 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[5], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
                ynew7 = fitForecast(y.iloc[trainIndex, :], len(testIndex), sumMat, nodes, methodList[6], freq, include_history, cap, capF, changepoints, n_changepoints, \
                                    yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                                    changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
#                    
            for key in ynew1.keys():
                MASE1.append(np.mean(abs(ynew1[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
                MASE2.append(np.mean(abs(ynew2[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
                MASE3.append(np.mean(abs(ynew3[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
                MASE4.append(np.mean(abs(ynew4[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
                MASE5.append(np.mean(abs(ynew5[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
                MASE6.append(np.mean(abs(ynew6[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
                MASE7.append(np.mean(abs(ynew7[key].yhat[-len(testIndex):].values - y1.iloc[testIndex, key+1].values)))
        ##
        # If the method has the minimum Average MASE, use it on all of the data
        ##
        choices = [np.mean(MASE1), np.mean(MASE2), np.mean(MASE3), np.mean(MASE4), np.mean(MASE5), np.mean(MASE6), np.mean(MASE7)]
        choice = methodList[choices.index(min(choices))]
        ynew = fitForecast(y, h, sumMat, nodes, choice, freq, include_history, cap, capF, changepoints, n_changepoints, \
                           yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                           changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
        print(choice)
    
    else:    
        if skipFitting == True:
            theDictionary = y
            i = 0
            for key in y.keys():
                if i == 0:
                    y = pd.DataFrame(theDictionary[key].ds)
                y[i] = theDictionary[key].yhat
                i += 1
        sumMat = SummingMat(nodes)
        ynew = fitForecast(y, h, sumMat, nodes, method, freq, include_history, cap, capF, changepoints, n_changepoints, \
                           yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                           changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT, skipFitting)
    ##
    # Inverse boxcox the data
    ##    
    if transform is not None:
        if transform == 'BoxCox':
            for column in range(len(y.columns.tolist())-1):
                y.iloc[:,column+1] = inv_boxcox(y.iloc[:, column+1], boxcoxT[column])
    ##
    # Put the values back in the dictionary for skipFitting
    ##
    if skipFitting == True:
        i = 0
        for key in theDictionary.keys():
            for column in theDictionary[key].columns:
                if column == 'yhat':
                    continue
                ynew[key][column] = theDictionary[key][column]
    ##
    # Rename keys so that dictionary can be easily understood
    ##

    i = -2
    for column in y:
        i += 1   
        if i == -1:
            continue
        else:
            ynew[column] = ynew.pop(i)
    
    return ynew

#%% Roll-up data to week level 
def makeWeekly(data):
    columnList = data.columns.tolist()
    columnCount = len(columnList)-2
    if columnCount < 1:
        sys.exit("you need at least 1 column")
    data[columnList[0]] = pd.to_datetime(data[columnList[0]])
    cl = tuple(columnList[1:-1])
    data1 = data.groupby([pd.Grouper(key = columnList[0], freq='W'), *cl], as_index = False)[columnList[-1]].sum()
    data2 = data.groupby([pd.Grouper(key = columnList[0], freq='W'), *cl])[columnList[-1]].sum()
    data1['week'] = data2.index.get_level_values(columnList[0])
    cols = data1.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data1 = data1[cols]
    return data1

#%% Create Ordering Function
def orderHier(data, col1 = 1, col2 = None, col3 = None, col4 = None, rmZeros = False):
    # 
    #This function will order the hierarchy the way you like it as long as you are
    #using max 4 layers
    #
    #
    #Inputs and desc:
    # 
    # Data - (pandas DataFrame) the data you want made into a hierarchical model
    #                           The first column should specify the time
    #                           The middle columns should be the names of the layers of the hierarchy (ex. Medium, Platform, BusinessMarket, etc.)
    #                           The last column should be the numeric column that you would like to forecast
    #
    # col1 - (int [1-4]) what layer you want the first column (thats not a timestamp) to be at
    #           Layers:
    #               1 - level right below total
    #               2 - level below 1
    #               3 - level below 2
    #               4 - Bottom Level
    #
    # col2 - (int [1-4]) what layer you want the second column (thats not a timestamp) to be at
    #           Layers:
    #               1 - level right below total
    #               2 - level below 1
    #               3 - level below 2
    #               4 - Bottom Level
    #
    # col3 - (int [1-4]) what layer you want the third column (thats not a timestamp) to be at
    #           Layers:
    #               1 - level right below total
    #               2 - level below 1
    #               3 - level below 2
    #               4 - Bottom Level
    #
    # 
    # col4 - (int [1-4]) what layer you want the fourth column (thats not a timestamp) to be at
    #           Layers:
    #               1 - level right below total
    #               2 - level below 1
    #               3 - level below 2
    #               4 - Bottom Level
    #
    #
    # Outputs and desc:
    # 
    # y - (pandas Dataframe) the DataFrame in a format that is consistent with the
    #                        Hierarchy function and ordered in a way that the user
    #                         specified.
    #
    #
    if col1 not in [1,2,3,4]:
        sys.exit("col1 should equal 1, 2, 3, or 4")
    if col2 is not None and col2 not in [1,2,3,4]:
        sys.exit("col3 should equal 1, 2, 3, or 4")
    if col3 is not None and col3 not in [1,2,3,4]:
        sys.exit("col3 should equal 1, 2, 3, or 4")
    if col1 is None:
        sys.exit("You need at least 1 column specified")
    if col2 is None:
        orderList = [col1]
        dimList = [data.columns.tolist()[1]]
        uniqueList = [data.iloc[:,1].unique()]
        lengthList = [len(uniqueList[0])]
        numIn = 1
    elif col3 is None:
        if col1 == col2:
            sys.exit("col1, col2 should all have different values")
        orderList = [col1, col2]
        dimList = [data.columns.tolist()[1], data.columns.tolist()[2]]
        uniqueList = [data.iloc[:,1].unique(), data.iloc[:,2].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1])]
        numIn = 2
    elif col4 is None:
        if col1 == col2 or col1 == col3 or col2 == col3:
            sys.exit("col1, col2, and col3 should all have different values")
        orderList = [col1,col2,col3]
        dimList = [data.columns.tolist()[1],data.columns.tolist()[2],data.columns.tolist()[3]]
        uniqueList = [data.iloc[:,1].unique(), data.iloc[:,2].unique(), data.iloc[:,3].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1]), len(uniqueList[2])]
        numIn = 3
    else:
        if col1 == col2 or col1 == col3 or col2 == col3 or col1 == col4 or col2 == col4 or col3 == col4:
            sys.exit("col1, col2, col3, and col4 should all have different values")
        orderList = [col1,col2,col3,col4]
        dimList = [data.columns.tolist()[1],data.columns.tolist()[2],data.columns.tolist()[3],data.columns.tolist()[4]]
        uniqueList = [data.iloc[:,1].unique(), data.iloc[:,2].unique(), data.iloc[:,3].unique(), data.iloc[:,4].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1]), len(uniqueList[2]), len(uniqueList[3])]
        numIn = 4
    
    numCol = data.columns.tolist()[-1]
    timeInterval = data.columns.tolist()[0]
    
    allDataframes = {}
    
    #Creating dataframes for the top level of the hierarchy (not total)
    for num in range(lengthList[orderList.index(1)]):
        allDataframes[uniqueList[orderList.index(1)][num]] = data.loc[data[dimList[orderList.index(1)]] == uniqueList[orderList.index(1)][num]]
        allDataframes[uniqueList[orderList.index(1)][num]+'1'] = (allDataframes[uniqueList[orderList.index(1)][num]].groupby([timeInterval])[numCol].sum()).to_frame()
        
        if numIn > 1:
            #Creating dataframes for the second level of the hierarchy
            placeholder = allDataframes[uniqueList[orderList.index(1)][num]].groupby([timeInterval, dimList[orderList.index(2)]])[numCol].sum()
            for ind in range(lengthList[orderList.index(2)]):
                allDataframes[uniqueList[orderList.index(1)][num]+'_'+uniqueList[orderList.index(2)][ind]] = (placeholder.loc[(placeholder.index.get_level_values(1) == uniqueList[orderList.index(2)][ind])]).to_frame()
                
                if numIn > 2:
                    placeholder1 = allDataframes[uniqueList[orderList.index(1)][num]].groupby([timeInterval, dimList[orderList.index(2)], dimList[orderList.index(3)]])[numCol].sum()
                    #Creating dataframes for the third level of the hierarchy
                    for cnt in range(lengthList[orderList.index(3)]):
                        allDataframes[uniqueList[orderList.index(1)][num]+'_'+uniqueList[orderList.index(2)][ind]+'_'+uniqueList[orderList.index(3)][cnt]] = (placeholder1.loc[(placeholder1.index.get_level_values(1) == uniqueList[orderList.index(2)][ind]) & (placeholder1.index.get_level_values(2) == uniqueList[orderList.index(3)][cnt])]).to_frame()
    
                        if numIn > 3:
                            placeholder2 = allDataframes[uniqueList[orderList.index(1)][num]].groupby([timeInterval, dimList[orderList.index(2)], dimList[orderList.index(3)], dimList[orderList.index(4)]])[numCol].sum()
                            #Creating dataframes for the third level of the hierarchy
                            for pos in range(lengthList[orderList.index(4)]):
                                allDataframes[uniqueList[orderList.index(1)][num]+'_'+uniqueList[orderList.index(2)][ind]+'_'+uniqueList[orderList.index(3)][cnt]+'_'+uniqueList[orderList.index(4)][pos]] = (placeholder2.loc[(placeholder2.index.get_level_values(1)\
                                               == uniqueList[orderList.index(2)][ind]) & (placeholder2.index.get_level_values(2) == uniqueList[orderList.index(3)][cnt]) & (placeholder2.index.get_level_values(3) == uniqueList[orderList.index(4)][pos])]).to_frame()
    
    #Creating total DataFrame
    allDataframes['Total'] = (data.groupby([timeInterval])[numCol].sum()).to_frame()

    #Take the sessions column from all those dataframes and put them into y
    y = pd.DataFrame(data = data[timeInterval].unique(), index = range(len(data[timeInterval].unique())), columns = ['time'])
    y = pd.merge(y, allDataframes['Total'], left_on = 'time', right_index = True)
    y.rename(columns = {numCol:'Total'}, inplace = True)
    
    for i in range(lengthList[orderList.index(1)]):
        y = pd.merge(y, allDataframes[uniqueList[orderList.index(1)][i]+'1'], how = 'left', left_on = 'time', right_index = True)
        y.rename(columns = {numCol:uniqueList[orderList.index(1)][i]}, inplace = True)
        if numIn > 1:
            for j in range(lengthList[orderList.index(2)]):
                allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]].index = allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]].index.droplevel(1)
                y = pd.merge(y, allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]], how = 'left', left_on = 'time', right_index = True)
                y.rename(columns = {numCol:uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]}, inplace = True)
                if numIn > 2:
                    for k in range(lengthList[orderList.index(3)]):
                        allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]].index = allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]].index.droplevel(2)
                        allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]].index = allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]].index.droplevel(1)
                        y = pd.merge(y, allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]], how = 'left', left_on = 'time', right_index = True)
                        y.rename(columns = {numCol:uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]}, inplace = True)
                        if numIn > 3:
                            for l in range(lengthList[orderList.index(4)]):
                                allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]].index = allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]].index.droplevel(3)
                                allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]].index = allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]].index.droplevel(2)
                                allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]].index = allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]].index.droplevel(1)
                                y = pd.merge(y, allDataframes[uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]], how = 'left', left_on = 'time', right_index = True)
                                y.rename(columns = {numCol:uniqueList[orderList.index(1)][i]+'_'+uniqueList[orderList.index(2)][j]+'_'+uniqueList[orderList.index(3)][k]+'_'+uniqueList[orderList.index(4)][l]}, inplace = True)
    
    
    if rmZeros == True:
        #Get rid of Missing columns and rows
        y.dropna(axis = 1, how = 'any', thresh = len(y['time'])/2, inplace = True)
        y.dropna(axis = 0, how = 'any', inplace = True)
    else:
        y = y.fillna(1)
    #Re-order the columns so that it is compatible with Hierarchy function
    cols = y.columns.tolist()
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for col in cols:
        if col.count('_') == 0:
            list1.append(col)
        if col.count('_') == 1:
            list2.append(col)
        if col.count('_') == 2:
            list3.append(col)
        if col.count('_') == 3:
            list4.append(col)
    
    newOrder = []
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for item in list1:
        newOrder.append(item)
        count1 += 1
    for item in list2:
        newOrder.append(item)
        count2 += 1
    for item in list3:
        newOrder.append(item)
        count3 += 1
    for item in list4:
        newOrder.append(item)
        count4 += 1
        
    y = y[newOrder]
    
    ##
    # Create Nodes variable (A list of lists that describes the hierarchical structure)
    ##
    nodes = []
    nodes.append([count1-2])
    if numIn > 1:
        numberList = []
        for column in range(2, count1):
            number = sum([i.count(y.columns.tolist()[column]) for i in y.columns.tolist()[count1:count1+count2]])
            numberList.append(number)
        nodes.append(numberList)
    if numIn > 2:
        numberList = []
        for column in range(count1, count1+count2):
            number = sum([i.count(y.columns.tolist()[column]) for i in y.columns.tolist()[count1+count2:count1+count2+count3]])
            numberList.append(number)
        nodes.append(numberList)
    if numIn > 3:
        numberList = []
        for column in range(count1+count2, count1+count2+count3):
            number = sum([i.count(y.columns.tolist()[column]) for i in y.columns.tolist()[count1+count2+count3:count1+count2+count3+count4]])
            numberList.append(number)
        nodes.append(numberList)

    return y, nodes

