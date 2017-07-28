# -*- coding: utf-8 -*-
"""

Name: runHTS.py
Author: Collin Rooney
Last Updated: 7/25/2017

This script will allow a user to quickly and easily run the hts package w/ prophet
for any data

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages.

"""

import pandas as pd
import sys
import hts
import htsPlot
import numpy as np

#%% Random data (Change this to whatever data you want)
date = pd.date_range("2014-04-02", "2017-07-17")
medium = ["Air", "Land", "Sea"]
businessMarket = ["Birmingham","Auburn","Evanston"]
platform = ["Stone Tablet","Car Phone"]
mediumDat = np.random.choice(medium, len(date))
busDat = np.random.choice(businessMarket, len(date))
platDat = np.random.choice(platform, len(date))
sessions = np.random.randint(10,1000,size=(len(date),1))
data = pd.DataFrame(date, columns = ["day"])
data["medium"] = mediumDat
data["platform"] = platDat
data["businessMarket"] = busDat
data["sessions"] = sessions

#%% Roll-up data to week level 
def makeWeekly(data):
    columnlist = data.columns.tolist()
    columnCount = len(columnlist)-2
    if columnCount < 1:
        sys.exit("you need at least 1 column")
    data[columnlist[0]] = pd.to_datetime(data[columnlist[0]])
    if columnCount < 2:
        data1 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1]], as_index = False)[columnlist[-1]].sum()
        data2 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1]])[columnlist[-1]].sum()
    elif columnCount < 3:
        data1 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1],columnlist[2]], as_index = False)[columnlist[-1]].sum()
        data2 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1],columnlist[2]])[columnlist[-1]].sum()
    elif columnCount < 4:
        data1 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1],columnlist[2],columnlist[3]], as_index = False)[columnlist[-1]].sum()
        data2 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1],columnlist[2],columnlist[3]])[columnlist[-1]].sum()
    else:
        data1 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1],columnlist[2],columnlist[3],columnlist[4]], as_index = False)[columnlist[-1]].sum()
        data2 = data.groupby([pd.Grouper(key = columnlist[0], freq='W'),columnlist[1],columnlist[2],columnlist[3],columnlist[4]])[columnlist[-1]].sum()
    data1['week'] = data2.index.get_level_values(columnlist[0])
    cols = data1.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data1 = data1[cols]
    return data1

#%% Create Ordering Function
def orderHier(data, col1 = None, col2 = None, col3 = None, col4 = None, rmZeros = False):
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
    if col1 not in [1,2,3]:
        sys.exit("col1 should equal 1, 2, 3, or 4")
    if col2 not in [1,2,3]:
        sys.exit("col2 should equal 1, 2, 3, or 4")
    if col3 is not None and col3 not in [1,2,3]:
        sys.exit("col3 should equal 1, 2, 3, or 4")
    if col1 == col2 | col1 == col3 | col2 == col3:
        sys.exit("col1, col2, and col3 should all have different values")
    if col1 is None:
        sys.exit("You need at least 1 column specified")
    if col2 is None:
        orderList = [col1]
        dimList = [data.columns.tolist()[1]]
        uniqueList = [data.iloc[:,1].unique()]
        lengthList = [len(uniqueList[0])]
        numIn = 1
    elif col3 is None:
        orderList = [col1, col2]
        dimList = [data.columns.tolist()[1], data.columns.tolist()[2]]
        uniqueList = [data.iloc[:,1].unique(), data.iloc[:,2].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1])]
        numIn = 2
    elif col4 is None:
        orderList = [col1,col2,col3]
        dimList = [data.columns.tolist()[1],data.columns.tolist()[2],data.columns.tolist()[3]]
        uniqueList = [data.iloc[:,1].unique(), data.iloc[:,2].unique(), data.iloc[:,3].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1]), len(uniqueList[2])]
        numIn = 3
    else:
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
    allDataframes['total'] = (data.groupby([timeInterval])[numCol].sum()).to_frame()

    #Take the sessions column from all those dataframes and put them into y
    y = pd.DataFrame(data = data[timeInterval].unique(), index = range(len(data[timeInterval].unique())), columns = ['time'])
    y = pd.merge(y, allDataframes['total'], left_on = 'time', right_index = True)
    y.rename(columns = {numCol:'total'}, inplace = True)
    
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

#%% Run HTS
##
# Make the data weekly (optional)
##
data1 = makeWeekly(data)
##
# Put the data in the format to run HTS, and get the nodes input (a list of list that describes the hierarchical structure)
##
data2, nodes = orderHier(data1, 1, 2, 3)
##
# load in prophet inputs (Running HTS runs prophet, so all inputs should be gathered beforehand)
# Made up holiday data
##
holidates = pd.date_range("12/25/2013","12/31/2017", freq = 'A')
holidays = pd.DataFrame(["Christmas"]*5, columns = ["holiday"])
holidays["ds"] = holidates
holidays["lower_window"] = [-4]*5
holidays["upper_window"] = [0]*5
##
# Run hts with the CVselect function (this decides which hierarchical aggregation method to use based on minimum mean Mean Absolute Scaled Error)
# h (which is 12 here) - how many steps ahead you would like to forecast.  If youre using daily data you don't have to specify freq.
#
# NOTE: CVselect takes a while, so if you want results in minutes instead of half-hours pick a different method
##
myDict = hts.hts(data2, 52, nodes, holidays = holidays, freq = 'W', method = "cvSelect")
##
# This output is a dictionary of dataframes, so you can do any further analysis that you may want. It also allows you to plot the forecasts.
# Some functions I've made are: (1 means I'm plotting the total node)
##
htsPlot.plotNode(myDict, 1, h = 52, xlabel = "Week", ylabel = "Number of Sessions")
htsPlot.plotChild(myDict, 1, nodes, h = 52, xlabel = "Week", ylabel = "Number of Sessions", legend = ["Total","Air","Land","Sea"])
htsPlot.plotNodeComponents(myDict, column = 1, holidays = holidays)