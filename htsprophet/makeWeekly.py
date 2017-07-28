# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:24:23 2017

This function is just for my convenience.  It probably isn't too efficient, but if you only have 4 layers and want to make
daily data weekly, go at it.

@author: collin.rooney

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages.
"""

import pandas as pd
import sys

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
