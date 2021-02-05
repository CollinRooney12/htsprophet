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
from htsprophet.hts import hts, orderHier, makeWeekly
from htsprophet.htsPlot import plotNode, plotChild, plotNodeComponents
import numpy as np

#%% Random data (Change this to whatever data you want)
date = pd.date_range("2015-04-02", "2017-07-17")
date = np.repeat(date, 10)
medium = ["Air", "Land", "Sea"]
businessMarket = ["Birmingham","Auburn","Evanston"]
platform = ["Stone Tablet","Car Phone"]
mediumDat = np.random.choice(medium, len(date))
busDat = np.random.choice(businessMarket, len(date))
platDat = np.random.choice(platform, len(date))
sessions = np.random.randint(1000,10000,size=(len(date),1))
data = pd.DataFrame(date, columns = ["day"])
data["medium"] = mediumDat
data["platform"] = platDat
data["businessMarket"] = busDat
data["sessions"] = sessions

#%% Run HTS
##
# Make the daily data weekly (optional)
##
data1 = makeWeekly(data)
##
# Put the data in the format to run HTS, and get the nodes input (a list of list that describes the hierarchical structure)
##
data2, nodes = orderHier(data, 1, 2, 3)
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
myDict = hts(data2, 52, nodes, holidays = holidays, method = "FP", transform = "BoxCox")
##
# This output is a dictionary of dataframes, so you can do any further analysis that you may want. It also allows you to plot the forecasts.
# Some functions I've made are: (1 means I'm plotting the total node)
##
plotNode(myDict, "Total", h = 52, xlabel = "Week", ylabel = "Number of Sessions")
plotChild(myDict, "Total", h = 52, xlabel = "Week", ylabel = "Number of Sessions")
plotNodeComponents(myDict, column = "Total", holidays = holidays)