# -*- coding: utf-8 -*-
"""
Name: htsPlot.py
Author: Collin Rooney
Last Updated: 7/17/2017

This script will contain functions for plotting the output of the hts.py file
These plots will be made to look like the plots Prophet creates

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import sys

#%%
def plotNode(dictframe, column, h = 1, xlabel = 'ds', ylabel = 'y', startFrom = 0, uncertainty = False, ax = None):
    '''
    Parameters
    ------------------
    
    dictframe - (dict) The dictionary of dataframes that is the output of the hts function
    
    column - (string) column title that you want to plot
    
    h - (int) number of steps in the forecast same as input to hts function
    
    xlabel - (string) label for the graph's x axis
    
    ylabel - (string) label for the graph's y axis
    
    start_from - (int) the number of values to skip at the beginning of yhat so that you can zoom in
    
    uncertainty - (Boolean) include the prediction intervals or not
    
    ax - (axes object) any axes object thats already created that you want to pass to the plot function
    
    Returns
    ------------------
    
    plot of that node's forecast
    
    '''
    nodeToPlot = dictframe[column]
    
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ##
    # plot the yhat forecast as a solid line and then the h-step ahead forecast as a dashed line
    ##
    ax.plot(nodeToPlot['ds'].values[startFrom:-h], nodeToPlot['yhat'][startFrom:-h], ls='-', c='#0072B2')
    ax.plot(nodeToPlot['ds'].values[-h:], nodeToPlot['yhat'][-h:], dashes = [2,1])
    ##
    # plot the cap and uncertainty if necessary
    ##
    if 'cap' in nodeToPlot:
        ax.plot(nodeToPlot['ds'].values[startFrom:], nodeToPlot['cap'][startFrom:], ls='--', c='k')
    if uncertainty:
        ax.fill_between(nodeToPlot['ds'].values[startFrom:], nodeToPlot['yhat_lower'][startFrom:],
                        nodeToPlot['yhat_upper'][startFrom:], color='#0072B2',
                        alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

#%%
def plotWeekly(dictframe, ax, uncertainty, weeklyStart, color='#0072B2'):

    if ax is None:
        figW = plt.figure(facecolor='w', figsize=(10, 6))
        ax = figW.add_subplot(111)
    else:
        figW = ax.get_figure()
    ##
    # Create a list of 7 days for the x axis of the plot
    ##
    days = (pd.date_range(start='2017-01-01', periods=7) +
            pd.Timedelta(days=weeklyStart))
    ##
    # Find the weekday seasonality values for each weekday
    ##
    weekdays = dictframe.ds.dt.weekday
    ind = []
    for weekday in range(7):
        ind.append(max(weekdays[weekdays == weekday].index.tolist()))
    ##
    # Plot only one weekday each
    ##
    ax.plot(range(len(days)), dictframe['weekly'][ind], ls='-', c=color)
    ##
    # Plot uncertainty if necessary
    ##
    if uncertainty:
        ax.fill_between(range(len(days)),dictframe['weekly_lower'][ind], dictframe['weekly_upper'][ind],color=color, alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(dictframe['ds'][ind].dt.weekday_name)
    ax.set_xlabel('Day of week')
    ax.set_ylabel('weekly')
    figW.tight_layout()
    return figW
    
def plotYearly(dictframe, ax, uncertainty, color='#0072B2'):

    if ax is None:
        figY = plt.figure(facecolor='w', figsize=(10, 6))
        ax = figY.add_subplot(111)
    else:
        figY = ax.get_figure()
    ##
    # Find the max index for an entry of each month
    ##
    months = dictframe.ds.dt.month
    ind = []
    for month in range(1,13):
        ind.append(max(months[months == month].index.tolist()))
    ##
    # Plot from the minimum of those maximums on (this will almost certainly result in only 1 year plotted)
    ##
    ax.plot(dictframe['ds'][min(ind):], dictframe['yearly'][min(ind):], ls='-', c=color)
    if uncertainty:
        ax.fill_between(dictframe['ds'].values[min(ind):], dictframe['yearly_lower'][min(ind):], dictframe['yearly_upper'][min(ind):], color=color, alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos=None: '{dt:%B} {dt.day}'.format(dt=num2date(x))))
    ax.xaxis.set_major_locator(months)
    ax.set_xlabel('Day of year')
    ax.set_ylabel('yearly')
    figY.tight_layout()
    return figY

def plotHolidays(dictframe, holidays, ax, uncertainty, color='#0072B2'):
    ##
    # This function is largely the same as the one in Prophet
    ##
    if ax is None:
        figH = plt.figure(facecolor='w', figsize=(10, 6))
        ax = figH.add_subplot(111)
    else:
        figH = ax.get_figure()
    holidayComps = holidays.holiday.unique().tolist()
    yHoliday = dictframe[holidayComps].sum(1)
    yHolidayL = dictframe[[h + '_lower' for h in holidayComps]].sum(1)
    yHolidayU = dictframe[[h + '_upper' for h in holidayComps]].sum(1)
    # NOTE the above CI calculation is incorrect if holidays overlap
    # in time. Since it is just for the visualization we will not
    # worry about it now.
    ax.plot(dictframe['ds'].values, yHoliday, ls='-',
                       c=color)
    if uncertainty:
        ax.fill_between(dictframe['ds'].values, yHolidayL, yHolidayU, color=color, alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    ax.set_ylabel('holidays')
    figH.tight_layout()
    return figH

def plotTrend(dictframe, ax, uncertainty, plotCap, color='#0072B2'):
    ##
    # This function is largely the same as the one in Prophet
    ##
    if ax is None:
        figT = plt.figure(facecolor='w', figsize=(10, 6))
        ax = figT.add_subplot(111)
    else:
        figT = ax.get_figure()
    ax.plot(dictframe['ds'].values, dictframe['trend'], ls='-', c=color)
    if 'cap' in dictframe and plotCap:
        ax.plot(dictframe['ds'].values, dictframe['cap'], ls='--', c='k')
    if uncertainty:
       ax.fill_between(dictframe['ds'].values, dictframe['trend_lower'], dictframe['trend_upper'], color=color, alpha=0.2)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel('ds')
    ax.set_ylabel('trend')
    figT.tight_layout()
    return figT

def plotNodeComponents(dictframe, column, holidays = None, uncertainty=False, plotCap=False, weeklyStart = 0, ax=None,):
    '''
    Parameters
    ------------------
    
    dictframe - (dict) The dictionary of dataframes that is the output of the hts function
    
    column - (string) column title that you want to plot
    
    uncertainty - (Boolean) include the prediction intervals or not
    
    plot_cap - (Boolean) include the cap lines or not
    
    weekly_start - (int) an integer that specifies the first day on the x axis of the plot
    
    ax - (axes object) any axes object thats already created that you want to pass to the plot function
    
    Returns
    ------------------
    
    plot of that node's trend, seasonalities, holidays, etc.
    
    '''
    nodeToPlot = dictframe[column]
    colNames = nodeToPlot.columns.tolist()
    trend = "trend" in colNames
    if holidays is not None:
        holiday = np.any(holidays.holiday[0] in colNames)
    weekly = "weekly" in colNames
    yearly = "yearly" in colNames

    if trend:
        plotTrend(nodeToPlot, ax=ax, uncertainty=uncertainty, plotCap=plotCap)
    if holiday:
        plotHolidays(nodeToPlot, holidays=holidays, ax=ax, uncertainty=uncertainty)
    if weekly:
        plotWeekly(nodeToPlot, ax=ax, uncertainty=uncertainty, weeklyStart = weeklyStart)
    if yearly:
        plotYearly(nodeToPlot, ax=ax, uncertainty=uncertainty)
    
    return

#%%
def plotChild(dictframe, column, h = 1, xlabel = 'ds', ylabel = 'y', startFrom = 0, uncertainty = False, ax = None):
    '''
    Parameters
    ------------------
    
    dictframe - (dict) The dictionary of dataframes that is the output of the hts function
    
    column - (string) column title that you want to plot
    
    h - (int) number of steps in the forecast same as input to hts function
    
    xlabel - (string) label for the graph's x axis
    
    ylabel - (string) label for the graph's y axis
    
    start_from - (int) the number of values to skip at the beginning of yhat so that you can zoom in
    
    uncertainty - (Boolean) include the prediction intervals or not
    
    ax - (axes object) any axes object thats already created that you want to pass to the plot function
    
    Returns
    ------------------
    
    plot of that node and its children's forecast
    
    '''
    ##
    # Set the color map to brg so that there are enough dark and discernably different choices
    ##
    cmap = plt.get_cmap('tab10')
    ##
    # Find the children nodes
    ##
    colOptions = list(dictframe.keys())
    allChildren = [s for s in colOptions if column in s]
    countChildren = [s.count('_') for s in colOptions if column in s]
    if min(countChildren)+1 not in countChildren and column != "Total":
        sys.exit("the specified column doesn't have children")
    if min(countChildren)+2 not in countChildren:
        columnsToPlot = allChildren
    else:
        ind = countChildren.index(min(countChildren)+2)
        columnsToPlot = allChildren[0:ind]
    if column == 'Total':
        allChildren = [s for s in colOptions]
        countChildren = [s.count('_') for s in colOptions]
        if max(countChildren) > 0:
            ind = countChildren.index(min(countChildren)+1)
            columnsToPlot = allChildren[0:ind]
        else:
            columnsToPlot = allChildren
    ##
    # Plot the node and its children the same way as the plot_node function did it
    ##
    i = 0
    N = len(columnsToPlot)
    for column in columnsToPlot:
        nodeToPlot = dictframe[column]
        if ax is None:
            fig = plt.figure(facecolor='w', figsize=(10, 6))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.plot(nodeToPlot['ds'].values[startFrom:-h], nodeToPlot['yhat'][startFrom:-h], ls='-', c = cmap(float(i)/N), label = column)
        ax.plot(nodeToPlot['ds'].values[-h:], nodeToPlot['yhat'][-h:], dashes = [2,1], c = cmap(float(i)/N), label = '_nolegend_')
        if 'cap' in nodeToPlot:
            ax.plot(nodeToPlot['ds'].values[startFrom:], nodeToPlot['cap'][startFrom:], ls='--', c='k')
        if uncertainty:
            ax.fill_between(nodeToPlot['ds'].values[startFrom:], nodeToPlot['yhat_lower'][startFrom:],
                            nodeToPlot['yhat_upper'][startFrom:], color='#0072B2',
                            alpha=0.2)
        i+=1
    
    ax.grid(True, which='major', color='gray', ls='-', lw=1, alpha = 0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    
    return fig