# htsprophet
Hierarchical Time Series Forecasting using Prophet

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work.

https://www.otexts.org/fpp

https://robjhyndman.com/publications/

Credit to Facebook and their fbprophet package.

https://facebookincubator.github.io/prophet/

It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages.

# Downloading

1. pip install htsprophet

If you'd like to just skip to coding with the package, **runHTS.py** should help you with that, but if you like reading, the following should help you understand how I built htsprophet and how it works.

# Part I: The Data

I originally used Redfin traffic data to build this package.

I pulled the data so that date was in the first column, my layers were the middle columns, and the number I wanted to forecast was in the last column.

I made a function called **makeWeekly()** (Assumes you have max 4 categorical columns), that rolls up your data into the weekly level.  It’s not a necessary function, it was mostly just convenient for me.

So the data looked like this:

|   Date   |   Platform   |   Medium   |   BusinessMarket   |   Sessions   |
|----------|--------------|------------|--------------------|--------------|
| 1100 B.C.| Stone Tablet |    Land    |     Birmingham     |	  23234    |
|   ...    |   Car Phone  |     Air    |       Auburn       |	  2342	   |
|          |      ...     |     Sea    |      Evanston      |	   233     |
|          |              |     ...    |       Seattle      |	   445     |
|          |              |            |	 ...	    |	  46362    |

I then ran my **orderHier()** function with just this dataframe as its input.  

**NOTE: you cannot run this function if you have more than 4 columns in the middle (in between Date and Sessions for ex.)**

To run this function, you specify the data, and how you want your middle columns to be ordered.  

So orderHier(data, 2, 1, 3) means you want the second column after date to be the first level of the hierarchy.

Our example would look like this:









Date	Total	Land	Air	Sea	Land_Stone tablet	Land_Car Phone	Air_Stone Tablet
1100 B.C.	24578	23135	555	888	23000	135	550
1099 B.C.	86753	86654	44	55	2342	84312	22
							
							
							
							

	*All numbers represent the number of sessions for each node in the Hierarchy


Because of the way orderHier was written, if you have more than 4 categorical columns, then you must get the data in this format on your own while also producing the list of lists called nodes
Nodes – describes the structure of the hierarchy.

Here it would equal [[3],[2,2,2],[4,4,4,4,4,4]]

There are 3 nodes in the first level: Land, Air, Sea.

There are 2 children for each of those nodes: Stone tablet, Car phone.

There are 4 business markets for each of those nodes: Tokyo, Hamburg etc.

If you use the orderHier function, nodes will be the second output of the function.

# Part II: Prophet Inputs

Anything that you would specify in Prophet you can specify in hts(). 

It’s flexible and will allow you to input a dataframe of values for inputs like cap, capF, and changepoints.

All of these inputs are specified when you call hts, and after that you just let it run.

The following is the description of inputs and outputs for hts as well as the specified defaults:

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
     
     method – (String)  the type of hierarchical forecasting method that the user wants to use. 
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
     

Don’t forget to specify the frequency if you’re not using Daily data.

All other functions should be self-explanatory.

# Part III: Room For Improvement

The package could benefit from the following two things:
1. A way to run some of it in parallel, cause it take a while sometimes.
2. Prediction intervals would be cool as well.
