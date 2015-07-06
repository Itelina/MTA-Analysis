# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:55:50 2015

@author: ItelinaMa
"""

'''
This document is divided in 3 sections. 
Section 1 includes solutions to challenges 1-3.
Section 2 includes self-defined functions which I have created to complete and project
Section 3 contains solutions to challenge 4 - 10, using the self-defined functions from section 2. Note that I am returning the solutions for 5, 6, 8
, 9 in pandas Dataframe format.'''

''' Section 1'''
import os
os.getcwd()
os.chdir('../Benson')
import csv
from collections import defaultdict
import dateutil.parser
from datetime import datetime
from pandas import DataFrame, Series
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#Challenge 1
def Challenge_1():
    f = open('turnstile_150404.txt')
    csv_f = csv.reader(f)
    keys = []
    values = []
    
    for row in csv_f:
        keys.append(row[0:4])
        values.append(row[4:10])
    del keys[0]  
    del values[0]

    subwaydict = {}
    dummylist = []
    for i, key in enumerate(keys):
        if tuple(key) in dummylist:
            subwaydict[tuple(key)].append(values[i])
        else:
            subwaydict[tuple(key)] = [values[i]]
            dummylist.append(tuple(key))
    
    return subwaydict

# Challenge 2
def Challenge_2():
    f = open('turnstile_150404.txt')
    csv_f = csv.reader(f)
    keys = []
    values = []

    for row in csv_f:
        keys.append(row[0:4])
        values.append([row[6], row[7], row[9]])
    del keys[0]  
    del values[0]
    
    values2 = []
    for i in range(len(values)):
        values2.append([values[i][0] + ' ' + values[i][1]])
        values2[i].extend([values[i][2]])
        values2[i][0] = dateutil.parser.parse(values2[i][0])
    
    subwaydict = {}
    dummylist = []
    for i, key in enumerate(keys):
        if tuple(key) in dummylist:
            subwaydict[tuple(key)].append(values2[i])
        else:
            subwaydict[tuple(key)] = [values2[i]]
            dummylist.append(tuple(key))
    
    return subwaydict

#Challenge 3
def Challenge_3(data=None):
    if not data:
        data = Challenge_2()
    for turnstiles in data.keys():
        i = 0
        while i < len(data[turnstiles]):
            timegaps = [datetime.time(0, 0), datetime.time(1, 0), datetime.time(2, 0), datetime.time(3, 0)]
            if data[turnstiles][i][0].time() not in timegaps:
                del data[turnstiles][i]
            else:
                i += 1
        dates = {}
        for i, values in enumerate(data[turnstiles]):
            if i == len(data[turnstiles]) -1:
                break
            dates[values[0]] = int(data[turnstiles][i+1][1]) - int(values[1])
        data[turnstiles] = dates
    return data
            
#Challenge 4
def Challenge_4(data=None):
    if not data:
        data = Challenge_3()
    turnID = data.keys()[0]
    turnTimeS = data[turnID]
    dates = turnTimeS.keys()
    counts = turnTimeS.values()
    dates, counts = zip(*sorted(zip(dates, counts)))
    plt.figure(figsize=(10,3))
    plt.plot(dates,counts)

'''Section 2'''
#Self-Defined Functions for Project Analysis
def ReadData(file='turnstile_150404.txt'):
    with open(file) as f:
        reader = csv.reader(f)
        rows = [[cell.strip() for cell in row] for row in reader]
    
    rows[0].extend(['DateTime', 'Weekday','Week','TurnstileID', 'SubwayID', 'TRAFFIC'])
    
    i = 1
    while i < len(rows):
        # Makes Additional Data Points        
        date = datetime.strptime(rows[i][6] + rows[i][7], '%m/%d/%Y%X')
        
        rows[i].append(date)
        rows[i].append(rows[i][-1].isoweekday())
        rows[i].append(rows[i][-2].isocalendar()[1]) #this accesses the week element in the isocalendar tuple
        
        turnstileID = rows[i][0] + rows[i][1] + rows[i][2] + rows[i][3]
        subwayID = rows[i][0] + rows[i][1] + rows[i][3]
        
        rows[i].append(turnstileID)
        rows[i].append(subwayID)

        # Reshapes the cumulative data
        if rows[i-1][-2] == rows[i][-2]:
            
            # Calculates Entries Traffic
            rows[i-1][9] = int(rows[i][9]) - int(rows[i-1][9])
            
            # Calculates Exit Traffic
            rows[i-1][10] = int(rows[i][10]) - int(rows[i-1][10])
            
            # Calculate Sum of Traffic
            rows[i-1].append(rows[i-1][9] + rows[i-1][10])
        elif i>1:
            rows[i-1][9]  = 0
            rows[i-1][10] = 0
            rows[i-1].append(0)

        i += 1
    
    rows.pop(-1)
    return rows

def CombineData(dlist=['turnstile_150404.txt', 'turnstile_150411.txt', 'turnstile_150418.txt', 'turnstile_150425.txt']):
    output = []
    for i, item in enumerate(dlist):
        if i>0:
            readin = ReadData(item)
            readin.pop(0)
            output.extend(readin)
        else:
            readin = ReadData(item)
            output.extend(readin)
    return output
    
def labelTime(num):
    if num.hour < 11:
        return 'Mornings'
    elif num.hour >= 11 and num.hour < 14:
        return 'Lunch'
    elif num.hour >= 14 and num.hour < 19:
        return 'Afternoons'
    else:
        return 'Evenings/Nights'

def labelWeekday(num):
    weekdays = {1:'MON', 2:'TUES', 3:'WED', 4:'THUR', 5:'FRI', 6:'SAT', 7:'SUN'}
    return weekdays[num]

def CleanData():
    data = CombineData()
    col = data[0]
    data = pd.DataFrame(data, columns = col)
    data = data.ix[1:, :]
    #Get rid of data that has negative entry values
    data = data[data.ENTRIES > 0]
    data = data[data.EXITS > 0]
    #Get rid of data with entries that are really high
    data = data[data.ENTRIES < 400000]
    
    data.index = data['DateTime']
    data['StationLine'] = data['STATION'] + ' ' + data['LINENAME']
    data['WeekdayLabel'] = data['Weekday'].apply(labelWeekday)
    data['TimePeriod'] = data['DateTime'].apply(labelTime)
    return data

MTAdata = CleanData()

def aggregateF(group1='DATE', group2=None, output='ENTRIES', data=None):
    try:
        clean_data = not data
    except ValueError:
        clean_data = data.empty
        
    if clean_data:
        data = MTAdata
    if not group2:
        values= data.groupby([group1])[output].agg('sum').order(ascending=False)
    else:
        values= data.groupby([group1,group2])[output].agg('sum')
        values = values.reset_index()
    #values = values.groupby(level=0).apply(lambda x: x.order(ascending=False))
    return values
    
def stationPlot(station='34 ST-PENN STA', x='TIME', y='ENTRIES'):
    data = aggregateF('STATION', x, y) 
    data = data[data['STATION'] == station]
    data.index = data[x]
    print(station)
    data.plot()
 
def plotByTime(station='34 ST-PENN STA'):
    print station
    data = MTAdata
    d1 = data.groupby(['STATION', 'TimePeriod', 'Week'])['TRAFFIC'].agg('sum')
    d1 = d1.reset_index()
    
    output = d1[d1['STATION'] == station]
    output = output.pivot('TimePeriod', 'Week', 'TRAFFIC')
    plt.figure(); output.plot(); plt.legend(loc='best')
 
def trafficSummary(timeperiod = 'Afternoons', perct = 5, values='TRAFFIC'):
    data = MTAdata
    d2 = data.groupby(['TimePeriod','StationLine'])[values].agg('sum').reset_index()
    d2 = d2[d2.TimePeriod == timeperiod].sort(values, ascending = False)
    selection = d2.shape[0]/100*perct
    stations = d2.StationLine[0:selection]
    d2 = d2[d2['StationLine'].isin(stations)]
    d2['ByWeek'] = d2[values]/4
    return d2
    
def trafficSummaryPlot(timeperiod = 'Afternoons', perct = 5, values='TRAFFIC'):
    data = MTAdata
    d1 = data.groupby(['TimePeriod','Weekday','StationLine'])[values].agg('sum').reset_index()
    
     #Calculates the top stations for the time period
    d2 = data.groupby(['TimePeriod','StationLine'])[values].agg('sum').reset_index()
    d2 = d2[d2.TimePeriod == timeperiod].sort(values, ascending = False)
    selection = d2.shape[0]/100*perct
    stations = d2.StationLine[0:selection]
    
    #Applies output stations into the graphing
    output = d1[d1['StationLine'].isin(stations)]
    output = output[output['TimePeriod'] == timeperiod]
    output = output.pivot('Weekday', 'StationLine', values)
    weekday_map= {1:'MON', 2:'TUE', 3:'WED', 4:'THU',
              5:'FRI', 6:'SAT', 7:'SUN'}
    ax = output.plot()
    ax.set_xticklabels([weekday_map[item] for item in ax.get_xticks()])
    ax.legend(loc='upper left', fontsize = 6)
    plt.title(values+ ' Summary for ' + timeperiod)
    #print(stations)

def stationWeeksPlot(stations = ['PATH WTC'], values = 'EXITS', timeperiod = 'Afternoons'):
    data = MTAdata
    d1 = data.groupby(['TimePeriod','Week','Weekday','STATION'])[values].agg('sum').reset_index()
    output = d1[d1['STATION'].isin(stations)]
    output = output[output['TimePeriod'] == timeperiod]
    output = output.pivot('Weekday', 'Week', values)
    weekday_map= {1:'MON', 2:'TUE', 3:'WED', 4:'THU',
          5:'FRI', 6:'SAT', 7:'SUN'}
    ax = output.plot()
    ax.set_xticklabels([weekday_map[item] for item in ax.get_xticks()])
    ax.legend(loc='upper left', fontsize = 6)
    plt.title(timeperiod + ': ' + values+ ' Summary by Weeks')
    print (stations)

'''Section 3: Challenge 5 - 10'''
#Challenge 5
def Challenge_5():
    a = aggregateF(group1='DATE', group2='SubwayID', output='ENTRIES')
    return a

#Challenge 6
def Challenge_6():
    a = aggregateF(group1='DATE', group2='STATION', output='ENTRIES')
    return a

#Challenge 7
def Challenge_7():
    a = stationPlot()
    return a
    
#Challenge 8
def Challenge_8(station='34 ST-PENN STA'):
    print station
    data = MTAdata
    d1 = data.groupby(['STATION', 'Weekday', 'Week'])['ENTRIES'].agg('sum')
    d1 = d1.reset_index()
    
    output = d1[d1['STATION'] == station]
    output = output.pivot('Weekday', 'Week', 'ENTRIES')
    plt.figure(); output.plot(); plt.legend(loc='best')

#Challenge 9   
def Challenge_9():
    data = MTAdata
    d2 = data.groupby(['Week','STATION'])['ENTRIES'].agg('sum').reset_index()
    #Choosing week 14 as an example to analyze highest volume stations for week 14
    d2 = d2[d2.Week == 14].sort('ENTRIES', ascending = False)
    return d2
    
#Challenge 10
def Challenge_10():
    data = MTAdata
    data = data[data.TRAFFIC > 0]
    doutput = aggregateF('STATION', output='TRAFFIC', data=data)
    doutput.hist(bins=20)




    
    
    
    
       
        
            
    
        
    
    




            
            
                
            
    
        
        
        
    


