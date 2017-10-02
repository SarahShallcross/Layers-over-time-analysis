#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:52:02 2017

@author: gy11s2s
This code will attempt to plot for all months, the point where backscatter > average to moving back to below
average, i.e. picking out the area where the volcanic material is.
"""

# Import necessary packages
import numpy as np
import os.path
import matplotlib.pyplot as plt

# Set necessary arrays/variables
monthdays_nly=[1,32,60,91,121,152,182,213,244,274,305,335]	#Julian day month days for non-leap years
monthdays = monthdays_nly

imonth=6	#Set to 7 for July for now
monthdays=monthdays_nly
daycount = 0
JulianDayList = []
lower_layer_values = []
upper_layer_values = []



# Load in data

i = 0                   
for day in range(1,32): #Day range for an entire month
        filename = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/Jan1991-Nov1992/*.txt' #+ str(day) + '.07.1991.txt' # Specifies July
        if os.path.isfile(filename):
            
            data = np.loadtxt(fname=filename, delimiter=',', skiprows=2, usecols=(0,1))
            col1 = data[:,0]
            col2 = data[:,1]
            altitude = col1/1000	# Converts altitude to km
            backscatter=col2*0.001	# Conversion factor for backscatter in data
            
                    
# Function to find top and bottom of the enhanced area - i.e. find average

            Average = np.average(backscatter)
            
            lower_layer_indices = np.where(backscatter > Average)	# Set variable to find INDICES where BSR > Average
            lower_layer_index = lower_layer_indices[0][0]
            upper_layer_indices = np.where(backscatter < Average)
            upper_layer_index = min([x for x in upper_layer_indices[0] if x > lower_layer_index])
            
# Append lists
            lower_layer_values.append(altitude[lower_layer_index])
            upper_layer_values.append(altitude[upper_layer_index])
            
            JulianDay = monthdays[imonth]+day-1
            JulianDayList.append(int(JulianDay))
            

# Plot to test

plt.figure()

plot = plt.scatter(JulianDayList, lower_layer_values, color='r',label='Bottom of layer')
plot = plt.scatter(JulianDayList, upper_layer_values, color='b',label='Top of layer')
plt.xlabel('Time (Julian Day)')
plt.ylabel('Height (km)')