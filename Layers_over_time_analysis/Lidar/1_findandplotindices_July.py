#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:59:37 2017

@author: gy11s2s
This code works for plotting the point where backscatter > average to moving back to below
average, i.e. picking out the area where the volcanic material is.  This is for July 1991.
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
lower_layer1_values = []
upper_layer1_values = []
lower_layer2_values = []
upper_layer2_values = []


# Load in data

i = 0                   
for day in range(1,32): #Day range for an entire month
        filename = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/July/' + str(day) + '.07.1991.txt' # Specifies July
        if os.path.isfile(filename):
            
            data = np.loadtxt(fname=filename, delimiter=',', skiprows=2, usecols=(0,1))
            col1 = data[:,0]
            col2 = data[:,1]
            altitude = col1/1000	# Converts altitude to km
            backscatter=col2*0.001	# Conversion factor for backscatter in data
            
                    
# Function to find top and bottom of the enhanced area - i.e. find average

            Average = np.average(backscatter)
            
# Plots the first lower layers
            above_average_indices = np.where(backscatter > Average)	# Set variable to find INDICES where BSR > Average
            layer1_lower_index = above_average_indices[0][0]
            below_average_indices = np.where(backscatter < Average)
            layer1_upper_index = min([x for x in below_average_indices[0] if x > layer1_lower_index])
            #upper_layer_index = lower_layer_indices[0][-1]
            
# Finds other layers

            #if len(([y for y in above_average_indices[0] if y > layer1_upper_index])) > 0: # If second layer exists, find minimum
            #    layer2_lower_index = min([y for y in above_average_indices[0] if y > layer1_upper_index])
            #else: 
            #    layer2_lower_index = 0
                
            
# Append lists
            lower_layer1_values.append(altitude[layer1_lower_index])
            upper_layer1_values.append(altitude[layer1_upper_index])
            
            #lower_layer2_values.append(altitude[layer2_lower_index])
            #upper_layer2_values.append(altitude[above_average_indices[0][-1]])
            
            JulianDay = monthdays[imonth]+day-1
            JulianDayList.append(int(JulianDay))
            

# Plot to test

plt.figure()

plot = plt.scatter(JulianDayList, lower_layer1_values, color='r',label='Bottom of 1st layer')
plot = plt.scatter(JulianDayList, upper_layer1_values, color='b',label='Top of 1st layer')

#if layer2_lower_index < 47 and above_average_indices[0][-1] > altitude[layer1_upper_index]:
#    plot = plt.scatter(JulianDayList, lower_layer2_values, color='b', label='Bottom of 2nd layer')
#    plot = plt.scatter(JulianDayList, upper_layer2_values, color='b',label='Top of 2nd layer')

plt.xlabel('Time (Julian Day)')
plt.ylabel('Height (km)')