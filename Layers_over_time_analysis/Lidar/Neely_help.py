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

imonth=5
j = 5

daycount = 0
JulianDayList = []
lower_layer1_values = []
upper_layer1_values = []
lower_layer2_values = []
upper_layer2_values = []


# Load in data

#for year in range(1991,1992):

for year in range(1991,1993):
    for mon in range(1,13):  
        for day in range(1,32): #Day range for an entire month
            filename = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/Jan1991-Nov1992/' + str(year) + '.' + format(mon,'02') + '.' + format(day,'02') + '.txt' # 
            
            print filename
            if os.path.isfile(filename):
                #print filename
            
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
                #layer1_upper_index = min([x for x in above_average_indices[0] if x > layer1_lower_index])
                layer1_upper_index = min([x for x in below_average_indices[0] if x > layer1_lower_index])
                 
# Finds other layers

                if len(([y for y in above_average_indices[0] if y > layer1_upper_index])) > 0: # If second layer exists, find minimum
                    layer2_lower_index = min([y for y in above_average_indices[0] if y > layer1_upper_index])
                else: 
                    pass
                #print layer2_lower_index
                    #layer2_lower_index = 0
                
            
# Append lists
                lower_layer1_values.append(altitude[layer1_lower_index])
                upper_layer1_values.append(altitude[layer1_upper_index])
            
                lower_layer2_values.append(altitude[layer2_lower_index])
                upper_layer2_values.append(altitude[above_average_indices[0][-1]])
               
                
                JulianDay = monthdays[mon-1]+day-1
                
                JulianDayList.append(int(JulianDay))
            
# Plot to test

plt.figure()

plot = plt.scatter(JulianDayList, lower_layer1_values, color='r',label='Bottom of 1st layer')
plot = plt.scatter(JulianDayList, upper_layer1_values, color='b',label='Top of 1st layer')
plt.ylim(15, 30)
plt.xlim(152,366)

#if layer2_lower_index < 47 and above_average_indices[0][-1] > altitude[layer1_upper_index]:
#    plot = plt.scatter(JulianDayList, lower_layer2_values, color='b', label='Bottom of 2nd layer')
#    plot = plt.scatter(JulianDayList, upper_layer2_values, color='b',label='Top of 2nd layer')

plt.xlabel('Time (Julian Day)')
plt.ylabel('Height (km)')