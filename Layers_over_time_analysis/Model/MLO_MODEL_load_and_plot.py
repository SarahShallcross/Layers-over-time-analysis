#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:16:46 2017

@author: gy11s2s
"""
import iris
import iris.coords as icoords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

level_number_coord = icoords.DimCoord(range(1,86), standard_name='model_level_number')
model_height = np.loadtxt('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter=',')
#hybrid_ht = np.array([20, 53.33334, 100, 160, 233.3333, 320, 420, 533.3334, 659.9999, 799.9999, 953.3337, 1120, 1300, 1493.333, 1700, 1920, 2153.333, 2400, 2659.999, 2933.333, 3220, 3520, 3833.333, 4160, 4500, 4853.333, 5220, 5600, 5993.333, 6400, 6820, 7253.333, 7700, 8160.001, 8633.34, 9120.007, 9620.02, 10133.37, 10660.08, 11200.16, 11753.64, 12320.55, 12900.93, 13494.88, 14102.48, 14723.88, 15359.24, 16008.82, 16672.9, 17351.9, 18046.29, 18756.7, 19483.89, 20228.78, 20992.53, 21776.51, 22582.39, 23412.16, 24268.18, 25153.22, 26070.59, 27024.11, 28018.26, 29058.23, 30150.02, 31300.54, 32517.71, 33810.59, 35189.52, 36666.24, 38254.03, 39967.93, 41824.85, 43843.83, 46046.21, 48455.83, 51099.35, 54006.43,57210.02,60746.7,64656.96,68985.52,73781.77,79100.02,85000])

MLO_data = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/model_extinction_sites/Extinction_550nm_site_10.nc')

x = np.array(range(0,3600)) #TIME FOR WHOLE RANGE OF MODEL DATA

# Setting time to 3600(240(10 days x 3 x 8 points/day) x 15 (number of months Dec1990-Feb1992))/120 = 30
# 1-30 = Dec1990, 31-60 = Jan1991 etc...
time = (x/240) # Each month has an index e.g. 0 = Dec1990
months = ['Dec 1990', 'Jan 1991', 'Feb 1991','Mar 1991', 'Apr 1991', 'May 1991', 'Jun 1991', 'Jul 1991', 'Aug 1991', 'Sep 1991', 'Oct 1991', 'Nov 1991', 'Dec 1991', 'Jan 1992', 'Feb 1992','Mar 1992']

x = np.array(range(0,(240*15)))

# NB If setting sliced time - x range has to match the sliced time within the array e.g. plt.contourf(range(1000,3000),y1,data[0:72,1000:3000])

y = list(level_number_coord.points) # height for model heights
y1 = model_height/1000 # Height for REAL heights relative to 85 model levels
# y1[0:72] = ~ 40 km (in order to compare with obs data)

# For data from raw file - requires slicing 
#plt.figure()

data = np.transpose(MLO_data.data)

# NEELY HELP
# define the colormap
cmap = plt.cm.get_cmap('OrRd', 11)
# extract all colors from the map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (1.0,1.0,1.0,1.0)
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
#Set Color For Values less than plot Bounds
cmap.set_under('w')

MLO_data_masked_array=np.ma.masked_where(data<=0, data)
MLO_data_masked_array_km=MLO_data_masked_array*1000
MLO_data_masked_array_Log=np.log10(MLO_data_masked_array_km)
#MLO_data_masked_array=np.ma.masked_where(data<=0, MLO_data_masked_array_Log)
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
MLO=ax.contourf(x[1680:3600],y1[0:72],MLO_data_masked_array_km[0:72,1680:3600],8,cmap=cmap)# vmin=0, vmax=2)
cbar = plt.colorbar(MLO)
plt.ylim(17,30)
plt.grid()
plt.title('Extinction (km$^{-1}$) at Mauna Loa (550 nm) July 1991 - February 1992')
plt.xlabel('Time')
plt.ylabel('Altitude (km)')
plt.xticks((x[7:15]*240), months[7:15]) # 
fig.savefig('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/July1991-Feb1992.png', bbox_inches='tight')

#plt.contourf(range(1680,3600),y1[0:72],data[0:72,1680:3600])
# 

# Specific time periods
Dec1990 = MLO_data.data[0:72,0:240] # Specific for model height and time, Python reads rows first (0:72) and then columns (time: 0:240)