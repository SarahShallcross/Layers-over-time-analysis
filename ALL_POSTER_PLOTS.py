#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:49:20 2018

@author: gy11s2s
"""
### LIDAR DATA ### 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as datetime
import fnmatch


MLO_path = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'
#MLO_path='C:/Users/s_sha/Documents/PhD/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'
TabMtn_path = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/Table_mountain/Plotting_months/June1991-Feb1992/'
#TabMtn_path = 'C:/Users/s_sha/Documents/PhD/Python/NDACC_files/Table_mountain/Plotting_months/June1991-Feb1992/'
Toronto_path = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/Toronto/Plotting_months/June1991-Feb1992/'
#Toronto_path = 'C:/Users/s_sha/Documents/PhD/Python/NDACC_files/Toronto/Plotting_months/June1991-Feb1992/'
OHP_path = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/OHP/June1991-Feb1992/'
#OHP_path = 'C:/Users/s_sha/Documents/PhD/Python/NDACC_files/OHP/June1991-Feb1992/'

# MLO
altitude_template = np.arange(15500,36000,300)
file_paths_MLO = []
file_names = sorted(os.listdir(MLO_path))
for name in file_names:
    filepath = os.path.join(MLO_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_MLO.append(filepath)
        
# Table Mountain
file_paths_TabMtn = []
file_names = sorted(os.listdir(TabMtn_path))
for name in file_names:
    filepath = os.path.join(TabMtn_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_TabMtn.append(filepath)
        
# Toronto
file_paths_Toronto = []
file_names = sorted(os.listdir(Toronto_path))
for name in file_names:
    filepath = os.path.join(Toronto_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_Toronto.append(filepath)
        
# OHP
file_paths_OHP = []
file_names = sorted(os.listdir(OHP_path))
for name in file_names:
    filepath = os.path.join(OHP_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_OHP.append(filepath)
            
# Check number of files in directory

#print len(fnmatch.filter(os.listdir('C:/Users/s_sha/Documents/PhD/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'), '*.txt'))
#print len(fnmatch.filter(os.listdir('C:/Users/s_sha/Documents/PhD/Python/NDACC_files/Table_mountain/Plotting_months/June1991-Feb1992/'), '*.txt'))
#print len(fnmatch.filter(os.listdir('C:/Users/s_sha/Documents/PhD/Python/NDACC_files/Toronto/Plotting_months/June1991-Feb1992/'), '*.txt'))
#print len(fnmatch.filter(os.listdir('C:/Users/s_sha/Documents/PhD/Python/NDACC_files/OHP/June1991-Feb1992/'), '*.txt'))
print len(fnmatch.filter(os.listdir('/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'), '*.txt'))
print len(fnmatch.filter(os.listdir('/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/Table_mountain/Plotting_months/June1991-Feb1992/'), '*.txt'))
print len(fnmatch.filter(os.listdir('/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/Toronto/Plotting_months/June1991-Feb1992/'), '*.txt'))
print len(fnmatch.filter(os.listdir('/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/OHP/June1991-Feb1992/'), '*.txt'))


# MLO
df_MLO = pd.DataFrame()

for files in file_paths_MLO:
    data = np.loadtxt(fname=files, delimiter=',', skiprows=1, usecols=(0,1))
    col1 = data[:,0]
    col2 = data[:,1]
    altitude = col1
    backscatter=col2*0.001
    gridded_BS = np.interp(altitude_template, altitude, backscatter)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d') # Converts filename date to Pandas datetime
    d_backscatter={Date : gridded_BS}

    if df_MLO.empty:
        df_MLO = pd.DataFrame(d_backscatter, index=altitude_template)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template)
        df_MLO = df_MLO.join(df_temp_BS)

MLO_data=df_MLO

# Table Mountain
df_TabMtn = pd.DataFrame()

for files in file_paths_TabMtn:
    data = np.loadtxt(fname=files, delimiter=',', skiprows=1, usecols=(0,1))
    col1 = data[:,0]
    col2 = data[:,1]
    altitude = col1
    backscatter=col2*0.001
    gridded_BS = np.interp(altitude_template, altitude, backscatter)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d')
    d_backscatter={Date : gridded_BS}

    if df_TabMtn.empty:
        df_TabMtn = pd.DataFrame(d_backscatter, index=altitude_template)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template)
        df_TabMtn = df_TabMtn.join(df_temp_BS)

TabMtn_data=df_TabMtn

# Toronto
df_Toronto = pd.DataFrame()

for files in file_paths_Toronto:
    data = np.loadtxt(fname=files, delimiter=',', skiprows=1, usecols=(0,1))
    col1 = data[:,0]
    col2 = data[:,1]
    altitude = col1
    backscatter=col2
    gridded_BS = np.interp(altitude_template, altitude, backscatter)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d')
    d_backscatter={Date : gridded_BS}

    if df_Toronto.empty:
        df_Toronto = pd.DataFrame(d_backscatter, index=altitude_template)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template)
        df_Toronto = df_Toronto.join(df_temp_BS)

Toronto_data=df_Toronto

# OHP

df_OHP = pd.DataFrame()
df_OHP_ext = pd.DataFrame()

for files in file_paths_OHP:
    data = np.loadtxt(fname=files, delimiter=',', skiprows=1, usecols=(0,3,5))
    col1 = data[:,0]
    col2 = data[:,1]
    col3 = data[:,2]
    altitude = col1
    extinction= col2
    backscatter=col3
    gridded_BS = np.interp(altitude_template, altitude, backscatter)
    gridded_ext = np.interp(altitude_template, altitude, extinction)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d')
    d_backscatter={Date : gridded_BS}
    d_extinction = {Date : gridded_ext}

    if df_OHP.empty:
        df_OHP = pd.DataFrame(d_backscatter, index=altitude_template)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template)
        df_OHP = df_OHP.join(df_temp_BS)
        
    if df_OHP_ext.empty:
        df_OHP_ext = pd.DataFrame(d_extinction, index=altitude_template)
    else:
        df_temp_ext=pd.DataFrame(d_extinction, index=altitude_template)
        df_OHP_ext = df_OHP_ext.join(df_temp_ext)

OHP_data=df_OHP
OHP_ext_data = df_OHP_ext

###################################
### Plotting lidar observations ###
###################################

# define the colormap
cmap = plt.cm.get_cmap('rainbow', 16)
# extract all colors from the map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (1.0,1.0,1.0,1.0)
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
#Set Color For Values less than plot Bounds
cmap.set_under('w')


#Using column and Index Info
Dates_MLO=MLO_data.columns# This is in date time format so it is easy to deal with
Alts_MLO=map(float,pd.Index.tolist(MLO_data.index)) #Initially this is in a funny Index object so we need to change to a list of float values
Alts_MLO = np.array(Alts_MLO)/1000
# Masking out data below a Threshold
MLO_data_masked_array=np.ma.masked_where(MLO_data<0.8, MLO_data)

Dates_TabMtn=TabMtn_data.columns# This is in date time format so it is easy to deal with
Alts_TabMtn=map(float,pd.Index.tolist(TabMtn_data.index)) #Initially this is in a funny Index object so we need to change to a list of float values
Alts_TabMtn = np.array(Alts_TabMtn)/1000
TabMtn_data_masked_array=np.ma.masked_where(TabMtn_data<0.8, TabMtn_data)

Dates_Toronto=Toronto_data.columns# This is in date time format so it is easy to deal with
Alts_Toronto=map(float,pd.Index.tolist(Toronto_data.index)) #Initially this is in a funny Index object so we need to change to a list of float values
Alts_Toronto = np.array(Alts_Toronto)/1000
Toronto_data_masked_array=np.ma.masked_where(Toronto_data<0.8, Toronto_data)

Dates_OHP=OHP_data.columns# This is in date time format so it is easy to deal with
Alts_OHP=map(float,pd.Index.tolist(OHP_data.index)) #Initially this is in a funny Index object so we need to change to a list of float values
Alts_OHP = np.array(Alts_OHP)/1000
OHP_data_masked_array=np.ma.masked_where(OHP_data<0.8, OHP_data)



# Set up figure
fig = plt.figure(figsize=(15, 5))

### MLO ###

ax1 = fig.add_subplot(411, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Backscatter ratio at MLO (694.3 nm) June 1991 - February 1992')
MLO=ax1.pcolor(Dates_MLO,Alts_MLO,MLO_data_masked_array, norm=LogNorm(vmin=MLO_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax1.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)]) # Adds measurement dates as blue diamonds and the Mount Pinatubo eruption as a red triangle.
plt.plot(Dates_MLO, np.ones((len(Dates_MLO),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)
ax1.tick_params(labelbottom=False)

### Table Mountain ###

ax2 = fig.add_subplot(412, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Backscatter ratio at TBM (353 nm) June 1991 - February 1992')
TabMtn=ax2.pcolor(Dates_TabMtn,Alts_TabMtn,TabMtn_data_masked_array, norm=LogNorm(vmin=TabMtn_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax2.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
plt.plot(Dates_TabMtn, np.ones((len(Dates_TabMtn),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)
ax2.tick_params(labelbottom=False)

### Toronto ###

ax3 = fig.add_subplot(413, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Backscatter ratio at TOR (532 nm) June 1991 - February 1992')
Toronto=ax3.pcolor(Dates_Toronto,Alts_Toronto,Toronto_data_masked_array, norm=LogNorm(vmin=Toronto_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax3.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
plt.plot(Dates_Toronto, np.ones((len(Dates_Toronto),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)
ax3.tick_params(labelbottom=False)

### OHP ###

ax4 = fig.add_subplot(414, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Backscatter ratio at OHP (532 nm) June 1991 - February 1992')
OHP=ax4.pcolor(Dates_OHP,Alts_OHP,OHP_data_masked_array, norm=LogNorm(vmin=OHP_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax4.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
plt.plot(Dates_OHP, np.ones((len(Dates_OHP),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)

plt.tight_layout()
fig.subplots_adjust(left = 0.05, bottom=0.15, hspace=0.05)
cbar_ax = fig.add_axes([0.25,0.05,0.5,0.05])
cbar = fig.colorbar(MLO, extend='both', orientation = 'horizontal', cax=cbar_ax)

##################################
          ### MODEL ###
### 550 MODEL EXTINCTION PLOTS ###
##################################

import iris
import iris.coords as icoords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

level_number_coord = icoords.DimCoord(range(1,86), standard_name='model_level_number')
model_height = np.loadtxt('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter=',')
#model_height = np.loadtxt('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter=',')
#hybrid_ht = np.array([20, 53.33334, 100, 160, 233.3333, 320, 420, 533.3334, 659.9999, 799.9999, 953.3337, 1120, 1300, 1493.333, 1700, 1920, 2153.333, 2400, 2659.999, 2933.333, 3220, 3520, 3833.333, 4160, 4500, 4853.333, 5220, 5600, 5993.333, 6400, 6820, 7253.333, 7700, 8160.001, 8633.34, 9120.007, 9620.02, 10133.37, 10660.08, 11200.16, 11753.64, 12320.55, 12900.93, 13494.88, 14102.48, 14723.88, 15359.24, 16008.82, 16672.9, 17351.9, 18046.29, 18756.7, 19483.89, 20228.78, 20992.53, 21776.51, 22582.39, 23412.16, 24268.18, 25153.22, 26070.59, 27024.11, 28018.26, 29058.23, 30150.02, 31300.54, 32517.71, 33810.59, 35189.52, 36666.24, 38254.03, 39967.93, 41824.85, 43843.83, 46046.21, 48455.83, 51099.35, 54006.43,57210.02,60746.7,64656.96,68985.52,73781.77,79100.02,85000])

MLO_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_10.nc')
#MLO_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_10.nc')
TabMtn_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_8.nc')
#TabMtn_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_8.nc')
Toronto_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_5.nc')
#Toronto_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_5.nc')
OHP_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_4.nc')
#OHP_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_4.nc')
                              
x = np.array(range(0,(240*15))) # x = np.array(range(0,3600)) #TIME FOR WHOLE RANGE OF MODEL DATA
# Setting time to 3600(240(10 days x 3 x 8 points/day) x 15 (number of months Dec1990-Feb1992))/120 = 30
# 1-30 = Dec1990, 31-60 = Jan1991 etc...
time = (x/240) # Each month has an index e.g. 0 = Dec1990
months = ['Dec 1990', 'Jan 1991', 'Feb 1991','Mar 1991', 'Apr 1991', 'May 1991', 'Jun 1991', 'Jul 1991', 'Aug 1991', 'Sep 1991', 'Oct 1991', 'Nov 1991', 'Dec 1991', 'Jan 1992', 'Feb 1992','Mar 1992']

# NB If setting sliced time - x range has to match the sliced time within the array e.g. plt.contourf(range(1000,3000),y1,data[0:72,1000:3000])

y = list(level_number_coord.points) # height for model heights
y1 = model_height/1000


### Transposing and masking data and converting altitude to km for plotting ###

MLO_550 = np.transpose(MLO_data_550.data)
TabMtn_550 = np.transpose(TabMtn_data_550.data)
Toronto_550 = np.transpose(Toronto_data_550.data)
OHP_550 = np.transpose(OHP_data_550.data)

MLO_data_masked_array=np.ma.masked_where(MLO_550<=0, MLO_550)
MLO_data_masked_array_km=MLO_data_masked_array*1000

TabMtn_data_masked_array=np.ma.masked_where(TabMtn_550<=0, TabMtn_550)
TabMtn_data_masked_array_km=TabMtn_data_masked_array*1000

Toronto_data_masked_array=np.ma.masked_where(Toronto_550<=0, Toronto_550)
Toronto_data_masked_array_km=Toronto_data_masked_array*1000

OHP_model_masked_array=np.ma.masked_where(OHP_550<=0, OHP_550)
OHP_model_masked_array_km=OHP_model_masked_array*1000


# Specific time periods
Dec1990 = MLO_data_550.data[0:72,0:240] # Specific for model height and time, Python reads rows first (0:72) and then columns (time: 0:240)


###########################
### Plotting model data ###
###########################


###################
### Daily plots ###
###################

def daily_mean(x):
    return np.mean(x.reshape(-1,8),axis=1)

daily_MLO_550 = np.apply_along_axis(daily_mean, axis=1, arr=MLO_data_masked_array_km)
daily_TabMtn_550 = np.apply_along_axis(daily_mean, axis=1, arr=TabMtn_data_masked_array_km)
daily_Toronto_550 = np.apply_along_axis(daily_mean, axis=1, arr=Toronto_data_masked_array_km)
daily_OHP_550 = np.apply_along_axis(daily_mean, axis=1, arr=OHP_model_masked_array_km)


fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(411, ylabel = 'Altitude (km)', ylim = (16.5, 31))
MLO_plot=ax1.pcolor(x[180:450],y1[0:72],daily_MLO_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8,vmax=0.000302860455661731),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*30), months[6:15])
ax1.tick_params(labelbottom=False)
    
ax2 = fig.add_subplot(412, ylabel = 'Altitude (km)', ylim = (16.5, 31))
TAB_plot=ax2.pcolor(x[180:450],y1[0:72],daily_TabMtn_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8, vmax=0.000302860455661731),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*30), months[6:15])
ax2.tick_params(labelbottom=False)    
    
ax3 = fig.add_subplot(413, ylabel = 'Altitude (km)', ylim = (16.5, 31))
TOR_plot=ax3.pcolor(x[180:450],y1[0:72],daily_Toronto_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8,vmax=0.000302860455661731),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*30), months[6:15])
ax3.tick_params(labelbottom=False) 
    
ax4 = fig.add_subplot(414, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5, 31))
OHP_plot=ax4.pcolor(x[180:450],y1[0:72],daily_OHP_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8, vmax=0.000302860455661731),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*30), months[6:15])

plt.tight_layout()
fig.subplots_adjust(left = 0.05, bottom=0.15, hspace=0.05)
cbar_ax = fig.add_axes([0.25,0.05,0.5,0.05])
cbar = fig.colorbar(MLO_plot, extend='both', orientation = 'horizontal', cax=cbar_ax)


##############################################
### TOTAL TIME PLOTS - 3 HOURLY RESOLUTION ### 
##############################################

fig = plt.figure(figsize=(15, 5))

### MLO ###
ax1 = fig.add_subplot(411, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Extinction (km$^{-1}$) at MLO (550 nm) June 1991 - February 1992')
MLO_plot = ax1.pcolor(x[1440:3600],y1[0:72],MLO_data_masked_array_km[0:72,1440:3600], norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])
ax1.tick_params(labelbottom=False)


### Table Mountain ###

#MLO_data_masked_array=np.ma.masked_where(data<=0, MLO_data_masked_array_Log)
ax2 = fig.add_subplot(412, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Extinction (km$^{-1}$) at TBM (550 nm) June 1991 - February 1992')
TabMtn_plot=ax2.pcolor(x[1440:3600],y1[0:72],TabMtn_data_masked_array_km[0:72,1440:3600],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15]) # 
ax2.tick_params(labelbottom=False)

### Toronto ###

#MLO_data_masked_array=np.ma.masked_where(data<=0, MLO_data_masked_array_Log)
ax3 = fig.add_subplot(413, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Extinction (km$^{-1}$) at TOR (550 nm) June 1991 - February 1992')
Toronto_plot=ax3.pcolor(x[1440:3600],y1[0:72],Toronto_data_masked_array_km[0:72,1440:3600],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])
ax3.tick_params(labelbottom=False)

### OHP ###

ax4 = fig.add_subplot(414, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))#, title = 'Extinction (km$^{-1}$) at OHP (550 nm) June 1991 - February 1992')
OHP_plot=ax4.pcolor(x[1440:3600],y1[0:72],OHP_model_masked_array_km[0:72,1440:3600],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])

plt.tight_layout()
fig.subplots_adjust(left = 0.05, bottom=0.15, hspace=0.05)
cbar_ax = fig.add_axes([0.25,0.05,0.5,0.05])
cbar = fig.colorbar(MLO_plot, extend='both', orientation = 'horizontal', cax=cbar_ax)



