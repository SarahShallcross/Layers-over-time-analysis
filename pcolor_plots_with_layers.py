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
altitude_template_MLO = np.arange(15500,30000,300)
file_paths_MLO = []
file_names = sorted(os.listdir(MLO_path))
for name in file_names:
    filepath = os.path.join(MLO_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_MLO.append(filepath)
        
# Table Mountain
altitude_template_TabMtn = np.arange(15500,30000,300)
file_paths_TabMtn = []
file_names = sorted(os.listdir(TabMtn_path))
for name in file_names:
    filepath = os.path.join(TabMtn_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_TabMtn.append(filepath)
        
# Toronto
altitude_template_Toronto = np.arange(15500,30000,300)
file_paths_Toronto = []
file_names = sorted(os.listdir(Toronto_path))
for name in file_names:
    filepath = os.path.join(Toronto_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_Toronto.append(filepath)
        
# OHP
altitude_template_OHP = np.arange(15500,30000,300)
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
    gridded_BS = np.interp(altitude_template_MLO, altitude, backscatter)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d') # Converts filename date to Pandas datetime
    d_backscatter={Date : gridded_BS}

    if df_MLO.empty:
        df_MLO = pd.DataFrame(d_backscatter, index=altitude_template_MLO)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template_MLO)
        df_MLO = df_MLO.join(df_temp_BS)

MLO_data=df_MLO

#Table Mountain
df_TabMtn = pd.DataFrame()

for files in file_paths_TabMtn:
    data = np.loadtxt(fname=files, delimiter=',', skiprows=1, usecols=(0,1))
    col1 = data[:,0]
    col2 = data[:,1]
    altitude = col1
    backscatter=col2*0.001
    gridded_BS = np.interp(altitude_template_TabMtn, altitude, backscatter)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d') # Converts filename date to Pandas datetime
    d_backscatter={Date : gridded_BS}

    if df_TabMtn.empty:
        df_TabMtn = pd.DataFrame(d_backscatter, index=altitude_template_TabMtn)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template_TabMtn)
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
    gridded_BS = np.interp(altitude_template_Toronto, altitude, backscatter)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d') # Converts filename date to Pandas datetime
    d_backscatter={Date : gridded_BS}

    if df_Toronto.empty:
        df_Toronto = pd.DataFrame(d_backscatter, index=altitude_template_Toronto)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template_Toronto)
        df_Toronto = df_Toronto.join(df_temp_BS)

Toronto_data=df_Toronto

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
    gridded_BS = np.interp(altitude_template_OHP, altitude, backscatter)
    gridded_ext = np.interp(altitude_template_OHP, altitude, extinction)
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d') # Converts filename date to Pandas datetime
    d_backscatter={Date : gridded_BS}
    d_extinction = {Date : gridded_ext}

    if df_OHP.empty:
        df_OHP = pd.DataFrame(d_backscatter, index=altitude_template_OHP)
    else:
        df_temp_BS=pd.DataFrame(d_backscatter, index=altitude_template_OHP)
        df_OHP = df_OHP.join(df_temp_BS)
        
    if df_OHP_ext.empty:
        df_OHP_ext = pd.DataFrame(d_extinction, index=altitude_template_OHP)
    else:
        df_temp_ext=pd.DataFrame(d_extinction, index=altitude_template_OHP)
        df_OHP_ext = df_OHP_ext.join(df_temp_ext)

OHP_data=df_OHP
OHP_ext_data = df_OHP_ext





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

Dates_Toronto=Toronto_data.columns
Alts_Toronto=map(float,pd.Index.tolist(Toronto_data.index))
Alts_Toronto = np.array(Alts_Toronto)/1000

Toronto_data_masked_array=np.ma.masked_where(Toronto_data<0.8, Toronto_data)

Dates_OHP=OHP_data.columns
Alts_OHP=map(float,pd.Index.tolist(OHP_data.index)) 
Alts_OHP = np.array(Alts_OHP)/1000

OHP_data_masked_array=np.ma.masked_where(OHP_data<0.8, OHP_data)

################
### Plotting ###
################


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


# Set up figure
fig = plt.figure(figsize=(15, 5))

### MLO ###

ax1 = fig.add_subplot(411, ylabel = 'Altitude (km)', ylim = (16.5, 31))
MLO=ax1.pcolor(Dates_MLO,Alts_MLO,MLO_data_masked_array, norm=LogNorm(vmin=MLO_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax1.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
ax1.tick_params(axis='x',which='both', labelbottom='off')

# Adds measurement dates as blue diamonds and the Mount Pinatubo eruption as a red triangle.
plt.plot(Dates_MLO, np.ones((len(Dates_MLO),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)


test = np.genfromtxt('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Lidar/' + 'all_layer_heights.txt')
#test = np.genfromtxt('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Lidar/' + 'all_layer_heights.txt')

for i in range(test.shape[0]):
    plt.plot([Dates_MLO[i],Dates_MLO[i]], test[i,0:2],'ko')
    plt.plot([Dates_MLO[i],Dates_MLO[i]], test[i,2:4],'ro')
    #plt.plot([Dates_MLO[i],Dates_MLO[i]], test[i,4:6],'yo')
plt.tick_params(axis='x', which='both', bottom = False, top = False, labelbottom=False)

### Table Mountain ###

ax2 = fig.add_subplot(412, ylabel = 'Altitude (km)', ylim = (16.5, 31))
TabMtn=ax2.pcolor(Dates_TabMtn,Alts_TabMtn,TabMtn_data_masked_array, norm=LogNorm(vmin=TabMtn_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax2.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
ax2.tick_params(axis='x',which='both', labelbottom='off')
plt.plot(Dates_TabMtn, np.ones((len(Dates_TabMtn),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)


### Toronto ###

ax3 = fig.add_subplot(413, ylabel = 'Altitude (km)', ylim = (16.5, 31))
Toronto=ax3.pcolor(Dates_Toronto,Alts_Toronto,Toronto_data_masked_array, norm=LogNorm(vmin=Toronto_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax3.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
ax3.tick_params(axis='x',which='both', labelbottom='off')
plt.plot(Dates_Toronto, np.ones((len(Dates_Toronto),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)

### OHP ###

ax4 = fig.add_subplot(414, ylabel = 'Altitude (km)', ylim = (16.5, 31))
OHP=ax4.pcolor(Dates_OHP,Alts_OHP,OHP_data_masked_array, norm=LogNorm(vmin=OHP_data_masked_array.min(), vmax=MLO_data_masked_array.max()), cmap=cmap)
plt.grid()
ax4.set_xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
plt.plot(Dates_OHP, np.ones((len(Dates_OHP),))*30.000, 'bd', markersize=1.5)
plt.plot('1991-06-16', 16.800, 'r^', markersize=12)

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%b %Y')
ax4.xaxis.set_major_formatter(myFmt)
ax4.xaxis_date()

plt.tight_layout()
fig.subplots_adjust(left = 0.05, bottom=0.15, hspace=0.05)
cbar_ax = fig.add_axes([0.25,0.05,0.5,0.05])
cbar = fig.colorbar(MLO, extend='both', orientation = 'horizontal', cax=cbar_ax)



##################
### MODEL DATA ###
##################


##################################
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
y1 = model_height/1000 # Height for REAL heights relative to 85 model levels
# y1[0:72] = ~ 40 km (in order to compare with obs data)


MLO_550 = np.transpose(MLO_data_550.data)
TabMtn_550 = np.transpose(TabMtn_data_550.data)
Toronto_550 = np.transpose(Toronto_data_550.data)
OHP_550 = np.transpose(OHP_data_550.data)


# Specific time periods
Dec1990 = MLO_data_550.data[0:72,0:240] # Specific for model height and time, Python reads rows first (0:72) and then columns (time: 0:240)


MLO_data_masked_array=np.ma.masked_where(MLO_550<=0, MLO_550)
MLO_data_masked_array_km=MLO_data_masked_array*1000

TabMtn_data_masked_array=np.ma.masked_where(TabMtn_550<=0, TabMtn_550)
TabMtn_data_masked_array_km=TabMtn_data_masked_array*1000

Toronto_data_masked_array=np.ma.masked_where(Toronto_550<=0, Toronto_550)
Toronto_data_masked_array_km=Toronto_data_masked_array*1000

OHP_model_masked_array=np.ma.masked_where(OHP_550<=0, OHP_550)
OHP_model_masked_array_km=OHP_model_masked_array*1000

###########################
### Plotting model data ###
###########################


### Set up daily plots ###

def daily_mean(x):
    return np.mean(x.reshape(-1,8),axis=1)

average_MLO_550 = np.apply_along_axis(daily_mean, axis=1, arr=MLO_data_masked_array_km)
average_TabMtn_550 = np.apply_along_axis(daily_mean, axis=1, arr=TabMtn_data_masked_array_km)
average_Toronto_550 = np.apply_along_axis(daily_mean, axis=1, arr=Toronto_data_masked_array_km)
average_OHP_550 = np.apply_along_axis(daily_mean, axis=1, arr=OHP_model_masked_array_km)

x = np.array(range(0,(30*15))) # x = np.array(range(0,3600)) #TIME FOR WHOLE RANGE OF MODEL DATA


###################
### Daily plots ###
###################


fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(411, ylabel = 'Altitude (km)', ylim = (10.5, 35), xticks = (x[6:15]*30))
data_plot1=ax1.pcolor(x[180:450],y1[0:72],average_MLO_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8,vmax=0.000302860455661731),cmap=cmap)
ax1.tick_params(labelbottom=False)
plt.grid()

test_model = np.genfromtxt('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/' + 'all_layer_heights_model.txt')
#test_model = np.genfromtxt('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/' + 'all_layer_heights_model.txt')

for i in range(0,270):
    plt.plot([x[180:450][i],x[180:450][i]], test_model[i][0:2],'ko')
    plt.plot([x[180:450][i],x[180:450][i]], test_model[i][2:4],'ro')
    #plt.plot([Dates_MLO[i],Dates_MLO[i]], test[i,4:6],'yo')
plt.tick_params(axis='x', which='both', bottom = False, top = False, labelbottom=False)

ax2 = fig.add_subplot(412, ylabel = 'Altitude (km)', ylim = (16.5, 31), xticks = (x[6:15]*30))
data_plot2=ax2.pcolor(x[180:450],y1[0:72],average_TabMtn_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8, vmax=0.000302860455661731),cmap=cmap)
ax2.tick_params(labelbottom=False)
plt.grid()

ax3 = fig.add_subplot(413, ylabel = 'Altitude (km)', ylim = (16.5, 31), xticks = (x[6:15]*30))
data_plot3=ax3.pcolor(x[180:450],y1[0:72],average_Toronto_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8,vmax=0.000302860455661731),cmap=cmap)
ax3.tick_params(labelbottom=False)
plt.grid()

ax4 = fig.add_subplot(414, ylabel = 'Altitude (km)', xlabel = 'Time', ylim = (16.5, 31))
data_plot4=ax4.pcolor(x[180:450],y1[0:72],average_OHP_550[0:72,180:450],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)#vmin=10**-8, vmax=0.000302860455661731),cmap=cmap)
plt.xticks((x[6:15]*30), months[6:15])
plt.grid()

plt.tight_layout()
fig.subplots_adjust(left = 0.05, bottom=0.15, hspace=0.05)
cbar_ax = fig.add_axes([0.25,0.05,0.5,0.05])
cbar = fig.colorbar(data_plot1, extend='both', orientation = 'horizontal', cax=cbar_ax)



##############################################
### TOTAL TIME PLOTS - 3 HOURLY RESOLUTION ### 
##############################################

x = np.array(range(0,(8*30*15)))

fig = plt.figure(figsize=(15, 5))

### MLO ###

ax1 = fig.add_subplot(411, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))
MLO_plot = ax1.pcolor(x[1440:3600],y1[0:72],MLO_data_masked_array_km[0:72,1440:3600], norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])
ax1.tick_params(labelbottom=False)

### Table Mountain ###

ax2 = fig.add_subplot(412, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))
TabMtn_plot=ax2.pcolor(x[1440:3600],y1[0:72],TabMtn_data_masked_array_km[0:72,1440:3600],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5),cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])
ax2.tick_params(labelbottom=False)

### Toronto ###

ax3 = fig.add_subplot(413, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))
Toronto_plot=ax3.pcolor(x[1440:3600],y1[0:72],Toronto_data_masked_array_km[0:72,1440:3600],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])
ax3.tick_params(labelbottom=False)

### OHP ###

ax4 = fig.add_subplot(414, xlabel = 'Time', ylabel = 'Altitude (km)', ylim = (16.5,31))
OHP_plot=ax4.pcolor(x[1440:3600],y1[0:72],OHP_model_masked_array_km[0:72,1440:3600],norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)
plt.grid()
plt.xticks((x[6:15]*240), months[6:15])

plt.tight_layout()
fig.subplots_adjust(left = 0.05, bottom=0.15, hspace=0.05)
cbar_ax = fig.add_axes([0.25,0.05,0.5,0.05])
cbar = fig.colorbar(MLO_plot, extend='both', orientation = 'horizontal', cax=cbar_ax)






