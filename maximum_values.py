# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:52:18 2018

@author: s_sha
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

#def organise_data(path,altitude_template):
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

from copy import deepcopy
MLO_data_copy=deepcopy(df_MLO)
TabMtn_data_copy=deepcopy(df_TabMtn)
Toronto_data_copy = deepcopy(df_Toronto)
OHP_data_copy = deepcopy(df_OHP)

MLO_data_masked=np.ma.masked_where(MLO_data<0.8, MLO_data)
TabMtn_data_masked=np.ma.masked_where(TabMtn_data<0.8, TabMtn_data)
Toronto_data_masked=np.ma.masked_where(Toronto_data<0.8, Toronto_data)
OHP_data_masked=np.ma.masked_where(OHP_data<0.8, OHP_data)


### Set up dates and altitudes ###
def dates_alts(data_arrays):

    if not type(data_arrays) == list:
        data_arrays = [data_arrays]
    
    dates = []
    alts  = []
    
    for data in data_arrays:
        dates.append(np.array(data.columns))
        alts.append((np.array(map(float,pd.Index.tolist(data.index))))/1000)
    
    return dates, alts

lidar_data = [MLO_data,TabMtn_data,Toronto_data,OHP_data]
lidar_masked = [MLO_data_masked,TabMtn_data_masked,Toronto_data_masked,OHP_data_masked]
lidar_dates, lidar_alts =  dates_alts(lidar_data)



def peak_altitude(data_arrays,alts_arrays):
    #Must provide same number of data as altitudes
    if not type(data_arrays) == list:
        data_arrays = [data_arrays]
        alts_arrays = [alts_arrays]
        
    maxindex = []
    peaks = []
    for i in range(len(data_arrays)):
        maxindex.append(np.nanargmax(data_arrays[i], axis=0))
        peaks.append(alts_arrays[i][maxindex[i]])
    
    return maxindex, peaks

lidar_maxindex, lidar_peaks = peak_altitude(lidar_data, lidar_alts)


##### ATTEMPTING TO  FIND REGRESSION OF FIRST LAYER #####
a = []
ordinal_dates = []
for i in range(len(lidar_dates[0])):
    a.append(datetime.datetime.utcfromtimestamp(lidar_dates[0][i].tolist()/1e9))
for j in range(59):
    ordinal_dates.append(a[j].toordinal())

x = ordinal_dates[4:14]
y = lidar_peaks[0][4:14]

import scipy
total_slope, total_intercept, total_r_value, total_p_value, total_std_error = scipy.stats.linregress(x,y)
first_slope, first_intercept, first_r_value, first_p_value, first_std_error = scipy.stats.linregress(x[0:6],y[0:6])
second_slope, second_intercept, second_r_value, second_p_value, second_std_error = scipy.stats.linregress(x[6:],y[6:])
print "Total r-squared:", total_r_value**2
print "First r-squared:", first_r_value**2
print "Second r-squared:", second_r_value**2

#########################################################


titles = ['MLO','TAB','TOR','OHP'] #plot titles

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

## PLOT ALL ON TOP OF EACH OTHER ##
plt.figure()
for i in range(4):
    plt.subplot(4,1,i+1)
    cbar = plt.pcolor(lidar_dates[i],lidar_alts[i],lidar_masked[i], norm=LogNorm(vmin=lidar_masked[i].min(), vmax=lidar_masked[0].max()), cmap=cmap)
    plt.scatter(lidar_dates[i],lidar_peaks[i], color = 'k', s=2)
    plt.title(titles[i])
    plt.xlim([datetime.date(1991, 6, 1), datetime.date(1992, 2, 28)])
#plt.colorbar(cbar)   
plt.ylim(16.5,31) 
plt.tight_layout()


### SET UP FIGURE FOR MLO ABOVE MID-LAT SITES
colors = ['navy', 'firebrick', 'orange', 'forestgreen']
fig, (ax0, ax1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 3]}, sharex=True)
fig.subplots_adjust(hspace=0.025, wspace=0.05)

## PLOT MLO ##
for i in range(1):
    ax0.plot(lidar_dates[0],lidar_peaks[0], color = colors[0], label = titles[i],zorder=1)
    ind1 = np.arange(0,np.shape(lidar_masked[0])[1])
    ind2 = np.nanargmax(lidar_masked[0], axis=0)
    ind = [ind2,ind1]
    ax0.scatter(lidar_dates[0],lidar_peaks[0],c=lidar_masked[0][ind],s=60,norm=LogNorm(vmin=lidar_masked[0].min(), vmax=lidar_masked[0].max()), cmap=cmap,edgecolors='k',zorder=2)
    ax0.legend(loc=0)
    ax0.set_ylabel('Altitude (km)')
    ax0.set_ylim(15, 31)

## PLOT MID-LAT SITES ##
#plt.setp(ax1.get_xticklabels(), visible=False)
for i in range(1,4):
    ax1.plot(lidar_dates[i],lidar_peaks[i], color = colors[i], label = titles[i],zorder=1)
    ind1 = np.arange(0,np.shape(lidar_masked[i])[1])
    ind2 = np.nanargmax(lidar_masked[i], axis=0)
    ind = [ind2,ind1]
    cbar = ax1.scatter(lidar_dates[i],lidar_peaks[i],c=lidar_masked[i][ind],s=60,norm=LogNorm(vmin=lidar_masked[i].min(), vmax=lidar_masked[0].max()), cmap=cmap,edgecolors='k',zorder=2)
    ax1.legend(loc=0)
    ax1.set_ylabel('Altitude (km)')
    ax1.set_ylim(15, 31)
ax1.set_xlabel('Time')

plt.tight_layout()
fig.subplots_adjust(left = 0.05, right=0.85)#, hspace=0)
cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
cbar = fig.colorbar(cbar, extend='both', orientation = 'vertical', cax=cbar_ax)

    

### MODEL ###
import iris
import iris.coords as icoords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

level_number_coord = icoords.DimCoord(range(1,86), standard_name='model_level_number')
model_height = np.loadtxt('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter=',')
#model_height = np.loadtxt('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter=',')
hybrid_ht = np.array([20, 53.33334, 100, 160, 233.3333, 320, 420, 533.3334, 659.9999, 799.9999, 953.3337, 1120, 1300, 1493.333, 1700, 1920, 2153.333, 2400, 2659.999, 2933.333, 3220, 3520, 3833.333, 4160, 4500, 4853.333, 5220, 5600, 5993.333, 6400, 6820, 7253.333, 7700, 8160.001, 8633.34, 9120.007, 9620.02, 10133.37, 10660.08, 11200.16, 11753.64, 12320.55, 12900.93, 13494.88, 14102.48, 14723.88, 15359.24, 16008.82, 16672.9, 17351.9, 18046.29, 18756.7, 19483.89, 20228.78, 20992.53, 21776.51, 22582.39, 23412.16, 24268.18, 25153.22, 26070.59, 27024.11, 28018.26, 29058.23, 30150.02, 31300.54, 32517.71, 33810.59, 35189.52, 36666.24, 38254.03, 39967.93, 41824.85, 43843.83, 46046.21, 48455.83, 51099.35, 54006.43,57210.02,60746.7,64656.96,68985.52,73781.77,79100.02,85000])

MLO_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_10.nc')
#MLO_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_10.nc')
TabMtn_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_8.nc')
#TabMtn_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_8.nc')
Toronto_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_5.nc')
#Toronto_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_5.nc')
OHP_data_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_4.nc')
#OHP_data_550 = iris.load_cube('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_4.nc')
                              
x = np.array(range(0,(30*15))) # x = np.array(range(0,3600)) #TIME FOR WHOLE RANGE OF MODEL DATA
# Setting time to 3600(240(10 days x 3 x 8 points/day) x 15 (number of months Dec1990-Feb1992))/120 = 30
time = (x/30) 
months = ['Dec 1990', 'Jan 1991', 'Feb 1991','Mar 1991', 'Apr 1991', 'May 1991', 'Jun 1991', 'Jul 1991', 'Aug 1991', 'Sep 1991', 'Oct 1991', 'Nov 1991', 'Dec 1991', 'Jan 1992', 'Feb 1992','Mar 1992']

# NB If setting sliced time - x range has to match the sliced time within the array e.g. plt.contourf(range(1000,3000),y1,data[0:72,1000:3000])

y = list(level_number_coord.points) # height for model heights
y1 = model_height[44:65]/1000 # Height for REAL heights relative to 85 model levels - 44:73 starts at 14 km to account for tropopause


MLO_550 = np.transpose(MLO_data_550.data)
TabMtn_550 = np.transpose(TabMtn_data_550.data)
Toronto_550 = np.transpose(Toronto_data_550.data)
OHP_550 = np.transpose(OHP_data_550.data)

def daily_mean(x):
    return np.mean(x.reshape(-1,8),axis=1)

#MLO_model_masked=np.ma.masked_where(MLO_550<=0, MLO_550)
MLO_model_masked=np.ma.masked_where(MLO_550[44:65]<=0, MLO_550[44:65])
MLO_model_masked_km=MLO_model_masked*1000

#TabMtn_model_masked=np.ma.masked_where(TabMtn_550<=0, TabMtn_550)
TabMtn_model_masked=np.ma.masked_where(TabMtn_550[44:65]<=0, TabMtn_550[44:65])
TabMtn_model_masked_km=TabMtn_model_masked*1000

#Toronto_model_masked=np.ma.masked_where(Toronto_550<=0, Toronto_550)
Toronto_model_masked=np.ma.masked_where(Toronto_550[44:65]<=0, Toronto_550[44:65])
Toronto_model_masked_km=Toronto_model_masked*1000

#OHP_model_masked=np.ma.masked_where(OHP_550<=0, OHP_550)
OHP_model_masked=np.ma.masked_where(OHP_550[44:65]<=0, OHP_550[44:65])
OHP_model_masked_km=OHP_model_masked*1000

daily_MLO_550 = np.array(np.apply_along_axis(daily_mean, axis=1, arr=MLO_model_masked_km))
daily_TabMtn_550 = np.array(np.apply_along_axis(daily_mean, axis=1, arr=TabMtn_model_masked_km))
daily_Toronto_550 = np.array(np.apply_along_axis(daily_mean, axis=1, arr=Toronto_model_masked_km))
daily_OHP_550 = np.array(np.apply_along_axis(daily_mean, axis=1, arr=OHP_model_masked_km))

model_data = [daily_MLO_550,daily_TabMtn_550,daily_Toronto_550,daily_OHP_550]
model_alts = [y1,y1,y1,y1]
model_masked = [MLO_model_masked_km,TabMtn_model_masked_km,Toronto_model_masked_km,OHP_model_masked_km]
model_maxindex, model_peaks = peak_altitude(model_data, model_alts)



plt.figure()
for i in range(4):
    #print(i)
    plt.subplot(4,1,i+1)
    cbar = plt.pcolor(x[180:450],y1,model_data[i][:,180:450], norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap)
    plt.scatter(x[180:450],model_peaks[i][180:450], color = 'k', s=2)
    plt.title(titles[i])

colors = ['navy', 'firebrick', 'orange', 'forestgreen']
fig, (ax0, ax1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1, 3]}, sharex=True)
fig.subplots_adjust(hspace=0.025, wspace=0.05)

for i in range(1):
    ax0.plot(x[180:450],model_peaks[i][180:450], color = colors[i], label = titles[i],zorder=1)
    ind3 = np.arange(180,450)
    ind4=np.nanargmax(model_data[i][:,180:450], axis=0)
    ind_mod = [ind4,ind3]
    cbar = ax0.scatter(x[180:450],model_peaks[i][180:450],c=model_data[i][ind_mod],s=60,norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap,edgecolors='k',zorder=2)
    ax0.legend(loc=0)
    ax0.set_ylabel('Altitude (km)')
    
## PLOT MID-LAT SITES ##
#plt.setp(ax1.get_xticklabels(), visible=False)
for i in range(1,4):
    ax1.plot(x[180:450],model_peaks[i][180:450], color = colors[i], label = titles[i],zorder=1)
    ind3 = np.arange(180,450)
    ind4=np.nanargmax(model_data[i][:,180:450], axis=0)
    ind_mod = [ind4,ind3]
    ax1.scatter(x[180:450],model_peaks[i][180:450],c=model_data[i][ind_mod],s=60,norm=LogNorm(vmin=10**-2.5, vmax=10**-0.5), cmap=cmap,edgecolors='k',zorder=2)
    ax1.legend(loc=0)
    ax1.set_ylabel('Altitude (km)')
    ax1.set_xlabel('Time')
plt.xticks((x[6:15]*30), months[6:15])
plt.tight_layout()
fig.subplots_adjust(left = 0.05, right=0.85)#, hspace=0)
cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
cbar = fig.colorbar(cbar, extend='both', orientation = 'vertical', cax=cbar_ax)



