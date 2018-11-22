#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:16:55 2018

@author: gy11s2s
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import fnmatch
#import datetime as datetime
#import pdb


### set file path
MLO_path = '/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'
print len(fnmatch.filter(os.listdir('/nfs/see-fs-01_users/gy11s2s/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'), '*.txt'))
#MLO_path = 'C:/Users/s_sha/Documents/PhD/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'
#print len(fnmatch.filter(os.listdir('C:/Users/s_sha/Documents/PhD/Python/NDACC_files/MLO/Plotting_months/June1991-Feb1992/'), '*.txt'))

### Set altitude template
altitude_template = np.arange(15500,30000,300)

### Check filepaths and read in all filenames
file_paths_MLO = []
file_names = sorted(os.listdir(MLO_path))
for name in file_names:
    filepath = os.path.join(MLO_path,name)
    if os.path.isfile(filepath) and filepath[-4:] == '.txt':
        file_paths_MLO.append(filepath)

### Read in actual data, separate altitude and BSR columns and use filename for the date
### Set empty arrays
bsr = np.zeros((84,59))
Dates = []
for i,files in enumerate(file_paths_MLO):
    data = np.loadtxt(fname=files, delimiter=',', skiprows=1, usecols=(0,1))
    col1 = data[:,0]
    col2 = data[:,1]
    alt = col1/1000
    bs=col2*0.001
    Date=pd.to_datetime(files[-14:-4], format='%Y.%m.%d')
    Date = Date.to_pydatetime()
    Dates.append(Date)

    bsr[:,i] = bs
        
#Set our background level and the delta they must exceed.
background = 2.2
delta = 1.5

all_layer_heights = []

#Define function to find layers
# Finding all indicies that go above 'background' and state that a layer is when it goes back to below that level.
def find_layer(alts,bs,start_index,background):
    
    """
    alts: altitude array
    bs: backscatter array
    start_index: index from where to start looking for layers
    background: background level backscatter must exceed to be a layer.
    """
    
    if np.min(bs) >= background:
        raise ValueError('No data below background level, cannot find layers!')
    #Find indicies where backscatter exceeds background
    above = np.where(bs[start_index:] > background)[0]
    
    
    #Check there are places where it exceeds
    if len(above)>1:
        
        #Set the index for the bottom of the layer (remembering we started at start_index)
        lower = int(above[0] + start_index)
        
        #Now find where the backscatter dips below background (starting from lower)
        below = np.where(bs[lower:] < background)[0]
        
        #Check there are places where it goes back below
        if len(below) > 0:
        
            upper =int(below[0] + lower)
        
        #otherwise set upper to the end of the array
        else:
            
            upper = int(bs.size-1)
     
    #If it doesn't exceed background anywhere, set the lower/upper to false
    else:
        
        lower = False
        upper = False
        
    return lower, upper


### Section to find upper and lower of any and all layers per day, appends to 'layers' per day and then to 'all_layer_heights'
#Loop over all dates
for i in range(len(file_paths_MLO)):
    
    #Set start_index to zero and this dates 'layers' list to an empty one.
    start_index = 0
    layers = []

    #Keep looping until we have found all layers upto the end of the array
    while start_index < int(bsr.shape[0]):
        print(start_index)
        #Find layer
        lower, upper = find_layer(altitude_template,bsr[:,i],start_index,background)
        #pdb.set_trace()
        #Check they values are not false
        if lower and upper:
            #Check the layer found exceeds background+delta
            if np.max(bsr[:,i][lower:upper]) > (background+delta):
            
            #save this layer as a tuple in our list of 'layers' for this date
                layers.append((alt[lower],alt[upper]))
            #start looking from the top of the current layer
                start_index = upper
            
            else:
            #If not bigger than background+delta, then don't save and start looking from the top of this layer.
                start_index = upper
            
        else:
        #if lower/upper are false, we've reached the end of this dates array. Set start index to the end of the
        #array (to trigger while statement) and save all these layers into the 'ith' element in the layer heights list.
            start_index = int(bsr.shape[0])
        
    all_layer_heights.append(layers)
    
    
m=0
for i in all_layer_heights:
    if len(i) > m:
        m = len(i)

layers = np.zeros((len(all_layer_heights),m*2))


for i in range(len(all_layer_heights)):
    layers[i,:2*len(all_layer_heights[i])] =np.array(all_layer_heights[i]).reshape(1,2*len(all_layer_heights[i]))
 
out_name = '/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Lidar/all_layer_heights.txt'
#out_name = 'C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Lidar/all_layer_heights.txt'
np.savetxt(out_name, layers, delimiter = '\t')

layer1_diff = []
for i in all_layer_heights[7:15]:
    layer1_diff.append(np.diff(i[0]))


### Check layers per day

plt.figure(figsize=(20,60))

for i in range(int(bsr.shape[1])):
    plt.subplot(10,6,i+1)
    plt.plot(alt, bsr[:,i])
    
    #plot each of the tuples (if any) in all_layer_heights
    for x in all_layer_heights[i]:
        plt.plot(x[0],background,'ro')
        plt.plot(x[1],background,'ro')
    
    #show the background level
    #plt.ylim([0,0.03])
    plt.axhline(background,ls='dashed',c='k')
plt.suptitle('Background = ' + str(background) + ", delta = " + str(delta))
plt.show()


###################################################
##################################################
    

### MODEL ##
import iris
import iris.coords as icoords
level_number_coord = icoords.DimCoord(range(1,86), standard_name='model_level_number')
model_height = np.loadtxt('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter=',')
#model_height = np.loadtxt('C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/model_height_levels.txt', delimiter = ',')
MLO_550 = iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_10.nc')
#MLO_550 = iris.load_cube('C:/Users/s_sha\Documents/PhD/Python/Layers_over_time_analysis/Model/xnbec/Extinction_sites/Extinction_550nm_site_10.nc')
MLO_550 = np.transpose(MLO_550.data)
MLO_data_masked_array=np.ma.masked_where(MLO_550<=0, MLO_550)
MLO_data_masked_array_km=MLO_data_masked_array*1000


def daily_mean(x):
    return np.mean(x.reshape(-1,8),axis=1)

daily_MLO_550 = np.apply_along_axis(daily_mean, axis=1, arr=MLO_data_masked_array_km)


time = np.array(range(0,(30*15))) 
months = ['Dec 1990', 'Jan 1991', 'Feb 1991','Mar 1991', 'Apr 1991', 'May 1991', 'Jun 1991', 'Jul 1991', 'Aug 1991', 'Sep 1991', 'Oct 1991', 'Nov 1991', 'Dec 1991', 'Jan 1992', 'Feb 1992','Mar 1992']

y = list(level_number_coord.points) # height for model heights
y1 = model_height/1000

#Set our background level and the delta they must exceed.
background = 0.003
delta = 0.001

all_layer_heights_model = []

data_search = daily_MLO_550[36:72,:]

for i in range(int(data_search.shape[1])):
#for i in range(241,242):    
    #Set start_index to zero and this dates 'layers' list to an empty one.
    start_index = 0
    layers_model = []
    #Keep looping until we have found all layers upto the end of the array
    while start_index < int(data_search.shape[0]):
        #print(start_index)
        #Find layer
        lower, upper = find_layer(y1[36:72],data_search[:,i],start_index,background)
        #pdb.set_trace()
        #Check they values are not false
        if type(lower) == int and type(upper) == int:
            #Check the layer found exceeds background+delta
            if np.max(data_search[lower:upper,i]) > (background+delta):
            
            #save this layer as a tuple in our list of 'layers' for this date
                layers_model.append((y1[36:72][lower],y1[36:72][upper]))
        
            #start looking from the top of the current layer
                start_index = upper
            
            else:
            #If not bigger than background+delta, then don't save and start looking from the top of this layer.
                start_index = upper
            
        else:
        #if lower/upper are false, we've reached the end of this dates array. Set start index to the end of the
        #array (to trigger while statement) and save all these layers into the 'ith' element in the layer heights list.
            start_index = int(data_search.shape[0])
            
  
        
    all_layer_heights_model.append(layers_model)

    
m=0
for i in all_layer_heights_model:
    if len(i) > m:
        m = len(i)

layers_model = np.zeros((len(all_layer_heights_model),m*2))


for i in range(len(all_layer_heights_model)):
    layers_model[i,:2*len(all_layer_heights_model[i])] =np.array(all_layer_heights_model[i]).reshape(1,2*len(all_layer_heights_model[i]))


out_name = '/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/Model/all_layer_heights_model.txt'
#out_name = 'C:/Users/s_sha/Documents/PhD/Python/Layers_over_time_analysis/Model/all_layer_heights_model.txt'
np.savetxt(out_name, layers_model[180:450], delimiter = '\t')


### Check layers per day

plt.figure(figsize=(20,60))

j=0
for i in range(210,230):
    plt.subplot(2,10,j+1)
    plt.plot(y1[36:72], data_search[:,i])
    j+=1
    #plot each of the tuples (if any) in all_layer_heights
    for x in all_layer_heights_model[i]:
        plt.plot(x[0],background,'ro')
        plt.plot(x[1],background,'ro')
        plt.xlim(10,34)
    #show the background level
    #plt.ylim([0,0.03])
    plt.axhline(background,ls='dashed',c='k')
plt.suptitle('Background = ' + str(background) + ", delta = " + str(delta))
plt.show()

###################################################







