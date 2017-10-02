#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:53:32 2017

@author: eelrm
"""
import iris
import iris.coords as icoords
import iris.quickplot as qplt
import matplotlib.pyplot as plt


level_number_coord = icoords.DimCoord(range(1,86), standard_name='model_level_number') # make a model level number coord

extinction = iris.load('/nfs/a133/eelrm/sarah_data/xnbeca.*',iris.AttributeConstraint(STASH='m01s02i162')) # this reads in all timesteps of EXTINCTION AT 550!!!
extinction = extinction.concatenate_cube() # this is dodge (sorry)

#site1 = extinction[:,0:85] # site 1 is all time steps but the first 85 of the second dimension.
#site1.remove_coord('model_level_number')

dir_out = '/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/model_extinction_sites/' # set this

n=0 # initialize n

for i in range(0,16):
    # loop through 16 times 
    
    site = extinction[:,n:n+85]
    
    site.remove_coord('model_level_number') # get rid of old metadata incase it's dodge
    site.remove_coord('level_height')
    site.remove_coord('sigma')
    
    site.add_dim_coord(level_number_coord,1)  # add new model level number coordinate to the second dimension in your cube
    
    site.rename('Extinction_550nm_site_'+str(i)) # rename cube with correct site number
    
    iris.save(site,dir_out+site.name()+'.nc') # save it as a netcdf file
    
    n+=85 # add 85 to n counter

# How to plot:

#x = range(0,3600) # look into datetime if want actual dates on bottom as opposed to numbers.

#y = level_number_coord.points # this is 1 to 85

#data = cube.data # only cube.data IF using matplotlib, not if using iris plotting functions

#plt.contourf(x,y,data) # matplotlib function
# load one site:
    
site_MLO=iris.load_cube('/nfs/see-fs-01_users/gy11s2s/Python/Layers_over_time_analysis/model_extinction_sites/Extinction_550nm_site_10.nc')

x = range(0,3600)
y = list(level_number_coord.points)
data = site_MLO

plt.contourf(range(0,3600), y, data.data)
#all_sites=iris.load('nameofncfiles*')

#-you will need to change model level number to level height!