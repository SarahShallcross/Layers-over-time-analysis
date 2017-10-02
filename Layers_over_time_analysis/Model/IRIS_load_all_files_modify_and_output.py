#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:18:32 2017

@author: gy11s2s
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
