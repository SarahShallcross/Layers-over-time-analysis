#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:34:00 2017

@author: gy11s2s
"""

import iris

dir_in = '/nfs/see-fs-01_users/gy11s2s/'

data = iris.load(dir_in+'xnbeca.pg1991jul',iris.AttributeConstraint(STASH = 'm01s02i162'))

data = data.concatenate_cube()

plt.figure()

for x in np.arange(0,85):
    print x
    
    plt.plot(lidar_jul.data[:,x])