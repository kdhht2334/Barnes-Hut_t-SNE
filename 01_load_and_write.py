#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:00:46 2019

@author: daehakim
@e-mail: kdhht5022@gmail.com
"""
import numpy as np

epoch = 75
img = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/x_src_{:02d}.npy'.format(epoch))
x = z = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/z_src_{:02d}.npy'.format(epoch)) * 1e2
y = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/y_src_{:02d}.npy'.format(epoch)).astype(np.str0)
y = np.asarray(y).astype(np.float32)

# Write array to txt
with open(r'/home/kdh/Desktop/tensorflow/bhtsne/manifold_db.txt', 'w') as f:
    for i in range(len(z)):
        f.write(" ".join(map(str, z[i])))
        f.write("\n")
f.close()