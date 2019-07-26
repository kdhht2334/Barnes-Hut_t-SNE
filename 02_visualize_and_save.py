#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:00:46 2019

@author: daehakim
@e-mail: kdhht5022@gmail.com
"""

import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt

import cv2

epoch = 75
img = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/x_src_{:02d}.npy'.format(epoch))
x = z = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/z_src_{:02d}.npy'.format(epoch)) * 1e2
y = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/y_src_{:02d}.npy'.format(epoch)).astype(np.str0)
y = np.asarray(y).astype(np.float32)

# Load new txt
txt_file = open('/home/kdh/Desktop/tensorflow/bhtsne/manifold_db.tsne.txt','r')
lines = txt_file.readlines()

axis_arr = np.zeros(shape=(len(lines),2))
for i in range(len(lines)):
    axis_arr[i,0] = lines[i].split('\t')[0]
    axis_arr[i,1] = lines[i].split('\t')[1]

# Plot and save
fig, ax = plt.subplots(figsize=(10,8))
ax.scatter(axis_arr[:440,0], axis_arr[:440,1])
for i in range(len(img)):
    imagebox = OffsetImage(img[i], zoom=0.2)

    ab = AnnotationBbox(imagebox, (axis_arr[i][0],axis_arr[i][1]), frameon=False)
    ax.add_artist(ab)

plt.draw()
plt.savefig('/home/kdh/Desktop/tensorflow/bhtsne/pic/bhtsne.png',bbox_inches='tight')
plt.show()

# If you want to see the high pixel one, do bellow :)

# Image interpolation
img_resized = np.zeros(shape=(440, 256, 256, 3))
for i in range(len(img)):
    img_resized[i] = cv2.resize(img[i], (256, 256), interpolation=cv2.INTER_CUBIC)
    
    
# Plot and save
fig, ax = plt.subplots(figsize=(60,48))
ax.scatter(axis_arr[:440,0], axis_arr[:440,1])
for i in range(len(img)):
    imagebox = OffsetImage(img_resized[i], zoom=0.2)

    ab = AnnotationBbox(imagebox, (axis_arr[i][0],axis_arr[i][1]), frameon=False)
    ax.add_artist(ab)

plt.draw()
plt.savefig('/home/kdh/Desktop/tensorflow/bhtsne/pic/bhtsne_resized.png',bbox_inches='tight')
plt.show()