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
from time import time

import cv2

from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

epoch = 75
img = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/x_src_{:02d}.npy'.format(epoch))
x = z = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/z_src_{:02d}.npy'.format(epoch)) * 1e2
y = np.load('/home/kdh/Desktop/tensorflow/bhtsne/data/manifold_db/y_src_{:02d}.npy'.format(epoch)).astype(np.str0)
y = np.asarray(y).astype(np.float32)


# Image interpolation
img_resized = np.zeros(shape=(440, 256, 256, 3))
for i in range(len(img)):
    img_resized[i] = cv2.resize(img[i], (256, 256), interpolation=cv2.INTER_CUBIC)


print("1. Computing random projection")
t0 = time()
X_projected = random_projection.SparseRandomProjection(n_components=2, random_state=42).fit_transform(x)
t1 = time()
#print("Computing time is {}s".format(t1-t0))

# Plot and save
fig, ax = plt.subplots(figsize=(60,48))
ax.scatter(X_projected[:,0], X_projected[:,1])
for i in range(len(img)):
    imagebox = OffsetImage(img_resized[i], zoom=0.2)

    ab = AnnotationBbox(imagebox, (X_projected[i][0],X_projected[i][1]), frameon=False)
    ax.add_artist(ab)

plt.draw()
plt.savefig('/home/kdh/Desktop/tensorflow/bhtsne/pic/random_resized.png',bbox_inches='tight')
plt.show()


print("2. Computing PCA projection")
t0 = time()
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(x)
t1 = time()
#print("Computing time is {}s".format(t1-t0))

# Plot and save
fig, ax = plt.subplots(figsize=(60,48))
ax.scatter(X_pca[:,0], X_pca[:,1])
for i in range(len(img)):
    imagebox = OffsetImage(img_resized[i], zoom=0.2)

    ab = AnnotationBbox(imagebox, (X_pca[i][0],X_pca[i][1]), frameon=False)
    ax.add_artist(ab)

plt.draw()
plt.savefig('/home/kdh/Desktop/tensorflow/bhtsne/pic/pca_resized.png',bbox_inches='tight')
plt.show()


print("3. Computing Spectral embedding")
t0 = time()
X_spec = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                    eigen_solver="arpack").fit_transform(x)
t1 = time()
#print("Computing time is {}s".format(t1-t0))

# Plot and save
fig, ax = plt.subplots(figsize=(60,48))
ax.scatter(X_spec[:,0], X_spec[:,1])
for i in range(len(img)):
    imagebox = OffsetImage(img_resized[i], zoom=0.2)

    ab = AnnotationBbox(imagebox, (X_spec[i][0],X_spec[i][1]), frameon=False)
    ax.add_artist(ab)

plt.draw()
plt.savefig('/home/kdh/Desktop/tensorflow/bhtsne/pic/spectral_resized.png',bbox_inches='tight')
plt.show()


print("4. Computing t-SNE embedding")
t0 = time()
X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(x)
t1 = time()
#print("Computing time is {}s".format(t1-t0))

# Plot and save
fig, ax = plt.subplots(figsize=(60,48))
ax.scatter(X_tsne[:,0], X_tsne[:,1])
for i in range(len(img)):
    imagebox = OffsetImage(img_resized[i], zoom=0.2)

    ab = AnnotationBbox(imagebox, (X_tsne[i][0],X_tsne[i][1]), frameon=False)
    ax.add_artist(ab)

plt.draw()
plt.savefig('/home/kdh/Desktop/tensorflow/bhtsne/pic/tsne_resized.png',bbox_inches='tight')
plt.show()
