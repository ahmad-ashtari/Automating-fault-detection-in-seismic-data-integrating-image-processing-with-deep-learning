# -*- coding: utf-8 -*-
"""Visualization.ipynb

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Loading seimsic data files

seismic = np.load('/content/drive/MyDrive/seismic.npy')
chaos = np.load('/content/drive/MyDrive/chaos.npy')
variance= np.load('/content/drive/MyDrive/variance.npy')
mpc= np.load('/content/drive/MyDrive/most_positive_curvature.npy')
similarity= np.load('/content/drive/MyDrive/similarity.npy')
semblance= np.load('/content/drive/MyDrive/semblance.npy')

# Loading normalization matrices

samples_DF_max = np.load('/content/drive/MyDrive/samples_DF_max.npy')
samples_DF_mean = np.load('/content/drive/MyDrive/samples_DF_mean.npy')

# Loading trained DNN model

from tensorflow.keras.models import load_model

model = load_model('/content/drive/MyDrive/model_f3_2.hdf5')

# Predicting fault probability at a given inline number using the trained DNN model

Inline_n = 99


[dm1,dm2,dm3] = seismic.shape
dm = dm1*dm2

chaos_flat = chaos[:,:,Inline_n].reshape((dm,1))
variance_flat = variance[:,:,Inline_n].reshape((dm,1))
mpc_flat = mpc[:,:,Inline_n].reshape((dm,1))
similarity_flat = similarity[:,:,Inline_n].reshape((dm,1))
semblance_flat = semblance[:,:,Inline_n].reshape((dm,1))

test1 = np.concatenate((chaos_flat,variance_flat),axis=1)
test1 = np.concatenate((test1,mpc_flat),axis=1)
test1 = np.concatenate((test1,similarity_flat),axis=1)
test1 = np.concatenate((test1,semblance_flat),axis=1)


test1 = pd.DataFrame(test1)
test1 = test1/samples_DF_max
test1 = test1 - samples_DF_mean
test1 = np.array(test1)

test_f = model.predict(test1)

image_f = test_f.reshape(dm1,dm2)

# Plotting the predicted fault probability at the given inline section


masked_data = np.ma.masked_where(image_f < .5, image_f)
fig = plt.figure(figsize=(9,9))

plt.imshow(seismic[:,:,Inline_n], cmap='Greys')
plt.imshow(masked_data, cmap=cm.hot, interpolation='none')

plt.show()
