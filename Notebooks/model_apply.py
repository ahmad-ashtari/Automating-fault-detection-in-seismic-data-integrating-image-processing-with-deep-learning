# -*- coding: utf-8 -*-
"""model_apply.ipynb

"""

import numpy as np
import pandas as pd

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

# Predicting fault probability in the entire volume

[dm1,dm2,dm3] = seismic.shape
dm = dm1*dm2

fault_prob = []

for inline in range(0,dm3):
  chaos_flat = chaos[:,:,inline].reshape((dm,1))
  variance_flat = variance[:,:,inline].reshape((dm,1))
  mpc_flat = mpc[:,:,inline].reshape((dm,1))
  similarity_flat = similarity[:,:,inline].reshape((dm,1))
  semblance_flat = semblance[:,:,inline].reshape((dm,1))

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

  fault_prob.append(image_f)

fault_prob = np.array(fault_prob)
fault_prob = fault_prob.transpose(1,2,0)

# Saving fault probability model

np.save('/content/drive/MyDrive/f3_fault_probability_full_cube_new.npy', fault_prob)
