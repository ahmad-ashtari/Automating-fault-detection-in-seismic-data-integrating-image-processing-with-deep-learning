# -*- coding: utf-8 -*-
"""model_training .ipynb

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Loading seimsic data files

seismic = np.load('/content/drive/MyDrive/seismic.npy')
chaos = np.load('/content/drive/MyDrive/chaos.npy')
variance= np.load('/content/drive/MyDrive/variance.npy')
mpc= np.load('/content/drive/MyDrive/most_positive_curvature.npy')
similarity= np.load('/content/drive/MyDrive/similarity.npy')
semblance= np.load('/content/drive/MyDrive/semblance.npy')

# Picking fault samples in inline sections

data_f = []
label_f = []



for inline in range(0,300,1):
  image = seismic[:,:,inline]
  if image is None:
    print("Error loading image")
    exit()
  [d1,d2] = image.shape
  image1 = np.zeros([d1,d2])

  # Shi-Tomasi corner detection parameters
  maxCorners = 30       # default = 100
  qualityLevel = 0.01   # default = 0.01
  minDistance = 10       # default = 10

  corners = cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)

  # Draw corners on the image
  if corners is not None:
    corners = np.int0(corners)
    for i in corners:
      x, y = i.ravel()
      image1[y,x] = 1


  data_f.append(image)
  label_f.append(image1)


data_f = np.array(data_f)
label_f = np.array(label_f)

# Display the result
fig = plt.figure(figsize=(15,15))

plt.imshow( image, cmap='Greys')

fig = plt.figure(figsize=(15,15))

plt.imshow( image1)

# Picking non-fault samples in inline sections


from numpy import random

label_nf = []
[a1,a2,a3] = data_f.shape
for i in range(0,a1):
  image2 = np.zeros([d1,d2])

  for j in range(1,maxCorners):
    x1 = random.randint(0,d1)
    y1 = random.randint(0,d2)
    if label_f[i,x1,y1] == 0:
      image2[x1,y1] = 1

  label_nf.append(image2)

label_nf = np.array(label_nf)

fig = plt.figure(figsize=(15,15))
plt.imshow( image2)

# Selecting a sub-valume of attributes to be consistent with the seismic sub-volume which has been used for sample picking

inline = 300
chaos_f = chaos[:,:,:inline].transpose(2, 0, 1)
variance_f = variance[:,:,:inline].transpose(2, 0, 1)
mpc_f = mpc[:,:,:inline].transpose(2, 0, 1)
similarity_f = similarity[:,:,:inline].transpose(2, 0, 1)
semblance_f = semblance[:,:,:inline].transpose(2, 0, 1)

# finding other attributes value at the location of fault and non-fault samples and adding them to the arrays 'samples_f' and 'samples_nf'

ind_f = np.where(label_f ==1)
ind_nf = np.where(label_nf ==1)

ind_f = pd.DataFrame(ind_f)
ind_nf = pd.DataFrame(ind_nf)


[d3,d4] = ind_f.shape
value_c_f = np.zeros((d4,1))
for i in range(0,d4):
  value_c_f[i] = chaos_f[ind_f[i][0],ind_f[i][1],ind_f[i][2]]

[d5,d6] = ind_nf.shape
value_c_nf = np.zeros((d6,1))
for i in range(0,d6):
  value_c_nf[i] = chaos_f[ind_nf[i][0],ind_nf[i][1],ind_nf[i][2]]

value_c_f = np.transpose(value_c_f)
value_c_nf = np.transpose(value_c_nf)


value_v_f = np.zeros((d4,1))
for i in range(0,d4):
  value_v_f[i] = variance_f[ind_f[i][0],ind_f[i][1],ind_f[i][2]]

value_v_nf = np.zeros((d6,1))
for i in range(0,d6):
  value_v_nf[i] = variance_f[ind_nf[i][0],ind_nf[i][1],ind_nf[i][2]]

value_v_f = np.transpose(value_v_f)
value_v_nf = np.transpose(value_v_nf)

samples_f = np.concatenate((value_c_f,value_v_f),axis=0)
samples_nf = np.concatenate((value_c_nf,value_v_nf),axis=0)

value_mpc_f = np.zeros((d4,1))
for i in range(0,d4):
  value_mpc_f[i] = mpc_f[ind_f[i][0],ind_f[i][1],ind_f[i][2]]

value_mpc_nf = np.zeros((d6,1))
for i in range(0,d6):
  value_mpc_nf[i] = mpc_f[ind_nf[i][0],ind_nf[i][1],ind_nf[i][2]]

value_mpc_f = np.transpose(value_mpc_f)
value_mpc_nf = np.transpose(value_mpc_nf)

samples_f = np.concatenate((samples_f,value_mpc_f),axis=0)
samples_nf = np.concatenate((samples_nf,value_mpc_nf),axis=0)


value_si_f = np.zeros((d4,1))
for i in range(0,d4):
  value_si_f[i] = similarity_f[ind_f[i][0],ind_f[i][1],ind_f[i][2]]

value_si_nf = np.zeros((d6,1))
for i in range(0,d6):
  value_si_nf[i] = similarity_f[ind_nf[i][0],ind_nf[i][1],ind_nf[i][2]]

value_si_f = np.transpose(value_si_f)
value_si_nf = np.transpose(value_si_nf)

samples_f = np.concatenate((samples_f,value_si_f),axis=0)
samples_nf = np.concatenate((samples_nf,value_si_nf),axis=0)

value_se_f = np.zeros((d4,1))
for i in range(0,d4):
  value_se_f[i] = semblance_f[ind_f[i][0],ind_f[i][1],ind_f[i][2]]

value_se_nf = np.zeros((d6,1))
for i in range(0,d6):
  value_se_nf[i] = semblance_f[ind_nf[i][0],ind_nf[i][1],ind_nf[i][2]]

value_se_f = np.transpose(value_se_f)
value_se_nf = np.transpose(value_se_nf)

samples_f = np.concatenate((samples_f,value_se_f),axis=0)
samples_nf = np.concatenate((samples_nf,value_se_nf),axis=0)

# Allocating labels to the fault and non-fault samples and merge arrays'samples_f' and 'samples_nf to form array 'samples'

samples_f = np.concatenate((samples_f,np.ones((1,d4))))
samples_nf = np.concatenate((samples_nf,np.zeros((1,d6))))

samples = np.concatenate((samples_f,samples_nf),axis=1)
samples = np.transpose(samples)
np.random.shuffle(samples)

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

# Building DNN model

model = Sequential()

model.add(Dense(5,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(1024,activation='relu'))


model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=1e-4,amsgrad=False),loss='binary_crossentropy', metrics=['accuracy'])

# Normalizing the training data

samples_DF = pd.DataFrame(samples[:,:5])
samples_DF_max = samples_DF.max()
samples_DF = samples_DF/samples_DF_max
samples_DF_mean = samples_DF.mean()
samples_DF = samples_DF - samples_DF_mean

samples_arr = np.array(samples_DF)

np.save('/content/drive/MyDrive/samples_DF_max.npy', samples_DF_max)
np.save('/content/drive/MyDrive/samples_DF_mean.npy', samples_DF_mean)

# Setting Early stop function parameters for model training

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Training the model

model.fit(x=samples_arr, y=samples[:,5], validation_split=0.2, epochs=50, verbose=1, callbacks=[early_stop])

# Saving the trained DNN model

model.save('/content/drive/MyDrive/model_f3_2.hdf5')

model.summary()

