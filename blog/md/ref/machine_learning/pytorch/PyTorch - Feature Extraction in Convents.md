Convolutional neural networks include a primary feature, extraction.

Step 1
Import the respective models to create the feature extraction model with “PyTorch”.

import torch
import torch.nn as nn
from torchvision import models

Step 2
Create a class of feature extractor which can be called as and when needed.

class Feature_extractor(nn.module):
   def forward(self, input):
      self.feature = input.clone()
      return input
new_net = nn.Sequential().cuda() # the new network
target_layers = [conv_1, conv_2, conv_4] # layers you want to extract`
i = 1
for layer in list(cnn):
   if isinstance(layer,nn.Conv2d):
      name = "conv_"+str(i)
      art_net.add_module(name,layer)
      if name in target_layers:
         new_net.add_module("extractor_"+str(i),Feature_extractor())
      i+=1
   if isinstance(layer,nn.ReLU):
      name = "relu_"+str(i)
      new_net.add_module(name,layer)
   if isinstance(layer,nn.MaxPool2d):
      name = "pool_"+str(i)
      new_net.add_module(name,layer)
new_net.forward(your_image)
print (new_net.extractor_3.feature)

PyTorch - Visualization of Convents

Step 1
Import the necessary modules which is important for the visualization of conventional neural networks.

import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
import torch

Step 2
To stop potential randomness with training and testing data, call the respective data set as given in the code below −

seed = 128
rng = np.random.RandomState(seed)
data_dir = "../../datasets/MNIST"
train = pd.read_csv('../../datasets/MNIST/train.csv')
test = pd.read_csv('../../datasets/MNIST/Test_fCbTej3.csv')
img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'train', img_name)
img = imread(filepath, flatten=True)


Step 3
Plot the necessary images to get the training and testing data defined in perfect way using the below code −

pylab.imshow(img, cmap ='gray')
pylab.axis('off')
pylab.show()

