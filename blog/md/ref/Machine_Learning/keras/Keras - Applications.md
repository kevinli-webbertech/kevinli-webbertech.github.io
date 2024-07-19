Keras applications module is used to provide pre-trained model for deep neural networks.

Pre-trained models

model consists of two parts model Architecture and model Weights.

Some of the popular pre-trained models

ResNet
VGG16
MobileNet
InceptionResNetV2
InceptionV3

Loading a model

import keras 
import numpy as np 

from keras.applications import vgg16, inception_v3, resnet50, mobilenet 

#Load the VGG model 
vgg_model = vgg16.VGG16(weights = 'imagenet') 

#Load the Inception_V3 model 
inception_model = inception_v3.InceptionV3(weights = 'imagenet') 

#Load the ResNet50 model 
resnet_model = resnet50.ResNet50(weights = 'imagenet') 

#Load the MobileNet model mobilenet_model = mobilenet.MobileNet(weights = 'imagenet')

Once the model is loaded, we can immediately use it for prediction purpose

