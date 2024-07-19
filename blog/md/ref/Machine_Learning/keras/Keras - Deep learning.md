Keras provides a complete framework to create any type of neural networks.
-it supports simpla and complex neural network models as well.

Architecture of Keras

Keras API can be divided into three main categories −
Model
Layer
Core Modules

In Keras, every ANN is represented by Keras Models. In turn, every Keras Model is composition of Keras Layers and represents ANN layers like input, hidden layer, output layers, convolution layer, pooling layer, etc., Keras model and layer access Keras modules for activation function, loss function, regularization function, etc., Using Keras model, Keras Layer, and Keras modules, any ANN algorithm (CNN, RNN, etc.,) can be represented in a simple and efficient manner.

The following diagram depicts the relationship between model, layer and core modules −


Model
-2 types
Sequential Model 
-linear composition of keras layers
-has the ability to represent nearly all available neural networks.
A simple sequential model

from keras.models import Sequential 
from keras.layers import Dense, Activation 

model = Sequential()  
model.add(Dense(512, activation = 'relu', input_shape = (784,)))

equential model exposes Model class to create customized models.can use sub-classing concept to create our own complex model.

Functional API
Functional API is basically used to create complex models.

Layer

Each Keras layer in the Keras model represent the corresponding layer (input layer, hidden layer and output layer) in the actual proposed neural network model.

-pre-build layers so that any complex neural network can be easily created.

-important Keras layers

Core Layers
Convolution Layers
Pooling Layers
Recurrent Layers

simple python code to represent a neural network model using sequential model 


from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout model = Sequential() 

model.add(Dense(512, activation = 'relu', input_shape = (784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation = 'relu')) model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation = 'softmax'))

Line 1 imports Sequential model from Keras models

Line 2 imports Dense layer and Activation module

Line 4 create a new sequential model using Sequential API

Line 5 adds a dense layer (Dense API) with relu activation (using Activation module) function.

Line 6 adds a dropout layer (Dropout API) to handle over-fitting.

Line 7 adds another dense layer (Dense API) with relu activation (using Activation module) function.

Line 8 adds another dropout layer (Dropout API) to handle over-fitting.

Line 9 adds final dense layer (Dense API) with softmax activation (using Activation module) function.

Core Modules

Activations module − Activation function is an important concept in ANN and activation modules provides many activation function like softmax, relu, etc.,

Loss module − Loss module provides loss functions like mean_squared_error, mean_absolute_error, poisson, etc.,

Optimizer module − Optimizer module provides optimizer function like adam, sgd, etc.,

Regularizers − Regularizer module provides functions like L1 regularizer, L2 regularizer, etc.,








