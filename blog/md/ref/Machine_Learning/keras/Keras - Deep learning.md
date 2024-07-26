# Keras Framework Overview

Keras provides a complete framework to create any type of neural networks.
- It supports both simple and complex neural network models as well.


## Architecture of Keras

Keras API can be divided into three main categories:
- Model
- Layer
- Core Modules

In Keras, every ANN is represented by Keras Models. In turn, every Keras Model is a composition of Keras Layers and represents ANN layers like input, hidden layer, output layers, convolution layer, pooling layer, etc. Keras model and layer access Keras modules for activation function, loss function, regularization function, etc. Using Keras model, Keras Layer, and Keras modules, any ANN algorithm (CNN, RNN, etc.) can be represented in a simple and efficient manner.

The following diagram depicts the relationship between model, layer, and core modules:

## Model

### 2 Types:

1. **Sequential Model**
   - Linear composition of Keras layers
   - Has the ability to represent nearly all available neural networks

#### A Simple Sequential Model
```python
from keras.models import Sequential 
from keras.layers import Dense, Activation 

model = Sequential()  
model.add(Dense(512, activation='relu', input_shape=(784,)))

Sequential model exposes the Model class to create customized models. You can use the sub-classing concept to create your own complex model.
```
### Functional API
Functional API is basically used to create complex models.
## Layer

Each Keras layer in the Keras model represents the corresponding layer (input layer, hidden layer, and output layer) in the actual proposed neural network model.

- Pre-built layers so that any complex neural network can be easily created.

### Important Keras Layers

- Core Layers
- Convolution Layers
- Pooling Layers
- Recurrent Layers

## Simple Python Code to Represent a Neural Network Model Using Sequential Model

```python
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout 

model = Sequential() 

model.add(Dense(512, activation='relu', input_shape=(784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation='softmax'))
```

## Explanation of the Neural Network Model Code

- **Line 1**: Imports the `Sequential` model from Keras models.
- **Line 2**: Imports the `Dense` layer and `Activation` module.
- **Line 4**: Creates a new sequential model using the Sequential API.
- **Line 5**: Adds a dense layer (`Dense` API) with ReLU activation (using the `Activation` module) function.
- **Line 6**: Adds a dropout layer (`Dropout` API) to handle overfitting.
- **Line 7**: Adds another dense layer (`Dense` API) with ReLU activation (using the `Activation` module) function.
- **Line 8**: Adds another dropout layer (`Dropout` API) to handle overfitting.
- **Line 9**: Adds the final dense layer (`Dense` API) with softmax activation (using the `Activation` module) function.

## Core Modules

- **Activations Module**: 
  - Activation functions are an important concept in ANN. The activations module provides many activation functions like `softmax`, `relu`, etc.

- **Loss Module**: 
  - The loss module provides loss functions such as `mean_squared_error`, `mean_absolute_error`, `poisson`, etc.

- **Optimizer Module**: 
  - The optimizer module provides optimizer functions like `adam`, `sgd`, etc.

- **Regularizers**: 
  - The regularizer module provides functions like `L1 regularizer`, `L2 regularizer`, etc.








