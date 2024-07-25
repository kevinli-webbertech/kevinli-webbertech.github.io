# Introduction

In Keras, a layer requires several key components to understand and process the input data effectively:

- **Shape of the input data** (`input_shape`): This defines the structure of the input data.
- **Initializers**: These are used to set the initial weights for each input.
- **Activations**: Functions that transform the output to introduce non-linearity.

To summarize, a Keras layer requires the following minimum details to create a complete layer:

- Shape of the input data
- Number of neurons/units in the layer
- Initializers
- Regularizers
- Constraints
- Activations


# Creating a Simple Keras Layer

Let's create a simple Keras layer using the Sequential model API to understand how Keras models and layers work.

## Code Example

```python
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import initializers
from keras import regularizers
from keras import constraints
```
# Initialize the Sequential model
model = Sequential()

# Add a Dense layer with specific parameters
model.add(Dense(32, input_shape=(16,),
                kernel_initializer='he_uniform',
                kernel_regularizer=None,
                kernel_constraint='MaxNorm',
                activation='relu'))

# Add additional Dense layers
model.add(Dense(16, activation='relu'))
model.add(Dense(8))

## Explanation

- **Line 1-5**: Imports the necessary modules.

- **Line 7**: Creates a new model using the Sequential API.

- **Line 9**: Adds a new Dense layer to the model. The `Dense` layer is a basic layer provided by Keras that accepts the number of neurons or units (32) as its required parameter. If it’s the first layer, you need to provide the `input_shape` (16,) as well. Otherwise, the output of the previous layer will be used as the input for the next layer. All other parameters are optional.

  - **First parameter**: Represents the number of units (neurons).
  - **`input_shape`**: Represents the shape of the input data.
  - **`kernel_initializer`**: Represents the initializer to be used. The `he_uniform` function is set as the value.
  - **`kernel_regularizer`**: Represents the regularizer to be used. `None` is set as the value.
  - **`kernel_constraint`**: Represents the constraint to be used. The `MaxNorm` function is set as the value.
  - **`activation`**: Represents the activation function to be used. The `relu` function is set as the value.

- **Line 10**: Adds a second Dense layer with 16 units and sets `relu` as the activation function.

- **Line 11**: Adds the final Dense layer with 8 units.


## Basic Concept of Layers

### Input Shape

In machine learning, all types of input data (text, images, videos) are first converted into arrays of numbers before being fed into algorithms. These arrays can be one-dimensional, two-dimensional (matrix), or multi-dimensional. The shape of these arrays, specified as a tuple of integers, indicates their dimensions. For example, `(4, 2)` represents a matrix with four rows and two columns.

```python
import numpy as np

shape = (4, 2)
input = np.zeros(shape)
print(input)
```
output
[
   [0. 0.]
   [0. 0.]
   [0. 0.]
   [0. 0.]
]

Similarly, a `(3,4,2)` three-dimensional matrix has three collections of `4x2` matrices (four rows and two columns).

```python
import numpy as np

shape = (3, 4, 2)
input = np.zeros(shape)
print(input)

```plaintext
[
   [[0. 0.] [0. 0.] [0. 0.] [0. 0.]]
   [[0. 0.] [0. 0.] [0. 0.] [0. 0.]]
   [[0. 0.] [0. 0.] [0. 0.] [0. 0.]]
]

>>>
```
## Initializers

In Machine Learning, weights are assigned to all input data. The Initializers module provides different functions to set these initial weights. Some of the Keras Initializer functions are as follows:

### Zeros

Generates 0 for all input data.

```python
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import initializers

my_init = initializers.Zeros()
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,),
                kernel_initializer=my_init))
```
### Ones

Generates 1 for all input data.

```python
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import initializers

my_init = initializers.Ones()
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,),
                kernel_initializer=my_init))
```
### Constant

Generates a constant value (say, 5) specified by the user for all input data.

```python
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import initializers

my_init = initializers.Constant(value=0)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,),
                kernel_initializer=my_init))

where, `value` represents the constant value.
```
### RandomNormal

Generates values using a normal distribution for input data.

```python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.RandomNormal(mean=0.0, 
stddev=0.05, seed=None) 
model.add(Dense(512, activation='relu', input_shape=(784,), 
   kernel_initializer=my_init))
```
where:

- `mean` represents the mean of the random values to generate
- `stddev` represents the standard deviation of the random values to generate
- `seed` represents the value to generate random numbers


### RandomUniform

Generates values using a uniform distribution of input data.

```python
from keras import initializers

my_init = initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
model.add(Dense(512, activation='relu', input_shape=(784,), 
   kernel_initializer=my_init))

where,

- `minval` represents the lower bound of the random values to generate.
- `maxval` represents the upper bound of the random values to generate.

```
### TruncatedNormal

Generates values using a truncated normal distribution of input data.

```python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer=my_init))
```
### VarianceScaling

Generates values based on the input shape and output shape of the layer along with the specified scale.

```python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.VarianceScaling(
   scale=1.0, mode='fan_in', distribution='normal', seed=None) 
model.add(Dense(512, activation='relu', input_shape=(784,), 
   kernel_initializer=my_init))

where,

- `scale` represents the scaling factor
- `mode` represents any one of `fan_in`, `fan_out`, or `fan_avg` values
- `distribution` represents either `normal` or `uniform`
```

### LeCun Normal

Generates values using the LeCun normal distribution of input data.

```python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.LeCunNormal(seed=None)
model.add(Dense(512, activation='relu', input_shape=(784,), 
   kernel_initializer=my_init))


stddev = `sqrt(1 / fan_in)`

where,
- 'fan_in' represent the number of input units.
```
### lecun_uniform
Generates value using lecun uniform distribution of input data.

'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.lecun_uniform(seed = None) 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_initializer = my_init))
It finds the limit using the below formula and then apply uniform distribution

limit = 'sqrt(3 / fan_in)'
where,

-fan_in represents the number of input units

-fan_out represents the number of output units
```
### glorot_normal
Generates value using glorot normal distribution of input data.

'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.glorot_normal(seed=None) model.add(
   Dense(512, activation = 'relu', input_shape = (784,), kernel_initializer = my_init)
)

It finds the stddev using the below formula and then apply normal distribution

stddev = 'sqrt(2 / (fan_in + fan_out))'
where,

-fan_in represents the number of input units

-fan_out represents the number of output units
```
###  glorot_uniform
Generates value using glorot uniform distribution of input data.

'''pytohn
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.glorot_uniform(seed = None) 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_initializer = my_init))
-It finds the limit using the below formula and then apply uniform distribution

limit = 'sqrt(6 / (fan_in + fan_out))'
where,

-fan_in represent the number of input units.

-fan_out represents the number of output units

### he_normal
Generates value using he normal distribution of input data.

'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.RandomUniform(minval = -0.05, maxval = 0.05, seed = None) 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_initializer = my_init))
-It finds the stddev using the below formula and then apply normal distribution.

stddev = 'sqrt(2 / fan_in)'
where,
-fan_in represent the number of input units.

### he_uniform

Generates value using he uniform distribution of input data.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.he_normal(seed = None) 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_initializer = my_init))
-It finds the limit using the below formula and then apply uniform distribution.

limit = 'sqrt(6 / fan_in)'
where,
-fan_in represent the number of input units.

### Orthogonal
Generates a random orthogonal matrix.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.Orthogonal(gain = 1.0, seed = None) 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_initializer = my_init))
where,
-gain represent the multiplication factor of the matrix.

### Identity
-Generates identity matrix.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.Identity(gain = 1.0) model.add(
   Dense(512, activation = 'relu', input_shape = (784,), kernel_initializer = my_init)
)
### Constraints

In machine learning, a constraint is set on the parameter (weight) during the optimization phase. The `constraints` module provides different functions to set constraints on the layer. Some of the constraint functions are as follows:

#### NonNeg
Constrains weights to be non-negative.

```python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import initializers 

my_init = initializers.Identity(gain=1.0) 
model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer=my_init))

where,
-kernel_constraint represent the constraint to be used in the layer.
```
### UnitNorm
Constrains weights to be unit norm.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import constraints 

my_constrain = constraints.UnitNorm(axis = 0) 
model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_constraint = my_constrain))
### MaxNorm
Constrains weight to norm less than or equals to the given value.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import constraints 

my_constrain = constraints.MaxNorm(max_value = 2, axis = 0) 
model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_constraint = my_constrain))
where,

-max_value represent the upper bound

axis represent the dimension in which the constraint to be applied. e.g. in Shape (2,3,4) axis 0 denotes first dimension, 1 denotes second dimension and 2 denotes third dimension

### MinMaxNorm
Constrains weights to be norm between specified minimum and maximum values.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import constraints 

my_constrain = constraints.MinMaxNorm(min_value = 0.0, max_value = 1.0, rate = 1.0, axis = 0) 
model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_constraint = my_constrain))
where, rate represent the rate at which the weight constrain is applied.

### Regularizers
In machine learning, regularizers are used in the optimization phase. It applies some penalties on the layer parameter during optimization. Keras regularization module provides below functions to set penalties on the layer. Regularization applies per-layer basis only.

### L1 Regularizer
It provides L1 based regularization.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import regularizers 

my_regularizer = regularizers.l1(0.) 
model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_regularizer = my_regularizer))
where, kernel_regularizer represent the rate at which the weight constrain is applied.

### L2 Regularizer
It provides L2 based regularization.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import regularizers 

my_regularizer = regularizers.l2(0.) 
model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,), 
   kernel_regularizer = my_regularizer))
### L1 and L2 Regularizer
It provides both L1 and L2 based regularization.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
from keras import regularizers 

my_regularizer = regularizers.l2(0.) 
model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,),
   kernel_regularizer = my_regularizer))


### Activations

In machine learning, an activation function is a special function used to determine whether a specific neuron is activated or not. Essentially, the activation function performs a nonlinear transformation of the input data, enabling neurons to learn better. The output of a neuron depends on the activation function.

```python
result = Activation(SUMOF(input * weight) + bias)


```
### linear
Applies Linear function. Does nothing.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'linear', input_shape = (784,)))

Where,
-activation refers the activation function of the layer. It can be specified simply by the name of the function and the layer will use corresponding activators.

### elu
Applies Exponential linear unit.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'elu', input_shape = (784,)))
### selu
Applies Scaled exponential linear unit.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'selu', input_shape = (784,)))
### relu
Applies Rectified Linear Unit.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,)))
### softmax
Applies Softmax function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'softmax', input_shape = (784,)))
### softplus
Applies Softplus function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'softplus', input_shape = (784,)))
### softsign
Applies Softsign function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'softsign', input_shape = (784,)))
### tanh
Applies Hyperbolic tangent function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 
model = Sequential() 
model.add(Dense(512, activation = 'tanh', input_shape = (784,)))
### sigmoid
Applies Sigmoid function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'sigmoid', input_shape = (784,)))
### hard_sigmoid
Applies Hard Sigmoid function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'hard_sigmoid', input_shape = (784,)))
### exponential
Applies exponential function.
'''python
from keras.models import Sequential 
from keras.layers import Activation, Dense 

model = Sequential() 
model.add(Dense(512, activation = 'exponential', input_shape = (784,)))
## Layers & Description

| Sr.No | Layers & Description | 
|-------|-----------------------| 
| 1     | **Dense Layer**<br>Dense layer is the regular deeply connected neural network layer. |
| 2     | **Dropout Layers**<br>Dropout is one of the important concepts in machine learning. |
| 3     | **Flatten Layers**<br>Flatten is used to flatten the input. |
| 4     | **Reshape Layers**<br>Reshape is used to change the shape of the input. |
| 5     | **Permute Layers**<br>Permute is also used to change the shape of the input using patterns. |
| 6     | **RepeatVector Layers**<br>RepeatVector is used to repeat the input for a set number, n, of times. |
| 7     | **Lambda Layers**<br>Lambda is used to transform the input data using an expression or function. |
| 8     | **Convolution Layers**<br>Keras contains a lot of layers for creating Convolution-based ANN, popularly called Convolutional Neural Network (CNN). |
| 9     | **Pooling Layer**<br>It is used to perform max pooling operations on temporal data. |
| 10    | **Locally Connected Layer**<br>Locally connected layers are similar to Conv1D layers but the difference is that Conv1D layer weights are shared, while here weights are unshared. |
| 11    | **Merge Layer**<br>It is used to merge a list of inputs. |
| 12    | **Embedding Layer**<br>It performs embedding operations in the input layer. |


