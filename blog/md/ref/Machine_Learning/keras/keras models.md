Keras model represents the actual neural network model. Keras provides a two mode to create the model, simple and easy to use Sequential API as well as more flexible and advanced Functional API

Sequential

The core idea of Sequential API is simply arranging the Keras layers in a sequential order
-A ANN model can be created by simply calling Sequential() API as specified below −

from keras.models import Sequential 
model = Sequential()

Add layers
To add a layer, simply create a layer using Keras layer API and then pass the layer through add() function as specified below −

from keras.models import Sequential 

model = Sequential() 
input_layer = Dense(32, input_shape=(8,)) model.add(input_layer) 
hidden_layer = Dense(64, activation='relu'); model.add(hidden_layer) 
output_layer = Dense(8) 
model.add(output_layer)

Here, we have created one input layer, one hidden layer and one output layer.

Access the model


Keras provides few methods to get the model information like layers, input data and output data. They are as follows −

model.layers − Returns all the layers of the model as list.

>>> layers = model.layers 
>>> layers 
[
   <keras.layers.core.Dense object at 0x000002C8C888B8D0>, 
   <keras.layers.core.Dense object at 0x000002C8C888B7B8>
   <keras.layers.core.Dense object at 0x 000002C8C888B898>
]
model.inputs − Returns all the input tensors of the model as list.

>>> inputs = model.inputs 
>>> inputs 
[<tf.Tensor 'dense_13_input:0' shape=(?, 8) dtype=float32>]

model.outputs − Returns all the output tensors of the model as list.

>>> outputs = model.outputs 
>>> outputs 
<tf.Tensor 'dense_15/BiasAdd:0' shape=(?, 8) dtype=float32>]

model.get_weights − Returns all the weights as NumPy arrays.

model.set_weights(weight_numpy_array) − Set the weights of the model.

Serialize the model


Keras provides methods to serialize the model into object as well as json and load it again later. They are as follows −

get_config() − IReturns the model as an object.

config = model.get_config()
from_config() − It accept the model configuration object as argument and create the model accordingly.

new_model = Sequential.from_config(config)
to_json() − Returns the model as an json object.

>>> json_string = model.to_json() 
>>> json_string '{"class_name": "Sequential", "config": 
{"name": "sequential_10", "layers": 
[{"class_name": "Dense", "config": 
{"name": "dense_13", "trainable": true, "batch_input_shape": 
[null, 8], "dtype": "float32", "units": 32, "activation": "linear", 
"use_bias": true, "kernel_initializer": 
{"class_name": "Vari anceScaling", "config": 
{"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}},
"bias_initializer": {"class_name": "Zeros", "conf 
ig": {}}, "kernel_regularizer": null, "bias_regularizer": null, 
"activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, 
{" class_name": "Dense", "config": {"name": "dense_14", "trainable": true, 
"dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, 
"kern el_initializer": {"class_name": "VarianceScaling", "config": 
{"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": null}}, 
"bias_initia lizer": {"class_name": "Zeros", 
"config": {}}, "kernel_regularizer": null, "bias_regularizer": null, 
"activity_regularizer": null, "kernel_constraint" : null, "bias_constraint": null}}, 
{"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, 
"dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, 
"kernel_initializer": {"class_name": "VarianceScaling", "config": 
{"scale": 1.0, "mode": "fan_avg", "distribution": " uniform", "seed": null}}, 
"bias_initializer": {"class_name": "Zeros", "config": {}}, 
"kernel_regularizer": null, "bias_regularizer": null, "activity_r egularizer": 
null, "kernel_constraint": null, "bias_constraint": 
null}}]}, "keras_version": "2.2.5", "backend": "tensorflow"}' 
>>>

model_from_json() − Accepts json representation of the model and create a new model.

from keras.models import model_from_json 
new_model = model_from_json(json_string)


to_yaml() − Returns the model as a yaml string.

>>> yaml_string = model.to_yaml() 
>>> yaml_string 'backend: tensorflow\nclass_name: 
Sequential\nconfig:\n layers:\n - class_name: Dense\n config:\n 
activation: linear\n activity_regular izer: null\n batch_input_shape: 
!!python/tuple\n - null\n - 8\n bias_constraint: null\n bias_initializer:\n 
class_name : Zeros\n config: {}\n bias_regularizer: null\n dtype: 
float32\n kernel_constraint: null\n 
kernel_initializer:\n cla ss_name: VarianceScaling\n config:\n 
distribution: uniform\n mode: fan_avg\n 
scale: 1.0\n seed: null\n kernel_regularizer: null\n name: dense_13\n 
trainable: true\n units: 32\n 
use_bias: true\n - class_name: Dense\n config:\n activation: relu\n activity_regularizer: null\n 
bias_constraint: null\n bias_initializer:\n class_name: Zeros\n 
config : {}\n bias_regularizer: null\n dtype: float32\n 
kernel_constraint: null\n kernel_initializer:\n class_name: VarianceScalin g\n 
config:\n distribution: uniform\n mode: fan_avg\n scale: 1.0\n 
seed: null\n kernel_regularizer: nu ll\n name: dense_14\n trainable: true\n 
units: 64\n use_bias: true\n - class_name: Dense\n config:\n 
activation: linear\n activity_regularizer: null\n 
bias_constraint: null\n bias_initializer:\n 
class_name: Zeros\n config: {}\n bias_regu larizer: null\n 
dtype: float32\n kernel_constraint: null\n 
kernel_initializer:\n class_name: VarianceScaling\n config:\n 
distribution: uniform\n mode: fan_avg\n 
scale: 1.0\n seed: null\n kernel_regularizer: null\n name: dense _15\n 
trainable: true\n units: 8\n 
use_bias: true\n name: sequential_10\nkeras_version: 2.2.5\n' 
>>>
model_from_yaml() − Accepts yaml representation of the model and create a new model.

from keras.models import model_from_yaml 
new_model = model_from_yaml(yaml_string)
Summarise the model
Understanding the model is very important phase to properly use it for training and prediction purposes. Keras provides a simple method, summary to get the full information about the model and its layers.

A summary of the model created in the previous section is as follows −

>>> model.summary() Model: "sequential_10" 
_________________________________________________________________ 
Layer (type) Output Shape Param 
#================================================================ 
dense_13 (Dense) (None, 32) 288 
_________________________________________________________________ 
dense_14 (Dense) (None, 64) 2112 
_________________________________________________________________ 
dense_15 (Dense) (None, 8) 520 
================================================================= 
Total params: 2,920 
Trainable params: 2,920 
Non-trainable params: 0 
_________________________________________________________________ 
>>>
Train and Predict the model
Model provides function for training, evaluation and prediction process. They are as follows −

compile − Configure the learning process of the model

fit − Train the model using the training data

evaluate − Evaluate the model using the test data

predict − Predict the results for new input.

Advertisement

Functional API
Sequential API is used to create models layer-by-layer. Functional API is an alternative approach of creating more complex models. Functional model, you can define multiple input or output that share layers. First, we create an instance for model and connecting to the layers to access input and output to the model. This section explains about functional model in brief.

Create a model
Import an input layer using the below module −

>>> from keras.layers import Input
Now, create an input layer specifying input dimension shape for the model using the below code −

>>> data = Input(shape=(2,3))

Define layer for the input using the below module −

>>> from keras.layers import Dense

Add Dense layer for the input using the below line of code −

>>> layer = Dense(2)(data) 
>>> print(layer) 
Tensor("dense_1/add:0", shape =(?, 2, 2), dtype = float32)

Define model using the below module −


from keras.models import Model

Create a model in functional way by specifying both input and output layer −

model = Model(inputs = data, outputs = layer)

The complete code to create a simple model is shown below −

from keras.layers import Input 
from keras.models import Model 
from keras.layers import Dense 

data = Input(shape=(2,3)) 
layer = Dense(2)(data) model = 
Model(inputs=data,outputs=layer) model.summary() 
_________________________________________________________________ 
Layer (type)               Output Shape               Param # 
================================================================= 
input_2 (InputLayer)       (None, 2, 3)               0 
_________________________________________________________________ 
dense_2 (Dense)            (None, 2, 2)               8 
================================================================= 
Total params: 8 
Trainable params: 8 
Non-trainable params: 0 
_________________________________________________________________


