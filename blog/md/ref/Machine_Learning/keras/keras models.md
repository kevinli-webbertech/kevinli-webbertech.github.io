# Keras Models


Keras models represent the actual neural network models used in deep learning tasks. Keras provides two primary methods for creating models:

1. **Sequential API**: A simple and easy-to-use method for creating models. It allows you to build models layer by layer in a straightforward manner.

2. **Functional API**: A more flexible and advanced method for creating models. It is used for creating complex models, such as those with multiple inputs or outputs, or models with shared layers.

Both APIs are powerful tools in Keras for designing and training neural networks.

**Sequential API**:

The core idea of Sequential API is simply arranging the Keras layers in a sequential order
A ANN model can be created by simply calling Sequential() API as specified below 
'''python
from keras.models import Sequential 
model = Sequential()

## Add Layers

To add a layer, simply create a layer using the Keras layer API and then pass the layer through the `add()` function as specified below:

```python
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Create and add layers
input_layer = Dense(32, input_shape=(8,))
model.add(input_layer)

hidden_layer = Dense(64, activation='relu')
model.add(hidden_layer)

output_layer = Dense(8)
model.add(output_layer)

Here,
-we have created one input layer, one hidden layer and one output layer.

### Access the Model

Keras provides several methods to get information about the model, including its layers, input data, and output data. They are as follows:

- **`model.layers`** – Returns all the layers of the model as a list.

  ```python
  >>> layers = model.layers
  >>> layers
  [
     <keras.layers.core.Dense object at 0x000002C8C888B8D0>, 
     <keras.layers.core.Dense object at 0x000002C8C888B7B8>,
     <keras.layers.core.Dense object at 0x000002C8C888B898>
  ]

- **`model.inputs`** – Returns all the input tensors of the model as a list.

  ```python
  >>> inputs = model.inputs
  >>> inputs
  [<tf.Tensor 'dense_13_input:0' shape=(?, 8) dtype=float32>]


- **`model.outputs`** – Returns all the output tensors of the model as a list.

  ```python
  >>> outputs = model.outputs
  >>> outputs
  [<tf.Tensor 'dense_15/BiasAdd:0' shape=(?, 8) dtype=float32>]


- **`model.get_weights()`** – Returns all the weights as NumPy arrays.

- **`model.set_weights(weight_numpy_array)`** – Sets the weights of the model.


### Serialize the Model

Keras provides methods to serialize the model into an object or JSON and load it again later. They are as follows:

- **`get_config()`** – Returns the model as an object.

  ```python
  config = model.get_config()

- **`from_config()`** – Accepts the model configuration object as an argument and creates the model accordingly.

  ```python
  new_model = Sequential.from_config(config)

- **`to_json()`** – Returns the model as a JSON object.

  ```python
  json_string = model.to_json()


{
  "class_name": "Sequential",
  "config": {
    "name": "sequential_10",
    "layers": [
      {
        "class_name": "Dense",
        "config": {
          "name": "dense_13",
          "trainable": true,
          "batch_input_shape": [null, 8],
          "dtype": "float32",
          "units": 32,
          "activation": "linear",
          "use_bias": true,
          "kernel_initializer": {
            "class_name": "VarianceScaling",
            "config": {
              "scale": 1.0,
              "mode": "fan_avg",
              "distribution": "uniform",
              "seed": null
            }
          },
          "bias_initializer": {
            "class_name": "Zeros",
            "config": {}
          },
          "kernel_regularizer": null,
          "bias_regularizer": null,
          "activity_regularizer": null,
          "kernel_constraint": null,
          "bias_constraint": null
        }
      },
      {
        "class_name": "Dense",
        "config": {
          "name": "dense_14",
          "trainable": true,
          "dtype": "float32",
          "units": 64,
          "activation": "relu",
          "use_bias": true,
          "kernel_initializer": {
            "class_name": "VarianceScaling",
            "config": {
              "scale": 1.0,
              "mode": "fan_avg",
              "distribution": "uniform",
              "seed": null
            }
          },
          "bias_initializer": {
            "class_name": "Zeros",
            "config": {}
          },
          "kernel_regularizer": null,
          "bias_regularizer": null,
          "activity_regularizer": null,
          "kernel_constraint": null,
          "bias_constraint": null
        }
      },
      {
        "class_name": "Dense",
        "config": {
          "name": "dense_15",
          "trainable": true,
          "dtype": "float32",
          "units": 8,
          "activation": "linear",
          "use_bias": true,
          "kernel_initializer": {
            "class_name": "VarianceScaling",
            "config": {
              "scale": 1.0,
              "mode": "fan_avg",
              "distribution": "uniform",
              "seed": null
            }
          },
          "bias_initializer": {
            "class_name": "Zeros",
            "config": {}
          },
          "kernel_regularizer": null,
          "bias_regularizer": null,
          "activity_regularizer": null,
          "kernel_constraint": null,
          "bias_constraint": null
        }
      }
    ],
    "keras_version": "2.2.5",
    "backend": "tensorflow"
  }
}

**'model_from_json()'** − Accepts json representation of the model and create a new model.

### `to_yaml()` – Returns the model as a YAML string.

```python
from keras.models import model_from_json

# Convert model to YAML



backend: tensorflow
class_name: Sequential
config:
  layers:
    - class_name: Dense
      config:
        activation: linear
        activity_regularizer: null
        batch_input_shape: !!python/tuple
          - null
          - 8
        bias_constraint: null
        bias_initializer:
          class_name: Zeros
          config: {}
        bias_regularizer: null
        dtype: float32
        kernel_constraint: null
        kernel_initializer:
          class_name: VarianceScaling
          config:
            distribution: uniform
            mode: fan_avg
            scale: 1.0
            seed: null
        kernel_regularizer: null
        name: dense_13
        trainable: true
        units: 32
        use_bias: true
    - class_name: Dense
      config:
        activation: relu
        activity_regularizer: null
        bias_constraint: null
        bias_initializer:
          class_name: Zeros
          config: {}
        bias_regularizer: null
        dtype: float32
        kernel_constraint: null
        kernel_initializer:
          class_name: VarianceScaling
          config:
            distribution: uniform
            mode: fan_avg
            scale: 1.0
            seed: null
        kernel_regularizer: null
        name: dense_14
        trainable: true
        units: 64
        use_bias: true
    - class_name: Dense
      config:
        activation: linear
        activity_regularizer: null
        bias_constraint: null
        bias_initializer:
          class_name: Zeros
          config: {}
        bias_regularizer: null
        dtype: float32
        kernel_constraint: null
        kernel_initializer:
          class_name: VarianceScaling
          config:
            distribution: uniform
            mode: fan_avg
            scale: 1.0
            seed: null
        kernel_regularizer: null
        name: dense_15
        trainable: true
        units: 8
        use_bias: true
  name: sequential_10
  keras_version: 2.2.5

**'model_from_yaml()'** − Accepts yaml representation of the model and create a new model.
'''python
from keras.models import model_from_yaml 
new_model = model_from_yaml(yaml_string)
### Summarize the Model

Understanding the model is an important phase to properly use it for training and prediction purposes. Keras provides a simple method, `summary()`, to get detailed information about the model and its layers.

#### Usage:

```python
# Summarize the model
model.summary()


>>> model.summary()
Model: "sequential_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 32)                288       
_________________________________________________________________
dense_14 (Dense)             (None, 64)                2112      
_________________________________________________________________
dense_15 (Dense)             (None, 8)                 520       
=================================================================
Total params: 2,920
Trainable params: 2,920
Non-trainable params: 0
_________________________________________________________________
>>>
### Train and Predict the Model

A Keras model provides several methods for training, evaluation, and prediction. These methods include:

- **`compile`**: Configures the learning process of the model.
- **`fit`**: Trains the model using the training data.
- **`evaluate`**: Evaluates the model using the test data.
- **`predict`**: Predicts the results for new input.

### Functional API

While the Sequential API is useful for creating models layer-by-layer, the Functional API offers a more flexible approach for building complex models. With the Functional API, you can define models with multiple inputs and outputs that share layers.

#### Key Concepts:

1. **Defining the Model**

   The Functional API allows you to create models by defining inputs and outputs explicitly, and connecting them through layers. This flexibility enables the construction of more complex network architectures compared to the Sequential API.

2. **Creating a Functional Model**

   Here's a brief overview of how to create a model using the Functional API:

   ```python
   from keras.layers import Input, Dense
   from keras.models import Model

   # Define the input layer
   input_layer = Input(shape=(784,))

   # Define the hidden layers
   hidden_layer1 = Dense(64, activation='relu')(input_layer)
   hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

   # Define the output layer
   output_layer = Dense(10, activation='softmax')(hidden_layer2)

   # Create the model
   model = Model(inputs=input_layer, outputs=output_layer)
### Explanation

- `Input(shape=(784,))`: Creates an input layer that accepts input data with 784 features.

- `Dense(64, activation='relu')(input_layer)`: Adds a dense layer with 64 units and ReLU activation. This layer receives its input from the `input_layer`.

- `Dense(64, activation='relu')(hidden_layer1)`: Adds another dense layer with 64 units and ReLU activation. This layer receives its input from the first hidden layer, `hidden_layer1`.

- `Dense(10, activation='softmax')(hidden_layer2)`: Adds the output layer with 10 units and softmax activation. This layer receives its input from the second hidden layer, `hidden_layer2`, and is suitable for classification tasks with 10 classes.


