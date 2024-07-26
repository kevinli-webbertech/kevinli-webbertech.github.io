# Available Modules in Keras

## Initializers
Provides a list of initializer functions. Detailed information can be found in the [Keras Layers chapter](#) during the model creation phase of machine learning.

## Regularizers
Provides a list of regularizer functions. Detailed information can be found in the [Keras Layers chapter](#).

## Constraints
Provides a list of constraint functions. Detailed information can be found in the [Keras Layers chapter](#).

## Activations
Provides a list of activation functions. Detailed information can be found in the [Keras Layers chapter](#).

## Losses
Provides a list of loss functions. Detailed information can be found in the [Model Training chapter](#).

## Metrics
Provides a list of metrics functions. Detailed information can be found in the [Model Training chapter](#).

## Optimizers
Provides a list of optimizer functions. Detailed information can be found in the [Model Training chapter](#).

## Callbacks
Provides a list of callback functions. Useful during the training process to print intermediate data and to stop training based on conditions (e.g., EarlyStopping).

## Text Processing
Provides functions to convert text into NumPy arrays suitable for machine learning. Useful in the data preparation phase.

## Image Processing
Provides functions to convert images into NumPy arrays suitable for machine learning. Useful in the data preparation phase.

## Sequence Processing
Provides functions to generate time-based data from given input data. Useful in the data preparation phase.

## Backend
Provides functions related to backend libraries like TensorFlow and Theano.

## Utilities
Provides a variety of utility functions useful in deep learning.

## Backend Module

The Backend module is used for Keras backend operations. By default, Keras runs on top of the TensorFlow backend. However, you can switch to other backends such as Theano or CNTK if needed. 

The default backend configuration is specified in the `.keras/keras.json` file located in your root directory.

### Configuration

To configure the backend, you can modify the `keras.json` file with the desired backend. For example:

```json
{
  "floatx": "float32",
  "epsilon": 1e-07,
  "backend": "tensorflow",  // Change this to "theano" or "cntk" as needed
  "image_data_format": "channels_last"
}
```
## Keras Backend Module

You can import the Keras Backend module using the following code:

```python
>>> from keras import backend as k
```
### `get_uid()`

The `get_uid()` function generates a unique identifier (UID). Optionally, you can provide a `prefix` argument to customize the UID. Each call to `get_uid()` returns an incremented UID.

#### Example Usage

```python
>>> k.get_uid(prefix='')
1
>>> k.get_uid(prefix='')
reset_uids

It is used resets the uid value.

>>> k.reset_uids()

Now, again execute the get_uid(). This will be reset and change again to 1.

>>> k.get_uid(prefix='') 
```

### `placeholder`

The `placeholder()` function is used to instantiate a placeholder tensor. This is a simple placeholder that can hold a tensor with a specified shape.

#### Example Usage

To create a placeholder tensor with a 3-D shape, you can use:

```python
>>> data = k.placeholder(shape=(1, 3, 3))
>>> data
<tf.Tensor 'Placeholder_9:0' shape=(1, 3, 3) dtype=float32>

>>> data = k.placeholder(shape = (1,3,3)) 
>>> data 
<tf.Tensor 'Placeholder_9:0' shape = (1, 3, 3) dtype = float32> 

-If you use int_shape(), it will show the shape. 

>>> k.int_shape(data) (1, 3, 3)
```
### `dot`

The `dot()` function is used to perform matrix multiplication between two tensors. Consider `a` and `b` as two tensors with shapes `(4, 2)` and `(2, 3)`, respectively. The result `c` will have the shape `(4, 3)`.

#### Example Usage

```python
>>> a = k.placeholder(shape=(4, 2))
>>> b = k.placeholder(shape=(2, 3))
>>> c = k.dot(a, b)
>>> c
<tf.Tensor 'MatMul_3:0' shape=(4, 3) dtype=float32>

```
### `ones`

The `ones()` function initializes a tensor where all elements are set to one.

#### Example Usage

```python
>>> res = k.ones(shape=(2, 2))
>>> k.eval(res)
array([[1., 1.], [1., 1.]], dtype=float32)
```
### `batch_dot`

The `batch_dot()` function performs a batch-wise dot product between two tensors. Both input tensors must have at least 2 dimensions.

#### Example Usage

```python
>>> a_batch = k.ones(shape=(2, 3))
>>> b_batch = k.ones(shape=(3, 2))
>>> c_batch = k.batch_dot(a_batch, b_batch)
>>> c_batch
<tf.Tensor 'ExpandDims:0' shape=(2, 1) dtype=float32>

>>> data = k.variable([[10,20,30,40],[50,60,70,80]])
 ```
# variable initialized here 
>>> result = k.transpose(data) 
>>> print(result) 
Tensor("transpose_6:0", shape = (4, 2), dtype = float32) 
>>> print(k.eval(result)) 
   [[10. 50.] 
   [20. 60.] 
   [30. 70.] 
   [40. 80.]]
If you want to access from numpy −

>>> data = np.array([[10,20,30,40],[50,60,70,80]]) 

>>> print(np.transpose(data)) 
   [[10 50] 
   [20 60] 
   [30 70] 
   [40 80]] 

>>> res = k.variable(value = data) 
>>> print(res) 
<tf.Variable 'Variable_7:0' shape = (2, 4) dtype = float32_ref>

### `is_sparse(tensor)`

The `is_sparse()` function checks whether a given tensor is sparse or not.

#### Example Usage

```python
>>> a = k.placeholder((2, 2), sparse=True)
>>> print(a)
SparseTensor(indices=Tensor("Placeholder_8:0", shape=(?, 2), dtype=int64),
             values=Tensor("Placeholder_7:0", shape=(?), dtype=float32),
             dense_shape=Tensor("Const:0", shape=(2,), dtype=int64))
>>> print(k.is_sparse(a))
True
```
### `to_dense()`

The `to_dense()` function converts a sparse tensor into a dense tensor.

#### Example Usage

```python
>>> b = k.to_dense(a)
>>> print(b)
Tensor("SparseToDense:0", shape=(2, 2), dtype=float32)
>>> print(k.is_sparse(b))
False

```
### `random_uniform_variable`

The `random_uniform_variable()` function initializes a tensor using a uniform distribution. The tensor is created with a specified shape and values are drawn from a uniform distribution within a given range.

#### Parameters

- `shape`: A tuple representing the shape of the tensor (rows and columns).
- `low`: The lower bound of the uniform distribution.
- `high`: The upper bound of the uniform distribution.

#### Example Usage

```python
>>> a = k.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> b = k.random_uniform_variable(shape=(3, 2), low=0, high=1)
>>> c = k.dot(a, b)
>>> k.int_shape(c)
(2, 2)

```

## `utils` Module

The `utils` module provides a variety of utility functions for deep learning. Some of the methods provided by the `utils` module include:

### `HDF5Matrix`

The `HDF5Matrix` class is used to represent input data in HDF5 format. It allows for efficient loading and processing of large datasets stored in HDF5 files.

#### Example Usage

```python
from keras.utils import HDF5Matrix

data = HDF5Matrix('data.hdf5', 'data')
```
### `to_categorical`

The `to_categorical()` function converts a class vector (integers) into a binary class matrix, also known as one-hot encoding. This is useful for transforming class labels into a format suitable for classification tasks.

#### Example Usage

```python
from keras.utils import to_categorical

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
one_hot_labels = to_categorical(labels)
print(one_hot_labels)
```
### output
array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32)
```python
from keras.utils import normalize

data = [1, 2, 3, 4, 5]
normalized_data = normalize([data])
print(normalized_data)
```
### `print_summary`

The `print_summary()` function prints a summary of the model, including details such as the layers, their shapes, and the number of parameters. This is useful for inspecting the architecture of a Keras model.

#### Example Usage

```python
from keras.utils import print_summary

# Assuming 'model' is a pre-defined Keras model
print_summary(model)
```
### `plot_model`

The `plot_model()` function creates a graphical representation of the model architecture and saves it to a file. This visualization can help in understanding and analyzing the structure of the model.

#### Example Usage

```python
from keras.utils import plot_model
 ```
# Assuming 'model' is a pre-defined Keras model
plot_model(model, to_file='model_image.png', show_shapes=True, show_layer_names=True)


