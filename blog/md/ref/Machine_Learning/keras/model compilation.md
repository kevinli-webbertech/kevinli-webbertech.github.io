## Compiling the Model

Compiling a Keras model is a crucial step in preparing it for training. During the compilation step, you specify the loss function, optimizer, and metrics to be used during training and evaluation.

### Steps to Compile a Model

## Loss

The loss function measures the error or deviation between the predicted values and the actual values during the learning process. It is a critical component required during model compilation in Keras. The choice of loss function depends on the type of problem being solved.

Keras provides a variety of loss functions in the `losses` module. Here are some of the commonly used loss functions:

### Available Loss Functions

- **`mean_squared_error`**: Computes the mean squared error between predictions and actual values. Suitable for regression tasks.
  
- **`mean_absolute_error`**: Computes the mean absolute error between predictions and actual values. Suitable for regression tasks.
  
- **`mean_absolute_percentage_error`**: Computes the mean absolute percentage error between predictions and actual values. Useful for regression with percentage errors.
  
- **`mean_squared_logarithmic_error`**: Computes the mean squared logarithmic error between predictions and actual values. Suitable for regression tasks with logarithmic scaling.
  
- **`squared_hinge`**: Computes the squared hinge loss. Used for classification tasks with a hinge loss formulation.
  
- **`hinge`**: Computes the hinge loss. Commonly used for binary classification tasks.
  
- **`categorical_hinge`**: Computes the categorical hinge loss. Used for multi-class classification tasks.
  
- **`logcosh`**: Computes the logarithm of the hyperbolic cosine of the prediction error. Suitable for regression tasks.
  
- **`huber_loss`**: Computes the Huber loss. Useful for regression tasks that are robust to outliers.
  
- **`categorical_crossentropy`**: Computes the categorical crossentropy loss. Used for multi-class classification tasks with one-hot encoded labels.
  
- **`sparse_categorical_crossentropy`**: Computes the sparse categorical crossentropy loss. Used for multi-class classification tasks with integer labels.
  
- **`binary_crossentropy`**: Computes the binary crossentropy loss. Used for binary classification tasks.
  
- **`kullback_leibler_divergence`**: Computes the Kullback-Leibler divergence between two probability distributions. Used for measuring divergence between distributions.
  
- **`poisson`**: Computes the Poisson loss. Used for count-based data or Poisson regression.
  
- **`cosine_proximity`**: Computes the cosine proximity loss. Measures the cosine similarity between predictions and actual values.

## Loss Functions in Keras

All loss functions in Keras accept two primary arguments:

- **`y_true`**: The true labels, provided as tensors.
- **`y_pred`**: The predictions made by the model, which should have the same shape as `y_true`.

These arguments are used to compute the error or deviation between the true labels and the predictions.

### Importing the Losses Module

Before using any loss function, you need to import the `losses` module from Keras. Here’s how you can do it:

```python
from keras import losses


## Optimizers

In machine learning, optimization is a crucial process that adjusts the model's weights to minimize the loss function and improve predictions. Optimizers play a key role in this process by determining how the weights should be updated based on the gradients of the loss function.

Keras provides a variety of optimizers through its `optimizers` module. Each optimizer has different properties and is suitable for various types of problems.
```
### Available Optimizers in Keras
### SGD (Stochastic Gradient Descent)

The SGD optimizer performs stochastic gradient descent with optional momentum.

```python
keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
```
### RMSprop (Root Mean Square Propagation)

The RMSprop optimizer adjusts the learning rate based on a moving average of the squared gradients.

```python
keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```
### Adagrad (Adaptive Gradient Algorithm)

The Adagrad optimizer adapts the learning rate based on the historical gradient values.

```python
keras.optimizers.Adagrad(learning_rate=0.01)
```
### Adadelta

The Adadelta optimizer is an extension of Adagrad that adapts learning rates based on a moving window of gradient updates.

```python
keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```
### Adam

The Adam optimizer combines the benefits of RMSprop and momentum-based optimization.

```python
keras.optimizers.Adam(
    learning_rate=0.001,     # Learning rate for the optimizer
    beta_1=0.9,             # Exponential decay rate for the first moment estimates
    beta_2=0.999,           # Exponential decay rate for the second moment estimates
    amsgrad=False           # Whether to apply AMSGrad variant
)
```
### Adamax

Adamax is a variant of the Adam optimizer, which uses the infinity norm to stabilize updates.

```python
keras.optimizers.Adamax(
    learning_rate=0.002,     # Learning rate for the optimizer
    beta_1=0.9,             # Exponential decay rate for the first moment estimates
    beta_2=0.999            # Exponential decay rate for the second moment estimates
)
```
### Nadam (Nesterov Adam)

The Nadam optimizer combines the ideas of Nesterov accelerated gradients with the Adam optimizer.

```python
keras.optimizers.Nadam(
    learning_rate=0.002,     # Learning rate for the optimizer
    beta_1=0.9,             # Exponential decay rate for the first moment estimates
    beta_2=0.999            # Exponential decay rate for the second moment estimates
)

```

### Importing the Optimizers Module

Before using any optimizers, you need to import the `optimizers` module as shown below:

```python
from keras import optimizers

```
### Metrics

In machine learning, metrics are used to evaluate the performance of your model. They are similar to loss functions but are not used in the training process. Instead, metrics are used to monitor and assess the performance of the model during and after training.

Keras provides several metrics through the `metrics` module. Some commonly used metrics include:

- **accuracy**: Measures the proportion of correctly predicted instances.
- **binary_accuracy**: Measures the accuracy of binary classification models.
- **categorical_accuracy**: Measures the accuracy of multi-class classification models.
- **sparse_categorical_accuracy**: Measures the accuracy for multi-class classification problems where labels are integers.
- **top_k_categorical_accuracy**: Measures the accuracy where the model's prediction is considered correct if the true label is among the top k predicted labels.
- **sparse_top_k_categorical_accuracy**: Similar to `top_k_categorical_accuracy`, but for sparse labels.
- **cosine_proximity**: Measures the cosine similarity between predicted and true labels.
- **clone_metric**: A placeholder metric used for cloning purposes.

You can import these metrics from the Keras library as follows:

```python
from keras import metrics
```
Similar to loss functions, metrics also accept the following two arguments:

- **y_true**: The true labels as tensors.
- **y_pred**: The predictions with the same shape as `y_true`.


### Import the Metrics Module

Import the metrics module before using metrics as specified below:

```python
from keras import metrics

```
### Compile the Model

Keras models provide a method, `compile()`, to compile the model. The arguments and their default values for the `compile()` method are as follows:

```python
compile(
   optimizer, 
   loss = None, 
   metrics = None, 
   loss_weights = None, 
   sample_weight_mode = None, 
   weighted_metrics = None, 
   target_tensors = None
)
```
### Important Arguments

The important arguments for the `compile()` method are as follows:

- **loss**: The loss function to minimize during training.
- **optimizer**: The optimizer to use for training the model.
- **metrics**: List of metrics to monitor during training and evaluation.
  
### Sample Code to Compile the Model

A sample code to compile the model is as follows:

```python
from keras import losses
from keras import optimizers
from keras import metrics

model.compile(
    loss='mean_squared_error',
    optimizer='sgd',
    metrics=[metrics.categorical_accuracy]
)
```
where,

-loss function is set as mean_squared_error

-optimizer is set as sgd

-metrics is set as metrics.categorical_accuracy

### Model Training

Models are trained using NumPy arrays with the `fit()` method. The primary purpose of this `fit()` function is to train your model and evaluate it on the provided data. It can also be used for graphing model performance. The syntax for the `fit()` method is as follows:

```python
model.fit(X, y, epochs=, batch_size=)
```
Here,

- **X, y**: Tuple to evaluate your data.
- **epochs**: Number of times the model is needed to be evaluated during training.
- **batch_size**: Training instances.

Let us take a simple example of numpy random data to use this concept.


### Create Data

Let us create random data using NumPy for `x` and `y` with the help of the following commands:

```python
import numpy as np 

x_train = np.random.random((100, 4, 8)) 
y_train = np.random.random((100, 10))

Now, create random validation data:

```python
x_val = np.random.random((100, 4, 8)) 
y_val = np.random.random((100, 10))
```
### Create Model

Let us create a simple sequential model:

```python
from keras.models import Sequential
model = Sequential()
```
### Add Layers

Create layers to add to the model:

```python
from keras.layers import LSTM, Dense 

# Add a sequence of vectors of dimension 16 
model.add(LSTM(16, return_sequences=True)) 
model.add(Dense(10, activation='softmax'))
```
### Compile Model

Now the model is defined. You can compile it using the following command:

```python
model.compile(
   loss='categorical_crossentropy', 
   optimizer='sgd', 
   metrics=['accuracy']
)
```
### Apply fit()

Now we apply the `fit()` function to train our data:

```python
model.fit(
   x_train, 
   y_train, 
   batch_size=32, 
   epochs=5, 
   validation_data=(x_val, y_val)
)

```
### Create a Multi-Layer Perceptron ANN

We have learned to create, compile, and train Keras models. Let us apply our learning and create a simple MLP-based ANN.


### Dataset Module

Before creating a model, we need to choose a problem, collect the required data, and convert the data to a NumPy array. Once data is collected, we can prepare the model and train it using the collected data. Data collection is one of the most challenging phases of machine learning. Keras provides a special module, `datasets`, to download online machine learning data for training purposes. It fetches the data from an online server, processes it, and returns the data as training and test sets. The data available in the Keras dataset module includes:

- CIFAR10: Small image classification
- CIFAR100: Small image classification
- IMDB: Movie reviews sentiment classification
- Reuters: Newswire topics classification
- MNIST: Database of handwritten digits
- Fashion-MNIST: Database of fashion articles
- Boston: Housing price regression dataset

### MNIST Database

-Let us use the MNIST database of handwritten digits as our input. MNIST is a collection of 60,000 grayscale images, each with a size of 28x28 pixels, representing digits from 0 to 9. The dataset also includes 10,000 test images.


### Loading the MNIST Dataset

Below code can be used to load the MNIST dataset:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

where

- Line 1 imports MNIST from the Keras dataset module.

- Line 3 calls the `load_data` function, which fetches the data from the online server and returns it as two tuples:
  - The first tuple, `(x_train, y_train)`, represents the training data with shape `(number_samples, 28, 28)` and its digit labels with shape `(number_samples, )`.
  - The second tuple, `(x_test, y_test)`, represents the test data with the same shape.

Other datasets can also be fetched using similar APIs, and each API returns similar data, except for the shape, which depends on the type of data.


```
## Create a Model

Let us choose a simple multi-layer perceptron (MLP) and create the model using Keras.

### Step 1 − Import the Modules

Import the necessary modules:

```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
```
### Step 2 − Load Data

Let us import the MNIST dataset:

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
### Step 3 − Process the Data

Let us change the dataset according to our model, so that it can be fed into our model:

```python
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

Where:

- `reshape` is used to reshape the input from `(28, 28)` tuple to `(784,)`
- `to_categorical` is used to convert vector to binary matrix

```
### Step 4 − Create the model

Let us create the actual model.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
```
### Step 5 − Compile the model

Let us compile the model using the selected loss function, optimizer, and metrics.

```python
from keras.optimizers import RMSprop

model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)
```
### Step 6 − Train the model

Let us train the model using the `fit()` method.

```python
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    verbose=1,
    validation_data=(x_test, y_test)
)
```
### Final thoughts

We have created the model, loaded the data, and trained the model. We still need to evaluate the model and predict output for unknown inputs, which we will cover in the upcoming chapter.

```python
import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.optimizers import RMSprop 
import numpy as np 
```
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# Preprocess the data
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784) 
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10) 
y_test = keras.utils.to_categorical(y_test, 10) 

# Create the model
model = Sequential() 
model.add(Dense(512, activation='relu', input_shape=(784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(), 
              metrics=['accuracy']) 

# Train the model
history = model.fit(x_train, y_train, 
                    batch_size=128, 
                    epochs=20, 
                    verbose=1, 
                    validation_data=(x_test, y_test))

## Model Training Output

Executing the application will give the following output:



Train on 60000 samples, validate on 10000 samples Epoch 1/20 
60000/60000 [==============================] - 7s 118us/step - loss: 0.2453 
- acc: 0.9236 - val_loss: 0.1004 - val_acc: 0.9675 Epoch 2/20 
60000/60000 [==============================] - 7s 110us/step - loss: 0.1023 
- acc: 0.9693 - val_loss: 0.0797 - val_acc: 0.9761 Epoch 3/20 
60000/60000 [==============================] - 7s 110us/step - loss: 0.0744 
- acc: 0.9770 - val_loss: 0.0727 - val_acc: 0.9791 Epoch 4/20 
60000/60000 [==============================] - 7s 110us/step - loss: 0.0599 
- acc: 0.9823 - val_loss: 0.0704 - val_acc: 0.9801 Epoch 5/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0504 
- acc: 0.9853 - val_loss: 0.0714 - val_acc: 0.9817 Epoch 6/20 
60000/60000 [==============================] - 7s 111us/step - loss: 0.0438 
- acc: 0.9868 - val_loss: 0.0845 - val_acc: 0.9809 Epoch 7/20 
60000/60000 [==============================] - 7s 114us/step - loss: 0.0391 
- acc: 0.9887 - val_loss: 0.0823 - val_acc: 0.9802 Epoch 8/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0364 
- acc: 0.9892 - val_loss: 0.0818 - val_acc: 0.9830 Epoch 9/20 
60000/60000 [==============================] - 7s 113us/step - loss: 0.0308 
- acc: 0.9905 - val_loss: 0.0833 - val_acc: 0.9829 Epoch 10/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0289 
- acc: 0.9917 - val_loss: 0.0947 - val_acc: 0.9815 Epoch 11/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0279 
- acc: 0.9921 - val_loss: 0.0818 - val_acc: 0.9831 Epoch 12/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0260 
- acc: 0.9927 - val_loss: 0.0945 - val_acc: 0.9819 Epoch 13/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0257 
- acc: 0.9931 - val_loss: 0.0952 - val_acc: 0.9836 Epoch 14/20
60000/60000 [==============================] - 7s 112us/step - loss: 0.0229 
- acc: 0.9937 - val_loss: 0.0924 - val_acc: 0.9832 Epoch 15/20 
60000/60000 [==============================] - 7s 115us/step - loss: 0.0235 
- acc: 0.9937 - val_loss: 0.1004 - val_acc: 0.9823 Epoch 16/20 
60000/60000 [==============================] - 7s 113us/step - loss: 0.0214 
- acc: 0.9941 - val_loss: 0.0991 - val_acc: 0.9847 Epoch 17/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0219 
- acc: 0.9943 - val_loss: 0.1044 - val_acc: 0.9837 Epoch 18/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0190 
- acc: 0.9952 - val_loss: 0.1129 - val_acc: 0.9836 Epoch 19/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0197 
- acc: 0.9953 - val_loss: 0.0981 - val_acc: 0.9841 Epoch 20/20 
60000/60000 [==============================] - 7s 112us/step - loss: 0.0198 
- acc: 0.9950 - val_loss: 0.1215 - val_acc: 0.9828

