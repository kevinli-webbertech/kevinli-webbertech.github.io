# Keras Backend Implementations: TensorFlow and Theano

## TensorFlow

### Installation
```bash
pip install tensorflow

```
# Keras Backend Configuration

Once you execute Keras, you will find the configuration file located in your home directory under `.keras/keras.json`.

The configuration file can be found at:

keras.json

{ 
   "image_data_format": "channels_last", 
   "epsilon": 1e-07, 
   "floatx": "float32", 
   "backend": "tensorflow" 
}


# Keras Configuration Details

## Configuration Parameters

- **`image_data_format`**: Represents the data format. For example, `"channels_last"` or `"channels_first"`.

- **`epsilon`**: Represents a numeric constant used to avoid divide-by-zero errors. The default value is `1e-07`.

- **`floatx`**: Represents the default data type, which is `float32`. You can also change it to `float16` or `float64` using the `set_floatx()` method.

## Creating the Configuration File

If the `keras.json` file is not created automatically, you can create it manually by following these steps:

```bash
cd ~
mkdir .keras
vi .keras/keras.json
```

# Keras Configuration

## Important Notes

- **Folder Name**: Make sure to specify `.keras` as the folder name.

- **Configuration File**: Add the configuration inside the `keras.json` file.

##Theano

# Setting Up Theano Backend

## Installation

To install Theano, use the following command:

```bash
pip install theano

{ 
   "image_data_format": "channels_last", 
   "epsilon": 1e-07, 
   "floatx": "float32", 
   "backend": "theano" 
}

1. Save the `keras.json` file.
2. Restart your terminal.
3. Start Keras.

import keras as k
# using theano backend.

