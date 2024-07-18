ResNet is a pre-trained model. It is trained using ImageNet. ResNet model weights pre-trained on ImageNet. It has the following syntax −

keras.applications.resnet.ResNet50 (
   include_top = True, 
   weights = 'imagenet', 
   input_tensor = None, 
   input_shape = None, 
   pooling = None, 
   classes = 1000
)

Here,

include_top refers the fully-connected layer at the top of the network.

weights refer pre-training on ImageNet.

input_tensor refers optional Keras tensor to use as image input for the model.

input_shape refers optional shape tuple. The default input size for this model is 224x224.

classes refer optional number of classes to classify images.

Step 1: import the modules

>>> import PIL 
>>> from keras.preprocessing.image import load_img 
>>> from keras.preprocessing.image import img_to_array 
>>> from keras.applications.imagenet_utils import decode_predictions 
>>> import matplotlib.pyplot as plt 
>>> import numpy as np 
>>> from keras.applications.resnet50 import ResNet50 
>>> from keras.applications import resnet50

Step 2: Select an input

>>> filename = 'banana.jpg' 
>>> ## load an image in PIL format 
>>> original = load_img(filename, target_size = (224, 224)) 
>>> print('PIL image size',original.size)
PIL image size (224, 224) 
>>> plt.imshow(original) 
<matplotlib.image.AxesImage object at 0x1304756d8> 
>>> plt.show()

Step 3: Convert images into NumPy array

>>> #convert the PIL image to a numpy array 
>>> numpy_image = img_to_array(original) 

>>> plt.imshow(np.uint8(numpy_image)) 
<matplotlib.image.AxesImage object at 0x130475ac8> 

>>> print('numpy array size',numpy_image.shape) 
numpy array size (224, 224, 3) 

>>> # Convert the image / images into batch format 
>>> image_batch = np.expand_dims(numpy_image, axis = 0) 

>>> print('image batch size', image_batch.shape) 
image batch size (1, 224, 224, 3)
>>> 

Step 4: Model prediction

>>> prepare the image for the resnet50 model >>> 
>>> processed_image = resnet50.preprocess_input(image_batch.copy()) 

>>> # create resnet model 
>>>resnet_model = resnet50.ResNet50(weights = 'imagenet') 
>>> Downloavding data from https://github.com/fchollet/deep-learning-models/releas
es/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5 
102858752/102853048 [==============================] - 33s 0us/step 

>>> # get the predicted probabilities for each class 
>>> predictions = resnet_model.predict(processed_image) 

>>> # convert the probabilities to class labels 
>>> label = decode_predictions(predictions) 
Downloading data from https://storage.googleapis.com/download.tensorflow.org/
data/imagenet_class_index.json 
40960/35363 [==================================] - 0s 0us/step 

>>> print(label)

Output

[
   [
      ('n07753592', 'banana', 0.99229723), 
      ('n03532672', 'hook', 0.0014551596), 
      ('n03970156', 'plunger', 0.0010738898), 
      ('n07753113', 'fig', 0.0009359837) , 
      ('n03109150', 'corkscrew', 0.00028538404)
   ]
]


Here, the model predicted the images as banana correctly.