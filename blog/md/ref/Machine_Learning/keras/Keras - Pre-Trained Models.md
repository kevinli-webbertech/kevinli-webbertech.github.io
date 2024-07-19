Keras - Pre-Trained Models

VGG16

VGG16 is another pre-trained model. It is also trained using ImageNet.

syntax

keras.applications.vgg16.VGG16(
   include_top = True, 
   weights = 'imagenet', 
   input_tensor = None, 
   input_shape = None, 
   pooling = None, 
   classes = 1000
)

The default input size for this model is 224x224.

MobileNetV2

MobileNetV2 is another pre-trained model. It is also trained uing ImageNet.

syntax

keras.applications.mobilenet_v2.MobileNetV2 (
   input_shape = None, 
   alpha = 1.0, 
   include_top = True, 
   weights = 'imagenet', 
   input_tensor = None, 
   pooling = None, 
   classes = 1000
)

alpha controls the width of the network. If the value is below 1, decreases the number of filters in each layer. If the value is above 1, increases the number of filters in each layer. If alpha = 1, default number of filters from the paper are used at each layer.

The default input size for this model is 224x224.

InceptionResNetV2

 It is also trained using ImageNet.

keras.applications.inception_resnet_v2.InceptionResNetV2 (
   include_top = True, 
   weights = 'imagenet',
   input_tensor = None, 
   input_shape = None, 
   pooling = None, 
   classes = 1000)

This model and can be built both with ‘channels_first’ data format (channels, height, width) or ‘channels_last’ data format (height, width, channels).

The default input size for this model is 299x299.

InceptionV3

 It is also trained uing ImageNet

keras.applications.inception_v3.InceptionV3 (
   include_top = True, 
   weights = 'imagenet', 
   input_tensor = None, 
   input_shape = None, 
   pooling = None, 
   classes = 1000
)


The default input size for this model is 299x299.


Conclusion
Keras is very simple, extensible and easy to implement neural network API
Keras is an optimal choice for deep leaning models.


