PyTorch includes following dataset loaders −

MNIST
COCO (Captioning and Detection)

Dataset includes majority of two types of functions given below −

Transform − a function that takes in an image and returns a modified version of standard stuff. These can be composed together with transforms.

Target_transform − a function that takes the target and transforms it. For example, takes in the caption string and returns a tensor of world indices.


MNIST
The following is the sample code for MNIST dataset −

dset.MNIST(root, train = TRUE, transform = NONE, 
target_transform = None, download = FALSE)
The parameters are as follows −

root − root directory of the dataset where processed data exist.

train − True = Training set, False = Test set

download − True = downloads the dataset from the internet and puts it in the root.

COCO
This requires the COCO API to be installed. The following example is used to demonstrate the COCO implementation of dataset using PyTorch −

import torchvision.dataset as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = ‘ dir where images are’, 
annFile = ’json annotation file’,
transform = transforms.ToTensor())
print(‘Number of samples: ‘, len(cap))
print(target)
The output achieved is as follows −

Number of samples: 82783
Image Size: (3L, 427L, 640L)

