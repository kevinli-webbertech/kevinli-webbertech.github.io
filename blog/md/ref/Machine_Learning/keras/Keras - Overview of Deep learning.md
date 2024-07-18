Deep learning is an evolving subfield of machine learning.
analyzes the input in layer by layer manner.

 Let us go through the basics of deep learning in this chapter.

Artificial Neural Networks

- inspired from the model of human brain
-The human brain is made up of more than 90 billion tiny cells called “Neurons”. Neurons are inter-connected through nerve fiber called “axons” and “Dendrites”. The main role of axon is to transmit information from one neuron to another to which it is connected.
- main role of dendrites is to receive the information being transmitted by the axons of another neuron to which it is connected

Based on this model, the first Artificial Neural Network (ANN) was invented by psychologist Frank Rosenblatt, in the year of 1958.

A single neuron (called as perceptron in ANN) can be represented as below −

need to upload image

Here,

Multiple input along with weight represents dendrites.

Sum of input along with activation function represents neurons. Sum actually means computed value of all inputs and activation function represent a function, which modify the Sum value into 0, 1 or 0 to 1.

Actual output represent axon and the output will be received by neuron in next layer.

Multi-Layer Perceptron

Multi-Layer perceptron is the simplest form of ANN. It consists of a single input layer, one or more hidden layer and finally an output layer. A layer consists of a collection of perceptron. Input layer is basically one or more features of the input data. Every hidden layer consists of one or more neurons and process certain aspect of the feature and send the processed information into the next hidden layer. The output layer process receives the data from last hidden layer and finally output the result.

need to upload image

Convolutional Neural Network (CNN)

-popular ANN widely used in the fields of image and video recognition.
-similar to multi-layer perceptron except it contains series of convolution layer and pooling layer before the fully connected hidden neuron layer
It has three important layers


Convolution layer − It is the primary building block and perform computational tasks based on convolution function.

Pooling layer − It is arranged next to convolution layer and is used to reduce the size of inputs by removing unnecessary information so computation can be performed faster.

Fully connected layer − It is arranged to next to series of convolution and pooling layer and classify input into various categories.

simple CNN

need to upload image

2 series of Convolution and pooling layer is used and it receives and process the input (e.g. image).

A single fully connected layer is used and it is used to output the data (e.g. classification of image)


Recurrent Neural Network (RNN)

 -useful to address the flaw in other ANN models
-RNN stores the past information and all its decisions are taken from what it has learnt from the past.
- mainly useful in image classification
 
Workflow of ANN

Collect required data

Deep learning requires lot of input data to successfully learn and predict the result. So, first collect as much data as possible.

Analyze data

Analyze the data and acquire a good understanding of the data to select correct ANN Algorithm

Choose an algorithm (model)

lgorithm is represented by Model in Keras. Algorithm includes one or more layers. Each layers in ANN can be represented by Keras Layer in Keras.


Prepare data − Process, filter and select only the required information from the data.

Split data − Split the data into training and test data set. Test data will be used to evaluate the prediction of the algorithm / Model (once the machine learn) and to cross check the efficiency of the learning process.

Compile the model − Compile the algorithm / model, so that, it can be used further to learn by training and finally do to prediction. This step requires us to choose loss function and Optimizer. loss function and Optimizer are used in learning phase to find the error (deviation from actual output) and do optimization so that the error will be minimized.

Fit the model − The actual learning process will be done in this phase using the training data set.

Predict result for unknown value − Predict the output for the unknown input data (other than existing training and test data)

Evaluate model − Evaluate the model by predicting the output for test data and cross-comparing the prediction with actual result of the test data.

Freeze, Modify or choose new algorithm − Check whether the evaluation of the model is successful. If yes, save the algorithm for future prediction purpose. If not, then modify or choose new algorithm / model and finally, again train, predict and evaluate the model. Repeat the process until the best algorithm (model) is found.
The above steps can be represented using below flow chart −




