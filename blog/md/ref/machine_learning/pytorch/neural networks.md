principle of neural network
includes a collection of basic elements, i.e., artificial neuron or perceptron. It includes several basic inputs such as x1, x2….. xn which produces a binary output if the sum is greater than the activation potential.

schematic representation of sample neuron

Output=∑jwjxj+Bias

The typical neural network architecture is described below −

include diagram
layers between input and output are referred to as hidden layers,
 the density and type of connections between layers is the configuration
example
a fully connected configuration has all the neurons of layer L connected to those of L+1. For a more pronounced localization, we can connect only a local neighbourhood, say nine neurons, to the next layer. Figure 1-9 illustrates two hidden layers with dense connections.

types of neural networks
Feedforward Neural Networks

Feedforward neural networks include basic units of neural network family. The movement of data in this type of neural network is from the input layer to output layer, via present hidden layers. The output of one layer serves as the input layer with restrictions on any kind of loops in the network architecture.
include diagram

Recurrent Neural Networks

used when the data pattern changes consequently over a period.

In RNN, same layer is applied to accept the input parameters and display output parameters in specified neural network.
include diagram

Neural networks can be constructed using the torch.nn package.

With the help of PyTorch, we can use the following steps for typical training procedure for a neural network −

Define the neural network that has some learnable parameters (or weights).
Iterate over a dataset of inputs.
Process input through the network.
Compute the loss (how far is the output from being correct).
Propagate gradients back into the network’s parameters.
Update the weights of the network, typically using a simple update as given below

rule: weight = weight -learning_rate * gradient



