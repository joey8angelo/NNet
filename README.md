## NNet
C++ implementation of a simple neural network, or multi-layer perceptron. The NNet class is constructed with a vector of integers, each integer represents how many nodes are in each layer.
Dependant on https://github.com/joey8angelo/Matrix, which could could be swapped with some other linear algebra package. See the Matrix readme for compilation.

### Training
The Neural Net trains the weights and biases with the train function which has parameters:
Matrix<float> input&: This data has shape (NumFeatures x NumExamples)
Matrix<float> target&: This data is shaped (OutputLayerSize x NumExamples)
float lambda: This is the regularization strength, this penalizes the weights when they get far from zero, which helps prevent over fitting
float error_threshold: The gradient descent stopping condition, if the loss does not change more than this value it stops
int max_iter: Gradient descent will stop after max_iter iterations

The learning rate, eta, is dynamically set using RMSprop, see https://www.ruder.io/optimizing-gradient-descent/#gradientdescentoptimizationalgorithms for more information.

The option to do online gradient descent, stochastic gradient descent, is set with the boolean doOnLineGradientDescent member
Additionally, mini batch size can also be set for the on line option with the miniBatchSize member

SoftMax can be applied to the output layer with the boolean applySoftMax member

User defined activation functions can be used by defining the function and using the userActivation and userActivationPrime members
The function should return Matrix<float> with const Matrix<float>& parameter
The derivative of the activation function must be defined with userActivationPrime if the userActivation is set
see mnist/mnist.cpp for an example

### Prediction
Prediction is a simple forward propagation, Matrix<float>& input is shaped (NumFeatures x NumExamples) and returns a Matrix<float> shaped (OutputLayerSize x NumExamples).