#pragma once
#include "../Matrix/Matrix.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

using type = Matrix<float>(*)(const Matrix<float>& i);

class NNet{
    public:
    NNet(std::vector<int>, std::string, std::string);
    ~NNet();
    Matrix<float> predict(const Matrix<float>&);
    void train(const Matrix<float>&, const Matrix<float>&, float, float, int);
    void setWeightsAndBias();
    void writeWeightsAndBias(std::string, std::string);
    void readWeightsAndBias(std::string, std::string);
    type userActivation;
    type userActivationPrime;
    bool applySoftMax;
    bool outputLoss;
    bool doOnLineGradientDescent;
    std::size_t miniBatchSize;
    private:
    std::vector<int> layers;
    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> biases;
    std::vector<Matrix<float>> backprop(const std::vector<Matrix<float>>&, const std::vector<Matrix<float>>&, const Matrix<float>&);
    Matrix<float> df(const Matrix<float>&, const Matrix<float>&);
    Matrix<float> activation(const Matrix<float>&, std::size_t);
    Matrix<float> tanh(const Matrix<float>&);
    Matrix<float> activationPrime(const Matrix<float>&, std::size_t);
    Matrix<float> tanhPrime(const Matrix<float>&);
    Matrix<float> softMax(const Matrix<float>&);
    float loss(const Matrix<float>&, const Matrix<float>&, float);
    void gradientDescent(const Matrix<float>&, const Matrix<float>&, float, float, int, float&);
    int nextPrime(int);
    bool isPrime(int);
};

NNet::NNet(std::vector<int> layers, std::string wfile = "", std::string bfile = "") : userActivation(nullptr), userActivationPrime(nullptr), applySoftMax(false), outputLoss(false), doOnLineGradientDescent(false), miniBatchSize(1), layers(layers){
    // initialize weights and biases
    for(std::size_t i = 0; i < layers.size()-1; i++){
        weights.push_back(Matrix<float>(layers[i+1], layers[i]));
        biases.push_back(Matrix<float>(layers[i+1], 1));
    }

    if(wfile != "" || bfile != "") // read weights and biases from file
        readWeightsAndBias(wfile, bfile);
    else                           // set weights and biases to random values near 0
        setWeightsAndBias();
}

NNet::~NNet(){}

void NNet::setWeightsAndBias(){
    srand(0);
    for(std::size_t i = 0; i < layers.size()-1; i++){
        for(int j = 0; j < layers[i+1]; j++){
            for(int k = 0; k < layers[i]; k++){
                weights[i].at(j,k) = (rand() % 1000 - 500) / 1000.0;
            }
            biases[i].at(i,0) = (rand() % 1000 - 500) / 1000.0;
        }
    }
}

void NNet::writeWeightsAndBias(std::string wfile, std::string bfile){
    for(std::size_t i = 0; i < layers.size()-1; i++){
        std::ofstream w(wfile + "_layer" + std::to_string(i) + ".txt");
        std::ofstream b(bfile + "_layer" + std::to_string(i) + ".txt");
        if(!w.is_open() || !b.is_open()){
            std::cerr << "Error opening weights and biases file... Not writing to file\n";
            return;
        }
        for(int j = 0; j < layers[i+1]; j++){
            for(int k = 0; k < layers[i]; k++){
                w << weights[i].at(j,k) << ", ";
            }
            w << '\n';
            b << biases[i].at(j,0) << ", ";
        }
        b << '\n';
    }
}

void NNet::readWeightsAndBias(std::string wfile, std::string bfile){
    std::ifstream w(wfile);
    std::ifstream b(bfile);
    if (!w.is_open() || !b.is_open()){
        std::cerr << "Error opening weights and biases file... Using default random initial weights and biases\n";
        setWeightsAndBias();
        return;
    }
    for(std::size_t i = 0; i < layers.size()-1; i++){
        for(int j = 0; j < layers[i+1]; j++){
            for(int k = 0; k < layers[i]; k++){
                w >> weights[i].at(j,k);
            }
            b >> biases[i].at(j,0);
        }
    }
}

float NNet::loss(const Matrix<float>& output, const Matrix<float>& target, float lambda){
    Matrix<float> t = df(output, target);
    return (t*t).sum();
}

Matrix<float> NNet::df(const Matrix<float>& output, const Matrix<float>& target){
    return output - target;
}

Matrix<float> NNet::activation(const Matrix<float>& A, std::size_t p){
    if(applySoftMax && p == layers.size()-2)
        return softMax(A);
    else if(userActivation)
        return userActivation(A);
    return tanh(A);
}
Matrix<float> NNet::activationPrime(const Matrix<float>& A, std::size_t p){
    if(userActivationPrime)
        return userActivationPrime(A);
    return tanhPrime(A);
}

Matrix<float> NNet::tanh(const Matrix<float>& A){
    return A.tanh();
}
Matrix<float> NNet::tanhPrime(const Matrix<float>& A){
    // 1-tanh^2
    Matrix<float> t = A.tanh();
    return 1 - t*t;
}

Matrix<float> NNet::softMax(const Matrix<float>& A){
    Matrix<float> ex = (A-A.max(1)).exp();
    return ex/ex.sum(1);
}

std::vector<Matrix<float>> NNet::backprop(const std::vector<Matrix<float>>& Zs, const std::vector<Matrix<float>>& As, const Matrix<float>& target){
    std::vector<Matrix<float>> deltas;
    deltas.push_back(df(As.back(), target));
    for(int i = layers.size()-3; i >= 0; i--){
        deltas.push_back(activationPrime(Zs[i], i) * weights[i+1].T().dot(deltas.back()));
    }
    return deltas;
}

void NNet::gradientDescent(const Matrix<float>& input, const Matrix<float>& target, float lambda, float error_threshold, int max_iter, float& Eg2){
    std::size_t m = input.shape().second;
    // forward propagate
    std::vector<Matrix<float>> Zs; // pre activations
    std::vector<Matrix<float>> As; // activations
    As.push_back(input);
    for(std::size_t k = 0; k < layers.size()-1; k++){
        Zs.push_back(weights[k].dot(As[k]) + biases[k]);
        As.push_back(activation(Zs[k], k));
    }
    // calculate gradients G
    std::vector<Matrix<float>> deltas = backprop(Zs, As, target);
    std::vector<Matrix<float>> G;
    std::vector<float> B;
    for(std::size_t k = 0; k < layers.size()-1; k++){
        auto temp = deltas[deltas.size()-k-1];
        auto temp1 = As[k].T();
        auto temp2 = temp.dot(temp1);
        G.push_back((deltas[deltas.size()-k-1].dot(As[k].T()) + weights[k]*2*lambda) / m);
        B.push_back(deltas[deltas.size()-k-1].sum() / m);
    }

    float sumofG2 = 0;
    for(std::size_t k = 0; k < layers.size()-1; k++){
        sumofG2 += (G[k]*G[k]).sum();
    }
    Eg2 = 0.9*Eg2 + 0.1*sumofG2;
    float eta = 0.01/sqrt(1e-10+Eg2);

    for(std::size_t k = 0; k < weights.size(); k++){
        weights[k] = weights[k] - G[k]*eta;
        biases[k] = biases[k] - B[k]*eta;
    }
}

/*
    Updates the weights and biases with training data input, target
    The regularization strength lambda penalizes the weights that stray too far away from zero - this can help prevent over fitting
    error_threshold is the gradient descent stopping condition, if the loss does not change more than this value it stops
    gradient descent will stop after max_iter iterations
    
    The learning rate, eta, is set dynamically using RMSprop see: https://www.ruder.io/optimizing-gradient-descent/#gradientdescentoptimizationalgorithms

    The option to do online gradient descent, stochastic gradient descent, is set with the boolean doOnLineGradientDescent member
    Additionally, mini batch size can also be set for the on line option with the miniBatchSize member

    SoftMax can be applied to the output layer with the applySoftMax member

    User defined activation functions can be used by defining the function and using the userActivation and userActivationPrime members
    The function should return Matrix<float> with const Matrix<float>& parameter
    The derivative of the activation function must be defined with userActivationPrime
*/
void NNet::train(const Matrix<float>& input, const Matrix<float>& target, float lambda=0, float error_threshold=1e-6, int max_iter=50000){
    float Eg2 = 1;
    float delloss = 1;
    float oldloss = std::numeric_limits<float>::max();
    std::size_t m = input.shape().second;
    int P = 0;

    std::vector<Matrix<float>> Xbatches;
    std::vector<Matrix<float>> Ybatches;

    if(doOnLineGradientDescent){
        for(std::size_t ol = 0; ol < m; ol+=miniBatchSize){
            Xbatches.push_back(input.cRange(ol, std::min(m, ol+miniBatchSize)));
            Ybatches.push_back(target.cRange(ol, std::min(m, ol+miniBatchSize)));
        }
        P = Xbatches.size();
    }

    for(int i = 0; i < max_iter && delloss >= error_threshold; i++){
        for(int t = 0; t < 10; t++){
            if(doOnLineGradientDescent){
                P = nextPrime(P); // pseudo random shuffle of possible positions in the array
                int pos = P % Xbatches.size();
                for(std::size_t ol = 0; ol < Xbatches.size(); ol++){
                    gradientDescent(Xbatches[pos], Ybatches[pos], lambda, error_threshold, max_iter, Eg2);
                    pos = (pos + P) % Xbatches.size();
                }
            } else
                gradientDescent(input, target, lambda, error_threshold, max_iter, Eg2);
        }
        Matrix<float> output = predict(input);
        float closs = loss(output, target, lambda);
        delloss = std::abs(oldloss - closs);
        oldloss = closs;
        if(outputLoss)
            std::cout << "loss: " << closs << std::endl;
    }
}

int NNet::nextPrime(int n){
    n++;
    if(n % 2 == 0)
        n++;
    while(!isPrime(n))
        n += 2;
    return n;
}

bool NNet::isPrime(int n){
    for(int i = 2; i * i <= n; i++){
        if(n%i == 0){
            return false;
        }
    }
    return true;
}

Matrix<float> NNet::predict(const Matrix<float>& input){
    Matrix<float> output;
    output = input;
    for(std::size_t i = 0; i < layers.size()-1; i++){
        output = activation(weights[i].dot(output) + biases[i], i);
    }
    return output;
}