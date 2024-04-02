#pragma once
#include "../Matrix/Matrix.h"
#include <cmath>

#include<vector>

class NNet{
    public:
    NNet(std::vector<int>);
    ~NNet();
    Matrix<float> predict(const Matrix<float>&);
    void train(const Matrix<float>&, const Matrix<float>&, float, float, int);
    std::vector<int> layers;
    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> biases;
    private:
    std::vector<Matrix<float>> backprop(const std::vector<Matrix<float>>&, const std::vector<Matrix<float>>&, const Matrix<float>&);
    Matrix<float> df(const Matrix<float>&, const Matrix<float>&);
    Matrix<float> activation(const Matrix<float>&);
    Matrix<float> activationPrime(const Matrix<float>&);
    float loss(const Matrix<float>&, const Matrix<float>&, float);
};

NNet::NNet(std::vector<int> layers) : layers(layers){
    for(std::size_t i = 0; i < layers.size()-1; i++){
        weights.push_back(Matrix<float>(layers[i+1], layers[i]));
        biases.push_back(Matrix<float>(1, layers[i+1], 1));
    }
}

NNet::~NNet(){
}

float NNet::loss(const Matrix<float>& output, const Matrix<float>& target, float lambda){
    float w2 = 0;
    for(std::size_t i = 0; i < layers.size()-1; i++){
        w2 += (weights[i]*weights[i]).sum();
    }
    float s = 0;
    Matrix<float> t = target+(output*-1);
    s = (t*t).sum();
    return s + lambda*w2;
}

Matrix<float> NNet::df(const Matrix<float>& output, const Matrix<float>& target){
    return 2 * (output + (target*-1));
}

Matrix<float> NNet::activation(const Matrix<float>& A){
    return A.tanh();
}
Matrix<float> NNet::activationPrime(const Matrix<float>& A){
    // 1-tanh^2
    Matrix<float> t = A.tanh();
    return 1 - t*t;
}

std::vector<Matrix<float>> NNet::backprop(const std::vector<Matrix<float>>& Zs, const std::vector<Matrix<float>>& As, const Matrix<float>& target){
    std::vector<Matrix<float>> deltas;
    deltas.push_back(df(Zs.back(), target));
    for(int i = layers.size()-3; i >= 0; i--){
        deltas.push_back(activationPrime(Zs[i]) * deltas.back().dot(weights[i+1]));
    }
    return deltas;
}

void NNet::train(const Matrix<float>& input, const Matrix<float>& target, float lambda=0.1, float error_threshold=1e-6, int max_iter=50000){
    // set weights and biases to random values near 0
    srand(0);
    for(std::size_t i = 0; i < layers.size()-1; i++){
        for(int j = 0; j < layers[i+1]; j++){
            for(int k = 0; k < layers[i]; k++){
                weights[i].at(j,k) = (rand() % 1000 - 500) / 1000.0;
            }
        }
    }

    float Eg2 = 1;
    float delloss = 1;
    float oldloss = INFINITY;

    for(int i = 0; i < max_iter && delloss >= error_threshold; i++){
        for(int t = 0; t < 10; t++){
            // calculate gradients G
            std::vector<Matrix<float>> Zs; // pre activations
            std::vector<Matrix<float>> As; // activations
            As.push_back(input);
            for(std::size_t i = 0; i < layers.size()-1; i++){
                Zs.push_back(As[i].dot(weights[i].T()) + biases[i]);
                As.push_back(activation(Zs[i]));
            }
            std::vector<Matrix<float>> deltas = backprop(Zs, As, target);
            std::vector<Matrix<float>> G;
            std::vector<Matrix<float>> B;
            for(std::size_t i = 0; i < layers.size()-1; i++){
                G.push_back(deltas[deltas.size()-i-1].T().dot(As[i]) + weights[i]*(2*lambda));
                B.push_back(deltas[deltas.size()-i-1].squishM() / deltas[deltas.size()-i-1].shape().first);
            }

            float sumofG2 = 0;
            for(std::size_t i = 0; i < layers.size()-1; i++){
                sumofG2 += (G[i]*G[i]).sum();
            }
            Eg2 = 0.9*Eg2 + 0.1*sumofG2;
            float eta = 0.01/sqrt(1e-10+Eg2);

            for(std::size_t k = 0; k < weights.size(); k++){
                weights[k] = weights[k] - G[k]*eta;
                biases[k] = biases[k] - B[k]*eta;
            }
        }
        Matrix<float> output = predict(input);
        float closs = loss(output, target, lambda);
        delloss = oldloss - closs;
        oldloss = closs;
        std::cout << "loss: " << closs << std::endl;
    }
}

Matrix<float> NNet::predict(const Matrix<float>& input){
    Matrix<float> output;
    output = input;
    for(std::size_t i = 0; i < layers.size()-1; i++){
        output = activation(output.dot(weights[i].T()) + biases[i]);
    }
    return output;
}