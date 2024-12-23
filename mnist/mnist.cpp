#include <iostream>
#include <vector>
#include <fstream>
#include "../NNet.h"

const std::string trainPath = "mnist_train.csv";
const std::string testPath = "mnist_test.csv";
const int trainSize = 60000;
const int testSize = 10000;
const int batches = 1;

std::vector<std::string> split(std::string in, char del){
    std::vector<std::string> out;
    std::string temp = "";
    for(std::size_t i = 0; i < in.size(); i++){
        if(in[i] == del){
            out.push_back(temp);
            temp = "";
        }else{
            temp += in[i];
        }
    }
    if(temp != "")
        out.push_back(temp);
    return out;
}

std::vector<float> yval(std::string in){
    std::vector<float> p(10);
    p[std::stoi(in)] = 1;
    return p;
}

void update(Matrix<float>& X, Matrix<float>& Y, std::string line, int p){
    std::vector<std::string> parts = split(line, ',');
    for(std::size_t i = 1; i < parts.size(); i++)
        X.at(p, i-1) = std::stof(parts[i]) / 255;

    std::vector<float> Yparts = yval(parts[0]);
    for(int i = 0; i < 10; i++)
        Y.at(p, i) = Yparts[i];
}

Matrix<float> max(Matrix<float>& m){
    Matrix<float> d(m.shape().first, m.shape().second);
    for(std::size_t i = 0; i < m.shape().first; i++){
        int p = 0;
        for(std::size_t j = 0; j < m.shape().second; j++){
            if(m.at(i, j) > m.at(i, p))
                p = j;
        }
        d.at(i, p) = 1;
    }
    return d;
}

// custom function must return Matrix<float> with argument const Matrix<float>&
Matrix<float> ReLU(const Matrix<float>& m){
    Matrix<float> t = m;
    for(std::size_t i = 0; i < m.shape().first; i++){
        for(std::size_t j = 0 ; j < m.shape().second; j++){
            if(m.at(i,j) < 0)
                t.at(i,j) = 0.01 * m.at(i,j);
                // t.at(i,j) = 0;
            else
                t.at(i,j) = m.at(i,j);
        }
    }
    return t;
}

Matrix<float> ReLUPrime(const Matrix<float>& m){
    Matrix<float> t = m;
    for(std::size_t i = 0; i < m.shape().first; i++){
        for(std::size_t j = 0; j < m.shape().second; j++){
            if(m.at(i,j) < 0)
                t.at(i,j) = 0.01;
                // t.at(i,j) = 0;
            else
                t.at(i,j) = 1;
        }
    }
    return t;
}

int main(){
    // read mnist training data
    std::ifstream file(trainPath);

    std::vector<Matrix<float>> XtrainBatches(batches, Matrix<float>(trainSize/batches, 784, -1));
    std::vector<Matrix<float>> YtrainBatches(batches, Matrix<float>(trainSize/batches, 10));
    Matrix<float> Xtest(testSize, 784);
    Matrix<float> Ytest(testSize, 10);

    std::cout << "getting data..." << std::endl;

    if(!file.is_open())
        file.open("mnist/"+trainPath);

    // parse input and insert into matrices for training data
    std::string line;
    int batch = 0;
    for(int p = 0; p < trainSize; p++){
        std::getline(file, line);
        if(p%(trainSize/batches)==0 && p)
            batch++;
        update(XtrainBatches[batch], YtrainBatches[batch], line, p%(trainSize/batches));
    }

    // shape is 784xM
    for(std::size_t i = 0; i < batches; i++){
        XtrainBatches[i] = XtrainBatches[i].T();
        YtrainBatches[i] = YtrainBatches[i].T();
    }

    file.close();
    file.open(testPath);

    if(!file.is_open())
        file.open("mnist/"+testPath);

    // parse and insert into matrices
    for(int p = 0; p < testSize; p++){
        std::getline(file, line);
        update(Xtest, Ytest, line, p);
    }
    file.close();
    
    Xtest = Xtest.T();
    Ytest = Ytest;

    std::vector<int> layers = {784, 16, 16, 10};
    NNet net(layers);
    net.applySoftMax = true; // use softmax on the final layer
    net.doOnLineGradientDescent = true; // use stochastic gradient descent
    net.outputLoss = true; // print out the current loss while training
    net.userActivation = ReLU; // use ReLU activation function
    net.userActivationPrime = ReLUPrime; // must define the derivative of the activation
    net.miniBatchSize = 20; // size of mini batches(only used when doOnlineGradientDescent is true)

    std::cout << "training..." << std::endl;
    
    for(int i = 0; i < batches; i++){
        std::cout << "Batch " << i << std::endl;
        net.train(XtrainBatches[i], YtrainBatches[i], 0.001, 1, 1000);
    }
    
    Matrix<float> Ypred = net.predict(Xtest).argMax(1);
    
    std::cout << "testing..." << std::endl;

    float correct = 0.0;
    for(std::size_t i = 0; i < Ypred.shape().second; i++){
        if(Ytest.at(i, Ypred.at(0, i)) == 1)
            correct++;
    }
    std::cout << "accuracy: " << correct/testSize << std::endl;

    std::vector<int> layers = {784, 16, 16, 10};
    NNet net(layers);

    net.readWeightsAndBias("weights.txt", "bias.txt");
    net.writeWeightsAndBias("weights", "bias");
}
