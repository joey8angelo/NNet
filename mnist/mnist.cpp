#include <iostream>
#include <vector>
#include <fstream>
#include "../NNet.h"

const std::string trainPath = "mnist/mnist_train.csv";
const std::string testPath = "mnist/mnist_test.csv";
const int trainSize = 60000;
const int testSize = 10000;

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
    for(std::size_t i = 1; i < parts.size(); i++){
        X.at(p, i-1) = std::stof(parts[i]);
    }
    std::vector<float> Yparts = yval(parts[0]);
    for(int i = 0; i < 10; i++){
        Y.at(p, i) = Yparts[i];
    }
}

Matrix<float> max(Matrix<float>& m){
    Matrix<float> d(m.shape().first, m.shape().second);
    for(std::size_t i = 0; i < m.shape().first; i++){
        int p = 0;
        for(std::size_t j = 0; j < m.shape().second; j++){
            if(m.at(i, j) > m.at(i, p))
                p = j;
        }
        d.at(i,p) = 1;
    }
    return d;
}

int main(){
    // read mnist testing data
    std::ifstream trainFile(trainPath);
    std::ifstream testFile(testPath);

    Matrix<float> Xtrain(trainSize, 784);
    Matrix<float> Ytrain(trainSize, 10);
    Matrix<float> Xtest(testSize, 784);
    Matrix<float> Ytest(testSize, 10);

    std::cout << "getting data..." << std::endl;

    // parse input and insert into matrices for testing data
    std::string line;
    for(int p = 0; p < trainSize; p++){
        std::getline(trainFile, line);
        update(Xtrain, Ytrain, line, p);
    }

    trainFile.close();

    // parse and insert into matrices
    for(int p = 0; p < testSize; p++){
        std::getline(testFile, line);
        update(Xtest, Ytest, line, p);
    }

    testFile.close();

    std::vector<int> layers = {784, 16, 16, 10};
    NNet net(layers);

    std::cout << "training..." << std::endl;
    
    net.train(Xtrain, Ytrain, 0.001, 0.001, 100);
    
    Matrix<float> Ypred = net.predict(Xtest);
    
    std::cout << "testing..." << std::endl;

    float correct = 0.0;
    Ypred = max(Ypred);
    for(std::size_t i = 0; i < Ytest.shape().first; i++){
        for(std::size_t j = 0; j < Ytest.shape().second; j++){
            if(Ytest.at(i, j) == 1 && Ypred.at(i, j) == 1){
                correct++;
                break;
            }
        }
    }
    std::cout << "accuracy: " << correct/testSize << std::endl;
}