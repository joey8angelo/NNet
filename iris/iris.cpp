#include <iostream>
#include <vector>
#include <fstream>
#include "../NNet.h"

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
    if(in == "Iris-setosa")
        return {1, 0, 0};
    if(in == "Iris-versicolor")
        return {0, 1, 0};
    if(in == "Iris-virginica")
        return {0, 0, 1};
    return {0, 0, 0};
}

void update(Matrix<float>& X, Matrix<float>& Y, std::string line, int p){
    std::vector<std::string> parts = split(line, ',');
    X.at(p, 0) = std::stof(parts[0]);
    X.at(p, 1) = std::stof(parts[1]);
    X.at(p, 2) = std::stof(parts[2]);
    X.at(p, 3) = std::stof(parts[3]);
    std::vector<float> yparts = yval(parts[4]);
    Y.at(p, 0) = yparts[0];
    Y.at(p, 1) = yparts[1];
    Y.at(p, 2) = yparts[2];
}

int main(){
    // read in iris data
    std::ifstream file("iris/iris.csv");
    Matrix<float> Xtrain(75, 4);
    Matrix<float> Ytrain(75, 3);

    Matrix<float> Xtest(75, 4);
    Matrix<float> Ytest(75, 3);

    std::string line;
    for(int p = 0; p < 75; p++){
        std::getline(file, line);
        update(Xtrain, Ytrain, line, p);
        std::getline(file, line);
        update(Xtest, Ytest, line, p);
    }

    Xtrain = Xtrain.T();
    Ytrain = Ytrain.T();

    Xtest = Xtest.T();
    Ytest = Ytest.T();

    std::vector<std::vector<double>> errs;

    for(int l = 1; l <= 3; l++){
        errs.push_back({});
        for(float lam = 0.0001; lam <= 1000; lam *= 10){
            std::vector<int> layers = {4, 4*l, 3*l, 3};
            NNet net(layers);
            net.train(Xtrain, Ytrain, lam, 0.001, 100000);

            Matrix<float> Ypred = net.predict(Xtest);
            int correct = 0;
            for(int i = 0; i < 75; i++){
                if(Ypred.at(0, i) > Ypred.at(1, i) && Ypred.at(0, i) > Ypred.at(2, i)){
                    if(Ytest.at(0, i) == 1)
                        correct++;
                }else if(Ypred.at(1, i) > Ypred.at(0, i) && Ypred.at(1, i) > Ypred.at(2, i)){
                    if(Ytest.at(1, i) == 1)
                        correct++;
                }else{
                    if(Ytest.at(2, i) == 1)
                        correct++;
                }
            }
            errs[errs.size()-1].push_back(correct/75.0);
            std::cout << "hidden layer sizes: " << 4*l << "," << 3*l << " lambda: " << lam << std::endl;
        }
    }

    for(auto i : errs){
        std::cout << "[";
        for(auto j : i){
            std::cout << j << ", ";
        }
        std::cout << "]" << std::endl;
    }
}