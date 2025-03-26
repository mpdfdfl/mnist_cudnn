#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cudnn.h>

#include "helper.h"
#include "CrossEntropyLoss.h"
#include "Layer.h"
#include "Activation.h"
#include "blob.h"
#include "Conv2D.h"
#include "CrossEntropyLoss.h"
#include "Dense.h"
#include "mnist.h"
#include "Softmax.h"
#include "Pooling.h"

typedef enum
{
    training,
    inference
} WorkloadType;

template <typename ftype>
class Network
{
public:
    Network();
    ~Network();

    void add_layer(Layer<ftype> *layer);

    Blob<ftype> *forward(Blob<ftype> *input);
    void backward(Blob<ftype> *input = nullptr);
    void update(float learning_rate = 0.02f, int t = 1);

    int load_pretrain();
    int write_file();

    float loss(Blob<ftype> *predict, Blob<ftype> *target);
    float get_accuracy(Blob<ftype> *predict, Blob<ftype> *target);

    void cuda();
    void train();
    void test();

    Blob<ftype> *output_;

    std::vector<Layer<ftype> *> layers();

private:
    std::vector<Layer<ftype> *> layers_;

    CudaContext *cuda_ = nullptr;

    WorkloadType phase_ = inference;
    CrossEntropyLoss<ftype> *Netloss = nullptr;
};

template <typename ftype>
Network<ftype>::Network()
{
    // nothing
}

template <typename ftype>
Network<ftype>::~Network()
{
    // destroy network
    for (auto layer : layers_)
        delete layer;

    // terminate CUDA context
    if (cuda_ != nullptr)
        delete cuda_;
}

template <typename ftype>
void Network<ftype>::add_layer(Layer<ftype> *layer)
{
    layers_.push_back(layer);

    // tagging layer to stop gradient if it is the first layer
    if (layers_.size() == 1)
        layers_.at(0)->set_gradient_stop();
}

template <typename ftype>
Blob<ftype> *Network<ftype>::forward(Blob<ftype> *input)
{
    output_ = input;
    // input->print("input", true, 1, 28);
    for (auto layer : layers_)
    {
        // std::cout << layer->name_ << std::endl;
        output_ = layer->forward(output_);
        // output_->print(layer->name_, true, 1, 28);
    }
    // output_->print("output", true, 1);
    return output_;
}

template <typename ftype>
void Network<ftype>::backward(Blob<ftype> *target)
{
    Blob<ftype> *gradient = target;

    if (phase_ == inference)
        return;

    // back propagation.. update weights internally.....
    for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
    {
        // getting back propagation status with gradient size

        gradient = (*layer)->backward(gradient);
    }
}

template <typename ftype>
void Network<ftype>::update(float learning_rate, int t)
{
    if (phase_ == inference)
        return;

    for (auto layer : layers_)
    {
        // if no parameters, then pass
        if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
            layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
            continue;
        layer->update_weights_biases(learning_rate, t);
    }
}

template <typename ftype>
int Network<ftype>::write_file()
{
    std::cout << ".. store weights to the storage .." << std::endl;
    for (auto layer : layers_)
    {
        int err = layer->save_parameter();

        if (err != 0)
        {
            std::cout << "-> error code: " << err << std::endl;
            exit(err);
        }
    }

    return 0;
}

template <typename ftype>
int Network<ftype>::load_pretrain()
{
    for (auto layer : layers_)
    {
        layer->set_load_pretrain();
    }

    return 0;
}

// 1. initialize cuda resource container
// 2. register the resource container to all the layers
template <typename ftype>
void Network<ftype>::cuda()
{
    cuda_ = new CudaContext();

    std::cout << ".. model Configuration .." << std::endl;
    for (auto layer : layers_)
    {
        std::cout << "CUDA: " << layer->get_name() << std::endl;
        layer->set_cuda_context(cuda_);
    }
}

//
template <typename ftype>
void Network<ftype>::train()
{
    phase_ = training;

    // unfreeze all layers
    for (auto layer : layers_)
    {
        layer->unfreeze();
    }
}

template <typename ftype>
void Network<ftype>::test()
{
    phase_ = inference;

    // freeze all layers
    for (auto layer : layers_)
    {
        layer->freeze();
    }
}

template <typename ftype>
std::vector<Layer<ftype> *> Network<ftype>::layers()
{
    return layers_;
}

template <typename ftype>
float Network<ftype>::loss(Blob<ftype> *predict, Blob<ftype> *target)
{
    if (Netloss == nullptr)
    {
        Netloss = new CrossEntropyLoss<ftype>();
    }
    return Netloss->loss(predict, target);
}

template <typename ftype>
float Network<ftype>::get_accuracy(Blob<ftype> *predict, Blob<ftype> *target)
{
    if (Netloss == nullptr)
    {
        Netloss = new CrossEntropyLoss<ftype>();
    }
    return Netloss->accuracy(predict, target);
}
