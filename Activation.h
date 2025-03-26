#pragma once

#include <string>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "helper.h"
#include "matmul.cuh"
#include <random>

template <typename ftype>
class Activation : public Layer<ftype>
{
public:
    Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
    ~Activation();

    Blob<ftype> *forward(Blob<ftype> *input);
    Blob<ftype> *backward(Blob<ftype> *grad_input);

private:
    cudnnActivationDescriptor_t act_desc_;
    cudnnActivationMode_t mode_;
    float coef_;
};

/****************************************************************
 * Activation Layer                                             *
 ****************************************************************/
template <typename ftype>
Activation<ftype>::Activation(std::string name, cudnnActivationMode_t mode, float coef)
{
    this->name_ = name;
    mode_ = mode;
    coef_ = coef;

    cudnnCreateActivationDescriptor(&act_desc_);
    cudnnSetActivationDescriptor(act_desc_, mode, CUDNN_PROPAGATE_NAN, coef);
}
template <typename ftype>
Activation<ftype>::~Activation()
{
    cudnnDestroyActivationDescriptor(act_desc_);
}

template <typename ftype>
Blob<ftype> *Activation<ftype>::forward(Blob<ftype> *input)
{
    if (this->input_ == nullptr || this->batch_size_ != input->n())
    {
        this->input_ = input;
        this->input_desc_ = input->tensor();
        this->batch_size_ = input->n();

        if (this->output_ == nullptr)
            this->output_ = new Blob<ftype>(input->shape());
        else
            this->output_->reset(input->shape());

        this->output_desc_ = this->output_->tensor();
    }

    float alpha = (1.0f), beta = (0.0f);
    cudnnActivationForward(this->cuda_->cudnn(),
                           act_desc_,
                           &alpha,
                           this->input_desc_,
                           input->cuda(), // 输入
                           &beta,
                           this->output_desc_,
                           this->output_->cuda() // 输出
    );

    return this->output_;
}
template <typename ftype>
Blob<ftype> *Activation<ftype>::backward(Blob<ftype> *grad_output)
{
    if (this->grad_input_ == nullptr || this->batch_size_ != grad_output->n())
    {
        this->grad_output_ = grad_output;

        if (this->grad_input_ == nullptr)
            this->grad_input_ = new Blob<ftype>(this->input_->shape());
        else
            this->grad_input_->reset(this->input_->shape());
    }
    float alpha = (1.0f), beta = (0.0f);
    cudnnActivationBackward(this->cuda_->cudnn(),
                            act_desc_,
                            &alpha,
                            this->output_desc_, this->output_->cuda(),
                            this->output_desc_, grad_output->cuda(),
                            this->input_desc_, this->input_->cuda(),
                            &beta,
                            this->input_desc_, this->grad_input_->cuda());

    return this->grad_input_;
}
