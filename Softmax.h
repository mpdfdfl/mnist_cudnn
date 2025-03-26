#pragma once
#include "Layer.h"

//
/****************************************************************
 * Softmax Layer                                                  *
 ****************************************************************/

template <typename ftype>
class Softmax : public Layer<ftype>
{
public:
    Softmax(std::string name);
    ~Softmax();

    Blob<ftype> *forward(Blob<ftype> *input);
    Blob<ftype> *backward(Blob<ftype> *grad_input);
};

template <typename ftype>
Softmax<ftype>::Softmax(std::string name)
{
    this->name_ = name;
}

template <typename ftype>
Softmax<ftype>::~Softmax()
{
}

template <typename ftype>
Blob<ftype> *Softmax<ftype>::forward(Blob<ftype> *input)
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
    checkCudnnErrors(
        cudnnSoftmaxForward(this->cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                            &alpha, this->input_desc_, input->cuda(),
                            &beta, this->output_desc_, this->output_->cuda()));

    return this->output_;
}

template <typename ftype>
Blob<ftype> *Softmax<ftype>::backward(Blob<ftype> *target)
{
    checkCudaErrors(cudaDeviceSynchronize());

    if (this->grad_input_ == nullptr || this->batch_size_ != target->n())
    {
        if (this->grad_input_ == nullptr)
            this->grad_input_ = new Blob<ftype>(this->input_->shape());
        else
            this->grad_input_->reset(this->input_->shape());
    }

    // set grad_input_ as predict
    checkCudaErrors(cudaMemcpyAsync(this->grad_input_->cuda(),
                                    this->output_->cuda(), this->output_->buf_size(),
                                    cudaMemcpyDeviceToDevice));

    // set grad_input_ = predict - target

    float alpha = (-1.0f);
    // y = alpha * x + y
    checkCublasErrors(cublasAxpyEx(this->cuda_->cublas(),
                                   target->len(),                                          // 数据的长度
                                   &alpha, getCudaDataType<float>(),                       // alpna
                                   target->cuda(), getCudaDataType<ftype>(), 1,            // x x的数据类型 x的步长
                                   this->grad_input_->cuda(), getCudaDataType<ftype>(), 1, // y
                                   getCudaDataType<float>()                                // 计算模式设置为FP32
                                   ));

    // normalize the grad_input_ by the batch size
    // x = scale * x
    float scale = (1.f / static_cast<float>(target->n()));
    checkCublasErrors(cublasScalEx(this->cuda_->cublas(),
                                   this->grad_input_->len(),
                                   &scale, getCudaDataType<float>(),                       // scale
                                   this->grad_input_->cuda(), getCudaDataType<ftype>(), 1, // x
                                   getCudaDataType<float>()));

    return this->grad_input_;
}