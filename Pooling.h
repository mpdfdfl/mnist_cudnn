#pragma once

#include <string>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "helper.h"
#include "Layer.h"

template <typename ftype>
class Pooling : public Layer<ftype>
{
public:
    Pooling(std::string name,
            int kernel_size,
            int padding,
            int stride,
            cudnnPoolingMode_t mode);
    ~Pooling();

    Blob<ftype> *forward(Blob<ftype> *input);
    Blob<ftype> *backward(Blob<ftype> *grad_output);

private:
    int kernel_size_;
    int padding_;
    int stride_;
    cudnnPoolingMode_t mode_;

    std::array<int, 4> output_size_;
    cudnnPoolingDescriptor_t pool_desc_; // 池化操作描述符
};

template <typename ftype>
Pooling<ftype>::Pooling(std::string name,
                        int kernel_size,
                        int padding,
                        int stride,
                        cudnnPoolingMode_t mode) : kernel_size_(kernel_size),
                                                   padding_(padding),
                                                   stride_(stride),
                                                   mode_(mode)
{
    this->name_ = name;

    cudnnCreatePoolingDescriptor(&pool_desc_);
    cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
                                kernel_size_, kernel_size_, // 池化窗口
                                padding_, padding_,         // 填充值
                                stride_, stride_            // 移动步数
    );
}

template <typename ftype>
Pooling<ftype>::~Pooling()
{
    cudnnDestroyPoolingDescriptor(pool_desc_);
}

template <typename ftype>
Blob<ftype> *Pooling<ftype>::forward(Blob<ftype> *input)
{
    if (this->input_ == nullptr || this->batch_size_ != input->n())
    {
        this->input_ = input;

        // resource initialize
        this->input_desc_ = this->input_->tensor();
        this->batch_size_ = input->n();

        // setting output 设置输出维度
        cudnnGetPooling2dForwardOutputDim(pool_desc_,        // 池化操作描述符
                                          this->input_desc_, // 输入张量描述符
                                          &output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]);
        if (this->output_ == nullptr)
            this->output_ = new Blob<ftype>(output_size_);
        else
            this->output_->reset(output_size_);

        this->output_desc_ = this->output_->tensor();
    }

    float alpha = (1.0f), beta = (0.0f);
    cudnnPoolingForward(this->cuda_->cudnn(),
                        pool_desc_,
                        &alpha,
                        this->input_desc_, this->input_->cuda(),
                        &beta,
                        this->output_desc_, this->output_->cuda());

    return this->output_;
}

template <typename ftype>
Blob<ftype> *Pooling<ftype>::backward(Blob<ftype> *grad_output)
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
    checkCudnnErrors(
        cudnnPoolingBackward(this->cuda_->cudnn(),
                             pool_desc_,
                             &alpha,
                             this->output_desc_, this->output_->cuda(),
                             this->output_desc_, grad_output->cuda(),
                             this->input_desc_, this->input_->cuda(),
                             &beta,
                             this->input_desc_, this->grad_input_->cuda()));

    return this->grad_input_;
}
