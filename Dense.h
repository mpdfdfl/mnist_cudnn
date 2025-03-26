#pragma once
#include "Layer.h"

//
/****************************************************************
 * Dense Layer                                                  *
 ****************************************************************/
// 全连接层
template <typename ftype>
class Dense : public Layer<ftype>
{
public:
    Dense(std::string name, int out_size);
    ~Dense();

    Blob<ftype> *forward(Blob<ftype> *input);
    Blob<ftype> *backward(Blob<ftype> *grad_input);

private:
    int input_size_ = 0;
    int output_size_ = 0;

    ftype *d_one_vec = nullptr;
};

template <typename ftype>
Dense<ftype>::Dense(std::string name, int output_size)
{
    this->name_ = name;
    output_size_ = output_size;
}
template <typename ftype>
Dense<ftype>::~Dense()
{
    if (d_one_vec != nullptr)
        cudaFree(d_one_vec);
}
template <typename ftype>
__global__ void init_one_vec(ftype *d_one_vec, size_t length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= length)
        return;

    d_one_vec[i] = ftype(1.f);
}
template <typename ftype>
Blob<ftype> *Dense<ftype>::forward(Blob<ftype> *input)
{

    // for (int i = 0; i < weights_->len(); i++)
    // {
    //     master_weights_->ptr()[i] = static_cast<float>(dis(gen));
    //     weights_->ptr()[i] = static_cast<ftype>(master_weights_->ptr()[i]);
    // }
    // for (int i = 0; i < biases_->len(); i++)
    // {
    //     master_biases_->ptr()[i] = static_cast<float>(0.f);
    //     biases_->ptr()[i] = static_cast<ftype>(master_biases_->ptr()[i]);
    // }

    // for (int i = 0; i < m_memory_weight->len(); i++)
    // {
    //     m_memory_weight->ptr()[i] = static_cast<float>(0.f);
    //     v_memory_weight->ptr()[i] = static_cast<float>(0.f);
    // }

    // for (int i = 0; i < m_memory_biases->len(); i++)
    // {
    //     m_memory_biases->ptr()[i] = static_cast<float>(0.f);
    //     v_memory_biases->ptr()[i] = static_cast<float>(0.f);
    // }

    // initialize weights and biases
    if (this->weights_ == nullptr)
    {
        // setup parameter size information
        input_size_ = input->c() * input->h() * input->w();

        // initialize weight, bias, and output
        this->weights_ = new Blob<ftype>(1, 1, input_size_, output_size_);
        this->biases_ = new Blob<ftype>(1, 1, output_size_);

        this->master_weights_ = new Blob<float>(1, 1, input_size_, output_size_); // 用于混合精度计算
        this->master_biases_ = new Blob<float>(1, 1, output_size_);

        this->m_memory_weight = new Blob<float>(1, 1, input_size_, output_size_); // 用于混合精度计算
        this->m_memory_biases = new Blob<float>(1, 1, output_size_);

        this->v_memory_weight = new Blob<float>(1, 1, input_size_, output_size_); // 用于混合精度计算
        this->v_memory_biases = new Blob<float>(1, 1, output_size_);
    }

    // initilaize input and output
    if (this->input_ == nullptr || this->batch_size_ != input->n())
    {
        this->input_ = input;
        this->batch_size_ = input->n();

        if (this->output_ == nullptr)
            this->output_ = new Blob<ftype>(this->batch_size_, output_size_);
        else
            this->output_->reset(this->batch_size_, output_size_);

        this->output_->tensor();

        if (d_one_vec != nullptr)
            cudaFree(d_one_vec);
        checkCudaErrors(cudaMalloc((void **)&d_one_vec, sizeof(ftype) * this->batch_size_));
        init_one_vec<<<CEIL_DIV(this->batch_size_, BLOCK_DIM_1D), BLOCK_DIM_1D>>>(d_one_vec, this->batch_size_); // 赋值为1
        cudaDeviceSynchronize();

        // initialize weights and biases
        if (this->load_pretrain_ && !this->freeze_)
        {
            if (this->load_parameter())
            {
                std::cout << "error occurred.." << std::endl;
                exit(-1);
            }
        }
        else if (!this->freeze_)
        {

            this->init_weight_bias(); // 初始化
        }
        else
        {
            /* do nothing */
        }
    }

    // 运算这里因该换成cublaslt
    // output = input * weight

    matmul_cublaslt(this->cuda_->cublaslt(),
                    this->input_->cuda(),
                    this->weights_->cuda(),
                    this->output_->cuda(),
                    this->batch_size_, output_size_, input_size_);

    // output += d_one_vec * biases
    matmul_cublaslt(this->cuda_->cublaslt(),
                    d_one_vec,
                    this->biases_->cuda(),
                    this->output_->cuda(),
                    this->batch_size_, output_size_, 1,
                    true // 在原结果上进行累加
    );
    // this->input_->print(this->name_ + "::input", true, 1, 28);
    // this->weights_->print(this->name_ + "::weight", true);
    // this->biases_->print(this->name_ + "::bias", true);
    // this->output_->print(this->name_ + "::output", true);
    return this->output_;
}

template <typename ftype>
Blob<ftype> *Dense<ftype>::backward(Blob<ftype> *grad_output)
{
    if (this->grad_weights_ == nullptr)
    {
        this->grad_weights_ = new Blob<ftype>(this->weights_->shape());
        this->grad_biases_ = new Blob<ftype>(this->biases_->shape());
    }

    if (this->grad_input_ == nullptr || this->batch_size_ != grad_output->n())
    {
        this->grad_output_ = grad_output;

        if (this->grad_input_ == nullptr)
            this->grad_input_ = new Blob<ftype>(this->input_->shape());
        else
            this->grad_input_->reset(this->input_->shape());
    }

    // db = d_one_vec * y
    matmul_cublaslt<ftype>(this->cuda_->cublaslt(),
                           d_one_vec,
                           this->grad_output_->cuda(),
                           this->grad_biases_->cuda(),
                           1, output_size_, this->batch_size_);

    // dw = x_T * dy
    matmul_cublaslt<ftype>(this->cuda_->cublaslt(),
                           this->input_->cuda(),
                           this->grad_output_->cuda(),
                           this->grad_weights_->cuda(),
                           input_size_, output_size_, this->batch_size_,
                           false,
                           true // 矩阵A需要进行转置
    );

    // dx =  dy * w_T
    if (!this->gradient_stop_)
        matmul_cublaslt<ftype>(this->cuda_->cublaslt(),
                               this->grad_output_->cuda(),
                               this->weights_->cuda(),
                               this->grad_input_->cuda(),
                               this->batch_size_, input_size_, output_size_,
                               false,
                               false,
                               true // 矩阵B需要进行转置
        );

    return this->grad_input_;
}
