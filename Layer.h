#pragma once

#include <string>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "helper.h"
#include "matmul.cuh"
#include "Network.h"
#include <random>
__device__ float lerp(float start, float end, float weight)
{
    // grad, m, beta1
    return fma(weight, end, fma(-weight, start, start));
}

// adamw优化器
template <typename ftype>
__global__ void adamw_kernel(ftype *params_memory,        // 参数
                             float *master_params_memory, // fp32存储的参数，用户混合精度计算
                             ftype *grads_memory,         // 梯度
                             float *m_memory,
                             float *v_memory,
                             size_t num_parameters, // 参数个数
                             float learning_rate,   // 学习率
                             float beta1, float beta2,
                             float beta1_correction, float beta2_correction,
                             float eps)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_parameters)
    {
        return;
    }
    float grad = (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;

    m /= beta1_correction;
    v /= beta2_correction;

    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // 更新参数
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps)));

    params_memory[idx] = (ftype)param; //?
    if (master_params_memory != NULL)
    {
        master_params_memory[idx] = param;
    }
}

/****************************************************************
 * Layer definition                                             *
 ****************************************************************/

template <typename ftype> // 模板前向声明
class Network;

template <typename ftype>
class Layer
{
public:
    Layer();
    ~Layer();

    virtual Blob<ftype> *forward(Blob<ftype> *input) = 0;
    virtual Blob<ftype> *backward(Blob<ftype> *grad_input) = 0;

    std::string get_name() { return name_; }

    virtual float get_loss(Blob<ftype> *target);
    virtual int get_accuracy(Blob<ftype> *target);

    void set_cuda_context(CudaContext *context) { cuda_ = context; }

    void set_load_pretrain() { load_pretrain_ = true; };
    void set_gradient_stop() { gradient_stop_ = true; }

    /* Weight Freeze or Unfreeze */
    void freeze() { freeze_ = true; }
    void unfreeze() { freeze_ = false; }

protected:
    // name of layer
    std::string name_;

    // tensor descriptor for the input/output tensors
    // 输入输出的tensor设置
    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;

    // output memory
    Blob<ftype> *input_ = nullptr;       /* x  */
    Blob<ftype> *output_ = nullptr;      /* y  */
    Blob<ftype> *grad_input_ = nullptr;  /* dx */
    Blob<ftype> *grad_output_ = nullptr; /* dy */

    // master weights & bias
    bool freeze_ = false;                 /* control parameter updates */
    Blob<ftype> *weights_ = nullptr;      /* w */
    Blob<ftype> *biases_ = nullptr;       /* b */
    Blob<ftype> *grad_weights_ = nullptr; /* dw */
    Blob<ftype> *grad_biases_ = nullptr;  /* db */

    // 进行混合精度计算所以要存储fp32的权重
    Blob<float> *master_weights_ = nullptr; /* m_w */
    Blob<float> *master_biases_ = nullptr;  /* m_b */

    // 存放adamw的两个动量
    Blob<float> *m_memory_weight = nullptr;
    Blob<float> *v_memory_weight = nullptr;

    Blob<float> *m_memory_biases = nullptr;
    Blob<float> *v_memory_biases = nullptr;

    int batch_size_ = 0; // mini-batch size

    // initialize weights along with the input size
    void init_weight_bias(unsigned int seed = 0);
    void update_weights_biases(float learning_rate, int t);

    // cuda handle container
    CudaContext *cuda_ = nullptr;

    // pretrain parameters
    bool load_pretrain_ = false;
    int load_parameter();
    int save_parameter();

    // gradient stop tagging
    bool gradient_stop_ = false;

    friend class Network<ftype>;
};
template <typename ftype>
Layer<ftype>::Layer()
{
    /* do nothing */
}
template <typename ftype>
Layer<ftype>::~Layer()
{
    if (output_ != nullptr)
        delete output_;
    if (grad_input_ != nullptr)
        delete grad_input_;

    if (weights_ != nullptr)
        delete weights_;
    if (biases_ != nullptr)
        delete biases_;
    if (grad_weights_ != nullptr)
        delete grad_weights_;
    if (grad_biases_ != nullptr)
        delete grad_biases_;

    // // 进行混合精度计算所以要存储fp32的权重
    // Blob<float> *master_weights_ = nullptr; /* m_w */
    // Blob<float> *master_biases_ = nullptr;  /* m_b */

    // // 存放adamw的两个动量
    // Blob<float> *m_memory_weight = nullptr;
    // Blob<float> *v_memory_weight = nullptr;

    // Blob<float> *m_memory_biases = nullptr;
    // Blob<float> *v_memory_biases = nullptr;

    if (master_weights_ != nullptr)
        delete master_weights_;
    if (master_biases_ != nullptr)
        delete master_biases_;
    if (m_memory_weight != nullptr)
        delete m_memory_weight;
    if (v_memory_weight != nullptr)
        delete v_memory_weight;
    if (m_memory_biases != nullptr)
        delete m_memory_biases;
    if (v_memory_biases != nullptr)
        delete v_memory_biases;
}

template <typename ftype>
void Layer<ftype>::init_weight_bias(unsigned int seed) // 初始化权重参数
{
    checkCudaErrors(cudaDeviceSynchronize());

    if (weights_ == nullptr || biases_ == nullptr)
        return;

    // Create random network
    std::random_device rd;
    // std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));
    unsigned int fixed_seed = 12345; // 设置固定的种子值
    std::mt19937 gen(fixed_seed);    // 使用固定的种子初始化

    // He uniform distribution
    float range = sqrt(6.f / input_->size()); // He's initialization
    std::uniform_real_distribution<> dis(-range, range);

    for (int i = 0; i < weights_->len(); i++)
    {
        master_weights_->ptr()[i] = static_cast<float>(dis(gen));
        weights_->ptr()[i] = static_cast<ftype>(master_weights_->ptr()[i]);
    }

    for (int i = 0; i < biases_->len(); i++)
    {
        master_biases_->ptr()[i] = static_cast<float>(0.f);
        biases_->ptr()[i] = static_cast<ftype>(master_biases_->ptr()[i]);
    }
    for (int i = 0; i < m_memory_weight->len(); i++)
    {
        m_memory_weight->ptr()[i] = static_cast<float>(0.f);
        v_memory_weight->ptr()[i] = static_cast<float>(0.f);
    }

    for (int i = 0; i < m_memory_biases->len(); i++)
    {
        m_memory_biases->ptr()[i] = static_cast<float>(0.f);
        v_memory_biases->ptr()[i] = static_cast<float>(0.f);
    }

    // copy initialized value to the device
    weights_->to(DeviceType::cuda);
    biases_->to(DeviceType::cuda);
    master_weights_->to(DeviceType::cuda);
    master_biases_->to(DeviceType::cuda);

    m_memory_weight->to(DeviceType::cuda);
    v_memory_weight->to(DeviceType::cuda);
    m_memory_biases->to(DeviceType::cuda);
    v_memory_biases->to(DeviceType::cuda);

    std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}
template <typename ftype>
void Layer<ftype>::update_weights_biases(float learning_rate, int t)
{
    float eps = -1.f * learning_rate;
    // 手写实现混合精度运算
    if (weights_ != nullptr && grad_weights_ != nullptr)
    {
        // w = w + eps * dw
        int block_size = 512;
        int num_blocks = CEIL_DIV(weights_->len(), block_size);
        float beta1 = 0.9;
        float beta2 = 0.999;
        float beta1_correction = 1.0f - powf(beta1, t);
        float beta2_correction = 1.0f - powf(beta2, t);
        adamw_kernel<<<num_blocks, block_size>>>(weights_->cuda(),
                                                 master_weights_->cuda(),
                                                 grad_weights_->cuda(),
                                                 m_memory_weight->cuda(),
                                                 v_memory_weight->cuda(),
                                                 weights_->len(),
                                                 learning_rate,
                                                 beta1,
                                                 beta2,
                                                 beta1_correction,
                                                 beta2_correction,
                                                 1e-8f);
        cudaDeviceSynchronize();
    }

    if (biases_ != nullptr && grad_biases_ != nullptr)
    {
        // b = b + eps * db
        int block_size = 512;
        int num_blocks = CEIL_DIV(biases_->len(), block_size);
        float beta1 = 0.9;
        float beta2 = 0.999;
        float beta1_correction = 1.0f - powf(beta1, t);
        float beta2_correction = 1.0f - powf(beta2, t);
        adamw_kernel<<<num_blocks, block_size>>>(biases_->cuda(),
                                                 master_biases_->cuda(),
                                                 grad_biases_->cuda(),
                                                 m_memory_biases->cuda(),
                                                 v_memory_biases->cuda(),
                                                 biases_->len(),
                                                 learning_rate,
                                                 beta1,
                                                 beta2,
                                                 beta1_correction,
                                                 beta2_correction,
                                                 1e-8f);
        cudaDeviceSynchronize();
    }
}

template <typename ftype>
float Layer<ftype>::get_loss(Blob<ftype> *target)
{
    assert("No Loss layer has no loss." && false);
    return EXIT_FAILURE;
}
template <typename ftype>
int Layer<ftype>::get_accuracy(Blob<ftype> *target)
{
    assert("No Loss layer cannot estimate accuracy." && false);
    return EXIT_FAILURE;
}
template <typename ftype>
int Layer<ftype>::load_parameter() // 这方模型没有读入参数文件
{
    std::stringstream filename_weights, filename_biases;

    // load weights and biases pretrained parameters
    filename_weights << name_ << ".bin";
    if (weights_->file_read(filename_weights.str()))
        return -1;

    filename_biases << name_ << ".bias.bin";
    if (biases_->file_read(filename_biases.str()))
        return -2;

    std::cout << ".. loaded " << name_ << " pretrain parameter.." << std::endl;

    return 0;
}
template <typename ftype>
int Layer<ftype>::save_parameter()
{
    std::stringstream filename_weights, filename_biases;

    std::cout << ".. saving " << name_ << " parameter ..";

    // Write weights file
    if (weights_)
    {
        filename_weights << name_ << ".bin";
        if (weights_->file_write(filename_weights.str()))
            return -1;
    }

    // Write bias file
    if (biases_)
    {
        filename_biases << name_ << ".bias.bin";
        if (biases_->file_write(filename_biases.str()))
            return -2;
    }

    std::cout << " done .." << std::endl;

    return 0;
}