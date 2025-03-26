#pragma once

#include <string>
#include <sstream>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.h"
#include "helper.h"
#include "Layer.h"

template <typename ftype>
class Conv2D : public Layer<ftype>
{
public:
    Conv2D(std::string name,
           int out_channels,
           int kernel_size,
           int stride = 1,
           int padding = 0,
           int dilation = 1);
    ~Conv2D();

    Blob<ftype> *forward(Blob<ftype> *input);
    Blob<ftype> *backward(Blob<ftype> *grad_output);

private:
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    int dilation_;

    std::array<int, 4> output_size_;

    // convolution
    cudnnConvolutionDescriptor_t conv_desc_; // 卷积描述符

    cudnnConvolutionFwdAlgo_t conv_fwd_algo_;              // 前向传播算法描述符
    cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;     // 后向传播算法描述符 ，计算输入向量的梯度
    cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_; // 计算权重梯度

    // weight/bias descriptor
    cudnnFilterDescriptor_t filter_desc_; // weight描述符
    cudnnTensorDescriptor_t bias_desc_;   // bias描述符

    size_t workspace_size = 0;
    void **d_workspace = nullptr;
    void set_workspace();
};

template <typename ftype>
Conv2D<ftype>::Conv2D(std::string name,
                      int out_channels, // 卷积核数量，即输出通道数
                      int kernel_size,  // 卷积核大小
                      int stride,       // 卷积核滑动步长
                      int padding,      // 填充
                      int dilation      // 膨胀系数，用于膨胀卷积
                      ) : out_channels_(out_channels),
                          kernel_size_(kernel_size),
                          stride_(stride),
                          padding_(padding),
                          dilation_(dilation)
{
    this->name_ = name;

    // create cudnn container handles
    cudnnCreateFilterDescriptor(&filter_desc_);

    cudnnCreateConvolutionDescriptor(&conv_desc_);
    checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                     padding_, padding_, // 两个方向上的填充
                                                     stride_, stride_,   // 两个方向上的步长
                                                     dilation_, dilation_,
                                                     CUDNN_CROSS_CORRELATION,
                                                     getCudnnDataType<ftype>() // 采用fp32的形式进行计算
                                                     ));

    d_workspace = nullptr;
}
template <typename ftype>
Conv2D<ftype>::~Conv2D()
{
    // distroy cudnn container resources
    cudnnDestroyFilterDescriptor(filter_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);

    // terminate internal created blobs
    if (d_workspace != nullptr)
        cudaFree(d_workspace);
}

template <typename ftype>
void Conv2D<ftype>::set_workspace() // 配置卷积核算法，即器所需要的空间
{
    size_t temp_size = 0;

    cudnnConvolutionFwdAlgoPerf_t fwd_algo_perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_algo_perf_results[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_algo_perf_results[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];

    // forward

    int algo_max_count;
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(this->cuda_->cudnn(), &algo_max_count)); //	获取当前 cuDNN 版本支持的最大前向卷积算法个数
    checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(this->cuda_->cudnn(),
                                                            this->input_desc_,  // 输入张量描述符
                                                            filter_desc_,       // 卷积核描述符
                                                            conv_desc_,         // 卷积操作描述符
                                                            this->output_desc_, // 输出张量描述符
                                                            algo_max_count,     // 返回的最大算法数量
                                                            0,
                                                            fwd_algo_perf_results));

    conv_fwd_algo_ = fwd_algo_perf_results[0].algo; // 选取第一个算法

    // 计算需要分配的空间大小
    checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(this->cuda_->cudnn(),
                                                             this->input_desc_,
                                                             filter_desc_,
                                                             conv_desc_,
                                                             this->output_desc_, // 各类描述符
                                                             conv_fwd_algo_,
                                                             &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // backward filter  计算卷积核梯度
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(this->cuda_->cudnn(), &algo_max_count));
    checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(this->cuda_->cudnn(),
                                                                   this->input_desc_,  // 输入张量描述符
                                                                   this->output_desc_, // 输出梯度描述符
                                                                   conv_desc_,         // 卷积描述符
                                                                   filter_desc_,       // 卷积核梯度描述符
                                                                   algo_max_count,
                                                                   0,
                                                                   bwd_filter_algo_perf_results));

    conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;

    checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(this->cuda_->cudnn(),
                                                                    this->input_desc_,
                                                                    this->output_desc_,
                                                                    conv_desc_,
                                                                    filter_desc_,
                                                                    conv_bwd_filter_algo_,
                                                                    &temp_size));

    workspace_size = std::max(workspace_size, temp_size);

    // bwd - data  计算反向传播的时候的输入张量梯度

    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(this->cuda_->cudnn(), &algo_max_count));
    checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(this->cuda_->cudnn(),
                                                                 filter_desc_,       // 卷积核描述符
                                                                 this->output_desc_, // 输出梯度张量描述符
                                                                 conv_desc_,         // 卷积操作描述符
                                                                 this->input_desc_,  // 输入梯度描述符
                                                                 algo_max_count,
                                                                 0,
                                                                 bwd_data_algo_perf_results));
    conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;

    checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(this->cuda_->cudnn(),
                                                                  filter_desc_,
                                                                  this->output_desc_,
                                                                  conv_desc_,
                                                                  this->input_desc_,
                                                                  conv_bwd_data_algo_,
                                                                  &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    if (workspace_size > 0)
    {
        if (d_workspace != nullptr)
            checkCudaErrors(cudaFree(d_workspace));
        checkCudaErrors(cudaMalloc((void **)&d_workspace, workspace_size));
    }
}

template <typename ftype>
Blob<ftype> *Conv2D<ftype>::forward(Blob<ftype> *input)
{
    // initialize weights and bias
    if (this->weights_ == nullptr)
    {
        // initialize containers handles
        checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_desc_,
                                                    getCudnnDataType<ftype>(), // 数据类型
                                                    CUDNN_TENSOR_NCHW,         // 数据存储格式
                                                    out_channels_,             // 输出的通道数
                                                    input->c(),                // 输入通道数
                                                    kernel_size_,              // 卷积核高度
                                                    kernel_size_               // 卷积核宽度
                                                    ));

        this->weights_ = new Blob<ftype>(out_channels_, input->c(), kernel_size_, kernel_size_);
        this->biases_ = new Blob<ftype>(1, out_channels_); // bias size

        this->master_weights_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        this->master_biases_ = new Blob<float>(1, out_channels_); // bias size

        this->m_memory_weight = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        this->m_memory_biases = new Blob<float>(1, out_channels_); // bias size

        this->v_memory_weight = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
        this->v_memory_biases = new Blob<float>(1, out_channels_); // bias size
        bias_desc_ = this->biases_->tensor();
    }

    // initilaize input and output
    if (this->input_ == nullptr || this->batch_size_ != input->n())
    {
        // initialize input
        this->input_ = input;
        this->input_desc_ = input->tensor();
        this->batch_size_ = input->n();

        // initilaize output 通过卷积描述符，以及输入tensor 计算输出维度
        checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(
            conv_desc_,
            this->input_desc_,
            filter_desc_,
            &output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]));

        if (this->output_ == nullptr)
            this->output_ = new Blob<ftype>(output_size_);
        else
            this->output_->reset(output_size_);

        this->output_desc_ = this->output_->tensor();

        // 设置工作区域，计算最优算法
        set_workspace();

        // initialize weights
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

            this->init_weight_bias();
        }
        else
        {
            /* do nothing */
        }
    }

    float alpha = (1.0f), beta = (0.0f);
    // this->weights_->print("weight_half", true, 1);
    // this->input_->print("asdasda", true, 1, 28);
    checkCudnnErrors(cudnnConvolutionForward(this->cuda_->cudnn(),
                                             &alpha,
                                             this->input_desc_,
                                             this->input_->cuda(),
                                             filter_desc_,                // 卷积核描述符
                                             this->weights_->cuda(),      // 卷积核权重
                                             conv_desc_,                  // 卷积操作描述符
                                             conv_fwd_algo_,              // 卷积fwd算法
                                             d_workspace, workspace_size, // 分配的工作区间
                                             &beta,
                                             this->output_desc_,
                                             this->output_->cuda()));
    // 这个函数有广播操作
    checkCudnnErrors(cudnnAddTensor(this->cuda_->cudnn(),
                                    &alpha, bias_desc_, this->biases_->cuda(),
                                    &alpha, this->output_desc_, this->output_->cuda()));

    return this->output_;
}

template <typename ftype>
Blob<ftype> *Conv2D<ftype>::backward(Blob<ftype> *grad_output)
{
    // initialize grad_output back-propagation space
    if (this->grad_input_ == nullptr || this->batch_size_ != grad_output->n())
    {
        this->grad_output_ = grad_output;
        this->grad_weights_ = new Blob<ftype>(this->weights_->shape());
        this->grad_biases_ = new Blob<ftype>(1, this->biases_->c());

        if (this->grad_input_ == nullptr)
            this->grad_input_ = new Blob<ftype>(this->input_->shape());
        else
            this->grad_input_->reset(this->input_->shape());
    }

    float alpha = (1.0f), beta = (0.0f);
    // gradients of biases
    checkCudnnErrors(
        cudnnConvolutionBackwardBias(this->cuda_->cudnn(),
                                     &alpha,
                                     this->output_desc_,  // 输出梯度描述符
                                     grad_output->cuda(), // 输出梯度
                                     &beta,
                                     bias_desc_, // bias张量描述符
                                     this->grad_biases_->cuda()));

    // gradients of weights
    checkCudnnErrors(
        cudnnConvolutionBackwardFilter(this->cuda_->cudnn(),
                                       &alpha,
                                       this->input_desc_, this->input_->cuda(),        // 输入描述符和数据
                                       this->output_desc_, this->grad_output_->cuda(), // 输出梯度和数据
                                       conv_desc_, conv_bwd_filter_algo_, d_workspace, workspace_size,
                                       &beta,
                                       filter_desc_, this->grad_weights_->cuda() // 权重描述符及其指针
                                       ));

    // gradients of input data
    if (!this->gradient_stop_)
        checkCudnnErrors(
            cudnnConvolutionBackwardData(this->cuda_->cudnn(),
                                         &alpha,
                                         filter_desc_, this->weights_->cuda(),    // 卷积核权重
                                         this->output_desc_, grad_output->cuda(), // 输出梯度
                                         conv_desc_, conv_bwd_data_algo_, d_workspace, workspace_size,
                                         &beta,
                                         this->input_desc_, this->grad_input_->cuda() // 输入梯度
                                         ));

    return this->grad_input_;
}
