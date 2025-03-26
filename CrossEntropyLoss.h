#pragma once

#include "blob.h"

template <typename ftype>
class CrossEntropyLoss
{
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    float loss(Blob<ftype> *predict, Blob<ftype> *target);
    float accuracy(Blob<ftype> *predict, Blob<ftype> *target);

private:
    // reduced loss
    float h_loss_ = 0.f;
    float *d_loss_ = nullptr;
};
template <typename ftype>
CrossEntropyLoss<ftype>::CrossEntropyLoss()
{
    cudaMalloc((void **)&d_loss_, sizeof(float));
}

template <typename ftype>
CrossEntropyLoss<ftype>::~CrossEntropyLoss()
{
    if (d_loss_ != nullptr)
        cudaFree(d_loss_);
    d_loss_ = nullptr;
}

template <typename ftype>
__global__ void softmax_loss_kernel(float *reduced_loss, ftype *predict, ftype *target, int N)
{

    __shared__ float s[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / 32; // 表示当前是第几个线程束
    int laneId = threadIdx.x % 32;
    float val = (idx < N) ? logf(fmaxf(predict[idx], 1e-8f)) * float(target[idx]) : (0.0f);
    val = -val;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset); // 与异或位置交换
    }

    if (laneId == 0)
        s[warpId] = val;
    __syncthreads();
    if (warpId == 0)
    {
        int warpNum = blockDim.x / 32;
        val = (laneId < warpNum) ? s[laneId] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        }
        if (laneId == 0)
        {
            atomicAdd(reduced_loss, val);
        }
    }
}
template <typename ftype>
float CrossEntropyLoss<ftype>::loss(Blob<ftype> *predict, Blob<ftype> *target)
{

    int batch_size = target->n();
    int num_outputs = target->c();

    int block_size = 1024;
    int grid_size = (block_size + num_outputs * batch_size - 1) / block_size;

    cudaMemset(d_loss_, 0, sizeof(float)); // 将 d_loss_ 置为 0

    softmax_loss_kernel<ftype><<<grid_size, block_size>>>(d_loss_, predict->cuda(), target->cuda(), batch_size * num_outputs);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss_, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);
    // batch mean loss
    return float(h_loss_) / float(batch_size);
}

template <typename ftype>
float CrossEntropyLoss<ftype>::accuracy(Blob<ftype> *predict, Blob<ftype> *target)
{
    int batch_size = predict->n();
    int output_size = predict->size();

    ftype *h_output, *h_target;
    int idx_output, idx_target;
    int hit_count = 0;

    // get predicts and targets
    h_output = predict->to(host);
    h_target = target->to(host);

    // idx_output = idx_target = 0;
    for (int b = 0; b < batch_size; b++)
    {
        idx_output = 0;
        idx_target = 0;

        for (int i = 1; i < 10; i++)
        {
            if (float(h_output[b * output_size + i]) > float(h_output[b * output_size + idx_output]))
                idx_output = i;
            if (float(h_target[b * output_size + i]) > float(h_target[b * output_size + idx_target]))
                idx_target = i;
        }

        if (idx_output == idx_target)
            hit_count++;
    }

    return float(hit_count) / float(batch_size);
}