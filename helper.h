#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <iostream>
#include <cuda_fp16.h> // 用于FP16和BF16数据类型
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <float.h>
// #include <helper_cuda.h>
#include <curand.h>

#define BLOCK_DIM_1D 512
#define BLOCK_DIM 16

/* DEBUG FLAGS */
#define DEBUG_FORWARD 0
#define DEBUG_BACKWARD 0

#define DEBUG_CONV 0
#define DEBUG_DENSE 0
#define DEBUG_SOFTMAX 0
#define DEBUG_UPDATE 0

#define DEBUG_LOSS 0
#define DEBUG_ACCURACY 0

/* CUDA API error return checker */
#ifndef checkCudaErrors
#define checkCudaErrors(err)                                                                        \
    {                                                                                               \
        if (err != cudaSuccess)                                                                     \
        {                                                                                           \
            fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, cudaGetErrorString(err), __FILE__, __LINE__);                              \
            fprintf(stderr, "%d\n", cudaSuccess);                                                   \
            exit(-1);                                                                               \
        }                                                                                           \
    }
#endif

static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define checkCublasErrors(err)                                                                        \
    {                                                                                                 \
        if (err != CUBLAS_STATUS_SUCCESS)                                                             \
        {                                                                                             \
            fprintf(stderr, "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, _cublasGetErrorEnum(err), __FILE__, __LINE__);                               \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

#define checkCudnnErrors(err)                                                                        \
    {                                                                                                \
        if (err != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                            \
            fprintf(stderr, "checkCudnnErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, cudnnGetErrorString(err), __FILE__, __LINE__);                              \
            exit(-1);                                                                                \
        }                                                                                            \
    }

// cuRAND API errors
static const char *_curandGetErrorEnum(curandStatus_t error)
{
    switch (error)
    {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define checkCurandErrors(err)                                                                        \
    {                                                                                                 \
        if (err != CURAND_STATUS_SUCCESS)                                                             \
        {                                                                                             \
            fprintf(stderr, "checkCurandErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, _curandGetErrorEnum(err), __FILE__, __LINE__);                               \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status)                        \
    {                                              \
        cublasCheck((status), __FILE__, __LINE__); \
    }

// container for cuda resources
class CudaContext
{
public:
    CudaContext()
    {
        cublasCheck(cublasCreate(&_cublas_handle));
        cublasCheck(cublasLtCreate(&_cublaslt_handle));
        checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudaContext()
    {
        cublasCheck(cublasDestroy(_cublas_handle));
        cublasCheck(cublasLtDestroy(_cublaslt_handle));
        checkCudnnErrors(cudnnDestroy(_cudnn_handle));
    }

    cublasHandle_t cublas()
    {
        // std::cout << "Get cublas request" << std::endl; getchar();
        return _cublas_handle;
    };
    cublasLtHandle_t cublaslt()
    {
        return _cublaslt_handle;
    };
    cudnnHandle_t cudnn()
    {
        return _cudnn_handle;
    };

    const float one = 1.f;
    const float zero = 0.f;
    const float minus_one = -1.f;

private:
    cublasHandle_t _cublas_handle;
    cublasLtHandle_t _cublaslt_handle;
    cudnnHandle_t _cudnn_handle;
};

// cuda data type
template <typename ftype>
cudaDataType getCudaDataType();

template <>
cudaDataType getCudaDataType<float>()
{
    return CUDA_R_32F; // 32位浮点数
}

template <>
cudaDataType getCudaDataType<double>()
{
    return CUDA_R_64F; // 64位浮点数
}

template <>
cudaDataType getCudaDataType<__half>()
{
    return CUDA_R_16F; // 16位浮点数（FP16）
}

template <>
cudaDataType getCudaDataType<__nv_bfloat16>()
{
    return CUDA_R_16BF; // 16位 BFloat16（BF16）
}

template <>
cudaDataType getCudaDataType<int>()
{
    return CUDA_R_32I; // 32位整数
}

// cudnn data type
template <typename ftype>
cudnnDataType_t getCudnnDataType();

template <>
cudnnDataType_t getCudnnDataType<float>()
{
    return CUDNN_DATA_FLOAT; // 32位浮点数
}

template <>
cudnnDataType_t getCudnnDataType<double>()
{
    return CUDNN_DATA_DOUBLE; // 64位浮点数
}

template <>
cudnnDataType_t getCudnnDataType<__half>()
{
    return CUDNN_DATA_HALF; // 16位浮点数（FP16）
}

template <>
cudnnDataType_t getCudnnDataType<__nv_bfloat16>()
{
    return CUDNN_DATA_BFLOAT16; // 16位 BFloat16（BF16）
}

template <>
cudnnDataType_t getCudnnDataType<int>()
{
    return CUDNN_DATA_INT32; // 32位整数
}

template <>
cudnnDataType_t getCudnnDataType<short>()
{
    return CUDNN_DATA_INT8; // 8位整数（有符号）
}

template <>
cudnnDataType_t getCudnnDataType<unsigned short>()
{
    return CUDNN_DATA_UINT8; // 8位整数（无符号）
}

const size_t cublaslt_workspace_size = 1024 * 1024;
void *cublaslt_workspace = NULL;

// ----------------------------------------------------------------------------
// random utils
inline void cudaCheck_(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))
float *make_random_float_01(size_t N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float *make_random_float(size_t N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int *make_random_int(size_t N, int V)
{
    int *arr = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float *make_zeros_float(size_t N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float *make_ones_float(size_t N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = 1.0f;
    }
    return arr;
}
// ----------------------------------------------------------------------------
// testing and benchmarking utils

template <class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType *d_ptr, float *h_ptr, size_t count)
{
    // copy from host to device with data type conversion.
    TargetType *converted = (TargetType *)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++)
    {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs &&...kernel_args)
{
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void *flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++)
    {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

template <class D, class T>
void validate_result(D *device_result, const T *cpu_reference, const char *name, std::size_t num_elements, T tolerance = 1e-4)
{
    D *out_gpu = (D *)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++)
    {
        // Skip masked elements
        if (!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5)
        {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff)
        {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults++;
            if (nfaults >= 10)
            {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0)
    {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}