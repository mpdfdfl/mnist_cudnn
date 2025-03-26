#pragma once

#include <assert.h>
#include "helper.h"
// 计算 a^(T/N_T) * b^(T/N_T) = c
// a^(T/N_T) (m,k) b^(T/N_T) (k,n) c(m,n)  //这个不一定
// 其中a b都是行主序
// 输出的c也要是行主序
// 注意mnk是转置过后的mnk

// cudnn data type
template <typename ftype>
cublasComputeType_t getCublasComputeType();

template <>
cublasComputeType_t getCublasComputeType<float>()
{
    return CUBLAS_COMPUTE_32F; // 32位浮点数
}

template <>
cublasComputeType_t getCublasComputeType<__half>()
{
    return CUBLAS_COMPUTE_32F_FAST_16F; // 16位浮点数
}

template <>
cublasComputeType_t getCublasComputeType<__nv_bfloat16>()
{
    return CUBLAS_COMPUTE_32F_FAST_16BF; // 32位浮点数
}

template <typename ftype>
void matmul_cublaslt(cublasLtHandle_t handle,
                     ftype *a, ftype *b, ftype *c,
                     int m, int n, int k,
                     bool hasbias = false,                     // 是否在原结果上累加
                     bool transA = false, bool transB = false, // 设置a，b需不需要转置
                     cudaStream_t stream = 0)
{

    // pass1: 创建矩阵乘法句柄
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc,                // 矩阵乘法句柄
                                         getCublasComputeType<ftype>(), // 计算模式设置为fp32这样计算的时候可以提升精度
                                         getCudaDataType<ftype>()));
    // pass2:设置计算偏好
    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;     // 存储计算偏好，用于设定 cuBLASLt 计算的策略，例如最大工作空间等。
    cublasLtMatmulHeuristicResult_t heuristic; // 存储启发式搜索出的最优计算方法，用于决定 cuBLASLt 应该如何执行矩阵乘法。

    // pass3:设置A B 矩阵是否转置
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // pass4:设置矩阵布局
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t CLayout;
    cublasLtMatrixLayout_t DLayout;
    if (transA)
    {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, getCudaDataType<ftype>(), m, k, m));
    }
    else
    {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, getCudaDataType<ftype>(), k, m, k));
    }

    if (transB)
    {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, getCudaDataType<ftype>(), k, n, k));
    }
    else
    {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, getCudaDataType<ftype>(), n, k, n));
    }

    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, getCudaDataType<ftype>(), n, m, n));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, getCudaDataType<ftype>(), n, m, n));

    // pass4:设置工作区域大小，用于寻找更好的优化策略
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, // 设置工作区域大小，用于寻找更好的优化策略
                                                     sizeof(cublaslt_workspace_size)));

    // pass5: 设置标量类型
    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // pass6: 寻找计算方法
    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, BLayout, ALayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);

    // pass7:设置标量值
    float alpha = 1.0f, beta = 0.0f;

    // pass8:进行计算
    // call the matmul
    if (hasbias)
    {
        beta = 1.0f;
    }
    cublasCheck(cublasLtMatmul(handle, operationDesc,
                               &alpha,
                               b, BLayout,
                               a, ALayout,
                               &beta,
                               c, CLayout,
                               c, DLayout,
                               &heuristic.algo,
                               cublaslt_workspace, cublaslt_workspace_size, stream)); // 记得分配cublaslt_workspace的空间

    // pass9:销毁句柄
    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
}