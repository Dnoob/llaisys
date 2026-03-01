#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// CUDA 设备端类型转换
template <typename To, typename From>
__device__ To cuda_cast(From v);

// float → float
template <> __device__ inline float cuda_cast<float, float>(float v) { return v; }

// bf16 ↔ float
template <> __device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float v) { return __float2bfloat16(v); }

// fp16 ↔ float
template <> __device__ inline float cuda_cast<float, __half>(__half v) { return __half2float(v); }
template <> __device__ inline __half cuda_cast<__half, float>(float v) { return __float2half(v); }
