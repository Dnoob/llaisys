#pragma once
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>

// MACA 设备端类型转换
template <typename To, typename From>
__device__ To maca_cast(From v);

// float → float
template <> __device__ inline float maca_cast<float, float>(float v) { return v; }

// bf16 ↔ float
template <> __device__ inline float maca_cast<float, __maca_bfloat16>(__maca_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ inline __maca_bfloat16 maca_cast<__maca_bfloat16, float>(float v) { return __float2bfloat16(v); }

// fp16 ↔ float
template <> __device__ inline float maca_cast<float, __half>(__half v) { return __half2float(v); }
template <> __device__ inline __half maca_cast<__half, float>(float v) { return __float2half(v); }
