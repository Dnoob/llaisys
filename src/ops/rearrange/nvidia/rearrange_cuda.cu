#include "rearrange_cuda.cuh"
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, size_t size, llaisysStream_t stream) {
    cudaMemcpyAsync(out, in, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
}
}
