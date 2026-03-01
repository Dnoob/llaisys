#include "embedding_cuda.cuh"
#include <cuda_runtime.h>
#include <cstdint>

__global__ void embedding_kernel(std::byte *out, const std::byte *index, const std::byte *weight, size_t seq_len, size_t row_bytes) {

    // 总共需要复制 seq_len * row_bytes 字节
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * row_bytes;
    if (i >= total) 
        return;

    // 算出当前字节属于第几行、行内第几字节
    size_t row = i / row_bytes;          // 第几个 token
    size_t col = i % row_bytes;          // 行内偏移

    // 从 index 数组读出这一行应该查表的行号
    int64_t idx = reinterpret_cast<const int64_t *>(index)[row];

    // 从 weight 表的第 idx 行复制到 out 的第 row 行
    out[i] = weight[idx * row_bytes + col];
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t seq_len, size_t dim, size_t elem_size, llaisysStream_t stream) {
    size_t row_bytes = dim * elem_size;
    size_t total = seq_len * row_bytes;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    embedding_kernel<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
        out, index, weight, seq_len, row_bytes
    );
}
}