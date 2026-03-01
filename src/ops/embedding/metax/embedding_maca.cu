#include "embedding_maca.cuh"
#include <mcr/mc_runtime.h>
#include <cstdint>

__global__ void embedding_kernel(std::byte *out, const std::byte *index, const std::byte *weight, size_t seq_len, size_t row_bytes) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * row_bytes;
    if (i >= total)
        return;

    size_t row = i / row_bytes;
    size_t col = i % row_bytes;

    int64_t idx = reinterpret_cast<const int64_t *>(index)[row];

    out[i] = weight[idx * row_bytes + col];
}

namespace llaisys::ops::metax {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t seq_len, size_t dim, size_t elem_size, llaisysStream_t stream) {
    size_t row_bytes = dim * elem_size;
    size_t total = seq_len * row_bytes;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    embedding_kernel<<<grid_size, block_size, 0, (mcStream_t)stream>>>(
        out, index, weight, seq_len, row_bytes
    );
}
}
