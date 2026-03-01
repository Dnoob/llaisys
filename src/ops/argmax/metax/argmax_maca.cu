#include "argmax_maca.cuh"
#include "../../../device/metax/maca_cast.cuh"
#include <mcr/mc_runtime.h>
#include <cmath>

static const int BLOCK_SIZE = 256;
static const int MAX_BLOCKS = 128;

template <typename T>
__global__ void argmax_phase1(float *partial_vals, int64_t *partial_idxs, const T *vals, size_t numel) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    int64_t local_idx = 0;
    for (size_t i = blockIdx.x * blockDim.x + tid; i < numel; i += blockDim.x * gridDim.x) {
        float v = maca_cast<float>(vals[i]);
        if (v > local_max) { local_max = v; local_idx = i; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) { partial_vals[blockIdx.x] = s_val[0]; partial_idxs[blockIdx.x] = s_idx[0]; }
}

__global__ void argmax_phase2(int64_t *max_idx, float *max_val, const float *partial_vals, const int64_t *partial_idxs, int num_blocks) {
    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int64_t s_idx[BLOCK_SIZE];
    int tid = threadIdx.x;

    if (tid < num_blocks) { s_val[tid] = partial_vals[tid]; s_idx[tid] = partial_idxs[tid]; }
    else { s_val[tid] = -INFINITY; s_idx[tid] = 0; }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) { *max_idx = s_idx[0]; *max_val = s_val[0]; }
}

namespace llaisys::ops::metax {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel, llaisysStream_t stream) {
    int grid_size = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid_size > MAX_BLOCKS) grid_size = MAX_BLOCKS;

    float *partial_vals = nullptr;
    int64_t *partial_idxs = nullptr;
    mcMalloc(&partial_vals, grid_size * sizeof(float));
    mcMalloc(&partial_idxs, grid_size * sizeof(int64_t));
    mcStream_t s = (mcStream_t)stream;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_phase1<<<grid_size, BLOCK_SIZE, 0, s>>>(
            partial_vals, partial_idxs, reinterpret_cast<const float *>(vals), numel);

        argmax_phase2<<<1, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
            partial_vals, partial_idxs, grid_size);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_phase1<<<grid_size, BLOCK_SIZE, 0, s>>>(
            partial_vals, partial_idxs, reinterpret_cast<const __maca_bfloat16 *>(vals), numel);

        argmax_phase2<<<1, BLOCK_SIZE, 0, s>>>(
            reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
            partial_vals, partial_idxs, grid_size);
        break;
    default:
        break;
    }

    mcFree(partial_vals);
    mcFree(partial_idxs);
}
}
