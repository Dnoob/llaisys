#include "self_attention_cuda.cuh"
#include "../../../device/nvidia/cuda_cast.cuh"
#include <cfloat>

template <typename T>
__global__ void self_attention_kernel(
    T *attn_val, const T *q, const T *k, const T *v,
    size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads,
    size_t head_dim, float scale) {

    size_t i = blockIdx.x / num_heads;
    size_t h = blockIdx.x % num_heads;
    size_t group = num_heads / num_kv_heads;
    size_t kv_h = h / group;
    int offset = static_cast<int>(kv_len) - static_cast<int>(query_len);
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const T *q_ptr = q + (i * num_heads + h) * head_dim;
    extern __shared__ float shared[];
    float *score = shared;

    // 阶段 1：Q·K 点积 + causal mask
    for (size_t j = tid; j < kv_len; j += block_size) {
        if (static_cast<int>(j) > static_cast<int>(i) + offset) {
            score[j] = -FLT_MAX;
        } else {
            const T *k_ptr = k + (j * num_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++)
                dot += cuda_cast<float>(q_ptr[d]) * cuda_cast<float>(k_ptr[d]);
            score[j] = dot * scale;
        }
    }
    __syncthreads();

    // 阶段 2：Softmax
    float *reduce_buf = shared + kv_len;
    float local_max = -FLT_MAX;
    for (size_t j = tid; j < kv_len; j += block_size)
        if (score[j] > local_max) local_max = score[j];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && reduce_buf[tid + stride] > reduce_buf[tid])
            reduce_buf[tid] = reduce_buf[tid + stride];
        __syncthreads();
    }
    float max_val = reduce_buf[0];

    float local_sum = 0.0f;
    for (size_t j = tid; j < kv_len; j += block_size) {
        score[j] = expf(score[j] - max_val);
        local_sum += score[j];
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }
    float sum_exp = reduce_buf[0];

    for (size_t j = tid; j < kv_len; j += block_size)
        score[j] /= sum_exp;
    __syncthreads();

    // 阶段 3：加权求和 score × V
    T *out_ptr = attn_val + (i * num_heads + h) * head_dim;
    for (size_t d = tid; d < head_dim; d += block_size) {
        float val = 0.0f;
        for (size_t j = 0; j < kv_len; j++) {
            const T *v_ptr = v + (j * num_kv_heads + kv_h) * head_dim;
            val += score[j] * cuda_cast<float>(v_ptr[d]);
        }
        out_ptr[d] = cuda_cast<T>(val);
    }
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t query_len, size_t kv_len, size_t num_heads, size_t num_kv_heads,
                    size_t head_dim, float scale, llaisysStream_t stream) {
    int grid_size = query_len * num_heads;
    int block_size = 256;
    size_t shared_mem = (kv_len + block_size) * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<<<grid_size, block_size, shared_mem, (cudaStream_t)stream>>>(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<<<grid_size, block_size, shared_mem, (cudaStream_t)stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            query_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
        return;
    default:
        break;
    }
}
}
