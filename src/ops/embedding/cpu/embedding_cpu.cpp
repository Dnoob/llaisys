#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t seq_len, size_t dim, size_t elem_size) {
    const int64_t *idx_ptr = reinterpret_cast<const int64_t * >(index);
    size_t row_bytes = dim * elem_size;
    for (size_t i = 0; i < seq_len; i++) {
        std::memcpy(out + i * row_bytes, weight + idx_ptr[i] * row_bytes, row_bytes);
    }
}
}