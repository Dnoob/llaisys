#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t seq_len, size_t dim, size_t elem_size, llaisysStream_t stream);
}