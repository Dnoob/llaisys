#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {
void rearrange(std::byte *out, const std::byte *in, size_t size, llaisysStream_t stream);
}
