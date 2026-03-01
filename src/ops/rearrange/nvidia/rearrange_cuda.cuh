#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, size_t size, llaisysStream_t stream);
}
