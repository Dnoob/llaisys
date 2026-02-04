#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, std::byte *weight, llaisysDataType_t type, size_t rows, size_t cols, float eps);
}