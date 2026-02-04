#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <vector>
#include <cmath>
#include <cstring>

using namespace llaisys;

__C {

    struct LlaisysQwen2Model {
        LlaisysQwen2Meta meta;
        LlaisysQwen2Weights weights;
        llaisysDeviceType_t device;
        int device_id;
        size_t kv_cache_pos; // kv cache 当前位置

        std::vector<tensor_t> k_cache;
        std::vector<tensor_t> v_cache;
    };

    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        auto *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids[0];
        model->kv_cache_pos = 0;

        size_t nlayer = meta->nlayer;

        for (size_t i = 0; i < nlayer; i++) {
            model->k_cache.push_back(Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id));
            model->v_cache.push_back(Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, model->device_id));
        }

        auto &weights = model->weights;
        weights.in_embed = nullptr;
        weights.out_embed = nullptr;
        weights.out_norm_w = nullptr;
        weights.attn_norm_w = new llaisysTensor_t[nlayer]();
        weights.attn_q_w = new llaisysTensor_t[nlayer]();
        weights.attn_q_b = new llaisysTensor_t[nlayer]();
        weights.attn_k_w = new llaisysTensor_t[nlayer]();
        weights.attn_k_b = new llaisysTensor_t[nlayer]();
        weights.attn_v_w = new llaisysTensor_t[nlayer]();
        weights.attn_v_b = new llaisysTensor_t[nlayer]();
        weights.attn_o_w = new llaisysTensor_t[nlayer]();
        weights.mlp_norm_w = new llaisysTensor_t[nlayer]();
        weights.mlp_gate_w = new llaisysTensor_t[nlayer]();
        weights.mlp_up_w = new llaisysTensor_t[nlayer]();
        weights.mlp_down_w = new llaisysTensor_t[nlayer]();

        return model;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        auto &weights = model->weights;
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return &model->weights;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        auto &meta = model->meta;
        auto &w = model->weights;
        auto dtype = meta.dtype;
        auto dev = model->device;
        auto dev_id = model->device_id;
        size_t pos = model->kv_cache_pos;

        auto tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, dev, dev_id);
        tokens->load(token_ids);

        // pos_ids: [pos, pos+1, ..., pos+ntoken-1]
        std::vector<int64_t> pos_data(ntoken);
        for (size_t i = 0; i < ntoken; i++) {
            pos_data[i] = pos + i;
        }
        auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, dev, dev_id);
        pos_ids->load(pos_data.data());

        auto x = Tensor::create({ntoken, meta.hs}, dtype, dev, dev_id);
        ops::embedding(x, tokens, w.in_embed->tensor);

        for (size_t l = 0; l < meta.nlayer; l++) {
            auto x_norm = Tensor::create({ntoken, meta.hs}, dtype, dev, dev_id);
            ops::rms_norm(x_norm, x, w.attn_norm_w[l]->tensor, meta.epsilon);

            auto q = Tensor::create({ntoken, meta.nh * meta.dh}, dtype, dev, dev_id);
            ops::linear(q, x_norm, w.attn_q_w[l]->tensor, w.attn_q_b[l]->tensor);
            auto q_3d = q->view({ntoken, meta.nh, meta.dh});

            auto k = Tensor::create({ntoken, meta.nkvh * meta.dh}, dtype, dev, dev_id);
            ops::linear(k, x_norm, w.attn_k_w[l]->tensor, w.attn_k_b[l]->tensor);
            auto k_3d = k->view({ntoken, meta.nkvh, meta.dh});

            auto v = Tensor::create({ntoken, meta.nkvh * meta.dh}, dtype, dev, dev_id);
            ops::linear(v, x_norm, w.attn_v_w[l]->tensor, w.attn_v_b[l]->tensor);
            auto v_3d = v->view({ntoken, meta.nkvh, meta.dh});

            ops::rope(q_3d, q_3d, pos_ids, meta.theta);
            ops::rope(k_3d, k_3d, pos_ids, meta.theta);

            // 写入 kv cache
            auto k_slice = model->k_cache[l]->slice(0, pos, pos + ntoken);
            auto v_slice = model->v_cache[l]->slice(0, pos, pos + ntoken);
            ops::rearrange(k_slice, k_3d);
            ops::rearrange(v_slice, v_3d);

            auto k_cached = model->k_cache[l]->slice(0, 0, pos + ntoken);
            auto v_cached = model->v_cache[l]->slice(0, 0, pos + ntoken);
            auto attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, dev, dev_id);
            float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
            ops::self_attention(attn_out, q_3d, k_cached, v_cached, scale);

            auto attn_flat = attn_out->view({ntoken, meta.hs});
            auto attn_proj = Tensor::create({ntoken, meta.hs}, dtype, dev, dev_id);
            ops::linear(attn_proj, attn_flat, w.attn_o_w[l]->tensor, nullptr);
            ops::add(x, x, attn_proj);

            // mlp
            auto x_norm2 = Tensor::create({ntoken, meta.hs}, dtype, dev, dev_id);
            ops::rms_norm(x_norm2, x, w.mlp_norm_w[l]->tensor, meta.epsilon);

            auto gate = Tensor::create({ntoken, meta.di}, dtype, dev, dev_id);
            ops::linear(gate, x_norm2, w.mlp_gate_w[l]->tensor, nullptr);
            auto up = Tensor::create({ntoken, meta.di}, dtype, dev, dev_id);
            ops::linear(up, x_norm2, w.mlp_up_w[l]->tensor, nullptr);
            auto swi = Tensor::create({ntoken, meta.di}, dtype, dev, dev_id);
            ops::swiglu(swi, gate, up);

            auto mlp_proj = Tensor::create({ntoken, meta.hs}, dtype, dev, dev_id);
            ops::linear(mlp_proj, swi, w.mlp_down_w[l]->tensor, nullptr);
            ops::add(x, x, mlp_proj);
        }

        // 取最后一个 token
        auto last = x->slice(0, ntoken - 1, ntoken);
        auto x_final = Tensor::create({1, meta.hs}, dtype, dev, dev_id);
        ops::rms_norm(x_final, last, w.out_norm_w->tensor, meta.epsilon);

        auto logits = Tensor::create({1, meta.voc}, dtype, dev, dev_id);
        ops::linear(logits, x_final, w.out_embed->tensor, nullptr);

        auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, dev_id);
        auto max_val = Tensor::create({1}, dtype, dev, dev_id);
        ops::argmax(max_idx, max_val, logits);

        int64_t result;
        std::memcpy(&result, max_idx->data(), sizeof(int64_t));
        model->kv_cache_pos += ntoken;

        return result;
    }
}
