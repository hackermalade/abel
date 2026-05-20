#include "Model.h"
#include "ggml.h"

// ---------------------------------------------------------------------------
//  NyxTransformer constructor – zeros everything
// ---------------------------------------------------------------------------
NyxTransformer::NyxTransformer() {
    embed_weight = nullptr;
    pos_emb_weight = nullptr;
    final_norm_weight = nullptr;
    lm_head_weight = nullptr;
    freqs_cis = nullptr;
    n_layers = 0;
    n_dense_layers = 0;
    n_heads = 0;
    n_activated_experts = 0;
    route_scale = 1.0f;
    qk_nope_head_dim = 0;
    qk_rope_head_dim = 0;
    v_head_dim = 0;
    kv_lora_rank = 0;
    q_lora_rank = 0;
    dim = 0;
    vocab_size = 0;
    max_seq_len = 0;
}

// ---------------------------------------------------------------------------
//  Destructor – frees all GGML tensors owned by this model
// ---------------------------------------------------------------------------
NyxTransformer::~NyxTransformer() {
    auto free_tensor = [](ggml_tensor*& t) {
        if (t) {
            // Tensors are allocated within a GGML context; we assume the context
            // is managed externally (e.g., by the inference runtime). To avoid
            // double‑free, we simply set the pointer to null. In a production
            // system, i'd later coordinate with the GGML context lifecycle.
            t = nullptr;
        }
    };

    free_tensor(embed_weight);
    free_tensor(pos_emb_weight);
    free_tensor(final_norm_weight);
    free_tensor(lm_head_weight);
    free_tensor(freqs_cis);

    for (auto& layer : layers) {
        free_tensor(layer.wq);
        free_tensor(layer.wq_b);
        free_tensor(layer.wkv_a);
        free_tensor(layer.wkv_b);
        free_tensor(layer.wo);
        free_tensor(layer.ffn_w1);
        free_tensor(layer.ffn_w2);
        free_tensor(layer.ffn_w3);
        free_tensor(layer.gate_weight);
        for (auto& exp : layer.experts) {
            free_tensor(exp.w1);
            free_tensor(exp.w2);
        }
    }
}
