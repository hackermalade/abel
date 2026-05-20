#pragma once

#include "ggml.h"
#include <vector>

// ---------------------------------------------------------------------------
//  Expert weight pair (for each MoE expert)
// ---------------------------------------------------------------------------
struct NyxExpert {
    struct ggml_tensor* w1 = nullptr;   // [inter_dim, D]
    struct ggml_tensor* w2 = nullptr;   // [D, inter_dim]
};

// ---------------------------------------------------------------------------
//  Single transformer layer
// ---------------------------------------------------------------------------
struct NyxLayer {
    // Attention
    struct ggml_tensor* wq     = nullptr;   // [q_lora_rank, D]      (low‑rank query)
    struct ggml_tensor* wq_b   = nullptr;   // [n_heads * qk_head_dim, q_lora_rank]
    struct ggml_tensor* wkv_a  = nullptr;   // [kv_lora_rank + qk_rope_head_dim, D]
    struct ggml_tensor* wkv_b  = nullptr;   // [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    struct ggml_tensor* wo     = nullptr;   // [D, n_heads * v_head_dim]

    // Dense feed‑forward (used for first n_dense_layers)
    struct ggml_tensor* ffn_w1 = nullptr;   // [inter_dim, D]   (gate projection)
    struct ggml_tensor* ffn_w2 = nullptr;   // [D, inter_dim]   (output projection)
    struct ggml_tensor* ffn_w3 = nullptr;   // [inter_dim, D]   (up projection – not used in current MLP)

    // MoE (used for layers >= n_dense_layers)
    std::vector<NyxExpert> experts;
    struct ggml_tensor* gate_weight = nullptr;   // [n_experts, D]
};

// ---------------------------------------------------------------------------
//  Full NyxTransformer model (GGML tensors)
// ---------------------------------------------------------------------------
struct NyxTransformer {
    // Embedding & output
    struct ggml_tensor* embed_weight     = nullptr;   // [dim, vocab_size]
    struct ggml_tensor* pos_emb_weight   = nullptr;   // [dim, max_seq_len]
    struct ggml_tensor* final_norm_weight= nullptr;   // [dim]
    struct ggml_tensor* lm_head_weight   = nullptr;   // [vocab_size, dim]
    struct ggml_tensor* freqs_cis        = nullptr;   // [max_seq_len, qk_rope_head_dim/2]

    // Layers
    std::vector<NyxLayer> layers;

    // Architecture hyper‑parameters
    int n_layers            = 0;
    int n_dense_layers      = 0;
    int n_heads             = 0;
    int n_activated_experts = 0;
    float route_scale       = 1.0f;
    int qk_nope_head_dim    = 0;
    int qk_rope_head_dim    = 0;
    int v_head_dim          = 0;
    int kv_lora_rank        = 0;
    int q_lora_rank         = 0;
    int dim                 = 0;
    int vocab_size          = 0;
    int max_seq_len         = 0;

    // Constructor & destructor
    NyxTransformer();
    ~NyxTransformer();
};
