#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Custom GGML operations ──────────────────────────────────────────────

// Root‑mean‑square normalization
struct ggml_tensor* ggml_rms_norm(
    struct ggml_context* ctx,
    struct ggml_tensor*  x,
    struct ggml_tensor*  weight,
    float eps);

// Rotary position embedding with YARN‑style frequency scaling
struct ggml_tensor* ggml_rope_yarn(
    struct ggml_context* ctx,
    struct ggml_tensor*  x,            // [S, qk_rope_head_dim/2, n_heads, B]
    struct ggml_tensor*  freqs_cis,    // [S, qk_rope_head_dim/2]
    int                  n_dims);      // = qk_rope_head_dim/2

// Multi‑Head Latent Attention (MLA)
struct ggml_tensor* ggml_mla(
    struct ggml_context* ctx,
    struct ggml_tensor*  x,            // [D, S, B]
    struct ggml_tensor*  wq,           // [q_lora_rank, D]
    struct ggml_tensor*  wq_b,         // [n_heads * qk_head_dim, q_lora_rank]
    struct ggml_tensor*  wkv_a,        // [kv_lora_rank + qk_rope_head_dim, D]
    struct ggml_tensor*  wkv_b,        // [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    struct ggml_tensor*  wo,           // [D, n_heads * v_head_dim]
    struct ggml_tensor*  freqs_cis,    // [S, qk_rope_head_dim/2]
    struct ggml_tensor*  mask,         // causal mask [S, S]
    int n_heads,
    int qk_nope_head_dim,
    int qk_rope_head_dim,
    int v_head_dim,
    int kv_lora_rank,
    int q_lora_rank,
    float softmax_scale);

// Mixture‑of‑Experts
struct ggml_tensor* ggml_moe(
    struct ggml_context* ctx,
    struct ggml_tensor*  x,                // [n_tokens, D]
    struct ggml_tensor*  gate_weight,      // [n_experts, D]
    const std::vector<struct ggml_tensor*>& experts,  // list of expert weight pairs (w1,w2)
    int topk,
    float route_scale);

// Register all custom operations with the GGML backend
void ggml_ext_register_custom_ops(void);

#ifdef __cplusplus
}
#endif
