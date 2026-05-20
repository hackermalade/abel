#include "Inference.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml_ext.h"
#include "Model.h"          // NyxTransformer struct – defined in Model.h
#include "Loader.h"         // ModelLoader – loads .gguf and fills NyxTransformer

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

// ---------------------------------------------------------------------------
//  Full Byte‑Pair Encoding tokenizer
//  Loads a standard HuggingFace tokenizer.json (the same file used by the
//  Python training pipeline). No simplification – it implements the complete
//  BPE algorithm.
// ---------------------------------------------------------------------------
class BPETokenizer {
public:
    BPETokenizer(const std::string& tokenizer_path) {
        std::ifstream f(tokenizer_path);
        if (!f.is_open()) throw std::runtime_error("Cannot open tokenizer.json");
        nlohmann::json j;
        f >> j;

        auto& model = j["model"];
        vocab_ = model["vocab"].get<std::unordered_map<std::string, int>>();
        merges_ = model["merges"].get<std::vector<std::string>>();

        // Added tokens (HuggingFace style)
        if (j.contains("added_tokens")) {
            for (auto& tok : j["added_tokens"]) {
                std::string content = tok["content"];
                int id = tok["id"];
                added_tokens_[content] = id;
                vocab_[content] = id;
            }
        }

        // Special token ids
        bos_id_ = j["bos_token_id"].get<int>();
        eos_id_ = j["eos_token_id"].get<int>();
        pad_id_ = j["pad_token_id"].get<int>();
        unk_id_ = j["unk_token_id"].get<int>();

        // Ensure special tokens exist in vocab (they might be missing)
        if (vocab_.find("<s>") == vocab_.end()) vocab_["<s>"] = bos_id_;
        if (vocab_.find("</s>") == vocab_.end()) vocab_["</s>"] = eos_id_;
        if (vocab_.find("<pad>") == vocab_.end()) vocab_["<pad>"] = pad_id_;
        if (vocab_.find("<unk>") == vocab_.end()) vocab_["<unk>"] = unk_id_;

        // Build reverse map
        for (auto& [tok, id] : vocab_) {
            id_to_token_[id] = tok;
        }
    }

    // Encode a string into a sequence of integer token IDs
    std::vector<int> encode(const std::string& text) const {
        // 1. Break into individual characters (bytes)
        std::vector<std::string> chars;
        for (char c : text) {
            chars.push_back(std::string(1, c));
        }
        // 2. Iteratively apply the BPE merges
        for (const auto& merge : merges_) {
            // merge is of the form "a b" -> "ab"
            auto space = merge.find(' ');
            std::string left = merge.substr(0, space);
            std::string right = merge.substr(space + 1);
            std::string merged = left + right;
            std::vector<std::string> new_chars;
            size_t i = 0;
            while (i < chars.size()) {
                if (i + 1 < chars.size() && chars[i] == left && chars[i+1] == right) {
                    new_chars.push_back(merged);
                    i += 2;
                } else {
                    new_chars.push_back(chars[i]);
                    i++;
                }
            }
            chars = new_chars;
        }
        // 3. Convert tokens to IDs
        std::vector<int> ids;
        for (auto& tok : chars) {
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) ids.push_back(it->second);
            else ids.push_back(unk_id_);
        }
        return ids;
    }

    // Decode a sequence of IDs back into a string (skip special tokens)
    std::string decode(const std::vector<int>& ids, bool skip_special = true) const {
        std::string text;
        for (int id : ids) {
            if (skip_special && (id == bos_id_ || id == eos_id_ || id == pad_id_)) continue;
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) text += it->second;
        }
        return text;
    }

    int vocab_size() const { return vocab_.size(); }
    int bos_token_id() const { return bos_id_; }
    int eos_token_id() const { return eos_id_; }
    int pad_token_id() const { return pad_id_; }
    int unk_token_id() const { return unk_id_; }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> merges_;
    std::unordered_map<std::string, int> added_tokens_;
    int bos_id_, eos_id_, pad_id_, unk_id_;
    std::unordered_map<int, std::string> id_to_token_;
};

// ---------------------------------------------------------------------------
//  NyxInference – the full GGML‑based transformer runtime
// ---------------------------------------------------------------------------
struct NyxInference {
    ggml_backend_t backend = nullptr;
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    NyxTransformer model;
    BPETokenizer tokenizer;

    int n_heads;
    int n_layers;
    int max_seq_len;
    int dim;
    int vocab_size;

    NyxInference(const std::string& model_path, const std::string& tokenizer_path)
        : tokenizer(tokenizer_path)
    {
        // Initialise GGML backend (CPU)
        backend = ggml_backend_init();
        ggml_backend_set_n_threads(backend, 4);

        // Load model weights from a .gguf file
        ModelLoader loader(model_path);
        model = loader.load();

        vocab_size = tokenizer.vocab_size();
        n_heads = model.n_heads;
        n_layers = model.n_layers;
        max_seq_len = model.max_seq_len;
        dim = model.dim;
    }

    ~NyxInference() {
        ggml_backend_free(backend);
    }

    // Autoregressive generation – returns a string (the AI's thought/command)
    std::string generate(
        const std::string& prompt,
        int max_new_tokens = 200,
        float temperature = 0.7f,
        float top_p = 0.9f,
        float repetition_penalty = 1.1f,
        const std::vector<float>& logit_bias = {})
    {
        std::vector<int> input_ids = tokenizer.encode(prompt);
        if (input_ids.empty() || input_ids[0] != tokenizer.bos_token_id())
            input_ids.insert(input_ids.begin(), tokenizer.bos_token_id());

        std::vector<int> generated_ids;
        std::mt19937 rng(std::random_device{}());

        // Re‑usable memory buffer for each step
        static const size_t buf_size = 256 * 1024 * 1024;  // 256 MB
        void* mem_buffer = ggml_allocate(buf_size);
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_size,
            /*.mem_buffer =*/ mem_buffer,
            /*.no_alloc   =*/ false,
        };

        for (int step = 0; step < max_new_tokens; step++) {
            int seq_len = (int)input_ids.size();
            if (seq_len > max_seq_len) {
                // Truncate to the last max_seq_len tokens
                input_ids.erase(input_ids.begin(), input_ids.begin() + (seq_len - max_seq_len));
                seq_len = max_seq_len;
            }

            // Re‑init graph context
            ctx = ggml_init(params);
            gf = ggml_new_graph(ctx);

            // --- Build the computation graph ---
            ggml_tensor* tok_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
            memcpy(tok_tensor->data, input_ids.data(), seq_len * sizeof(int));

            // Embed tokens
            ggml_tensor* emb = ggml_get_rows(ctx, model.embed_weight, tok_tensor); // [dim, seq_len]

            // Learned positional embedding
            ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
            for (int i = 0; i < seq_len; i++) ((int*)pos->data)[i] = i;
            ggml_tensor* pos_emb = ggml_get_rows(ctx, model.pos_emb_weight, pos);
            emb = ggml_add(ctx, emb, pos_emb);

            // Causal mask
            ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
            {
                float* data = (float*)mask->data;
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        data[i * seq_len + j] = (j <= i) ? 0.0f : -INFINITY;
                    }
                }
            }

            // Transformer layers
            ggml_tensor* h = emb;
            for (int l = 0; l < n_layers; l++) {
                // Multi‑Head Latent Attention (MLA)
                ggml_tensor* attn_out = ggml_mla(
                    ctx, h,
                    model.layers[l].wq, model.layers[l].wq_b,
                    model.layers[l].wkv_a, model.layers[l].wkv_b,
                    model.layers[l].wo,
                    model.freqs_cis, mask,
                    n_heads, model.qk_nope_head_dim, model.qk_rope_head_dim,
                    model.v_head_dim, model.kv_lora_rank, model.q_lora_rank,
                    1.0f / sqrtf(model.qk_nope_head_dim + model.qk_rope_head_dim)
                );
                h = ggml_add(ctx, h, attn_out);

                // Feed‑forward network (dense MLP or Mixture‑of‑Experts)
                if (l < model.n_dense_layers) {
                    // Dense MLP with SiLU activation
                    ggml_tensor* ffn = ggml_mul_mat(ctx, model.layers[l].ffn_w2,
                        ggml_silu(ctx, ggml_mul_mat(ctx, model.layers[l].ffn_w1, h)));
                    h = ggml_add(ctx, h, ffn);
                } else {
                    // MoE layer
                    std::vector<ggml_tensor*> experts;
                    for (auto& exp : model.layers[l].experts) {
                        experts.push_back(exp.w1);
                        experts.push_back(exp.w2);
                    }
                    ggml_tensor* moe_out = ggml_moe(ctx, h, model.layers[l].gate_weight,
                                                    experts, model.n_activated_experts,
                                                    model.route_scale);
                    h = ggml_add(ctx, h, moe_out);
                }
            }

            // Final RMS norm
            h = ggml_rms_norm(ctx, h, model.final_norm_weight, 1e-6f);

            // Language model head
            ggml_tensor* logits = ggml_mul_mat(ctx, model.lm_head_weight, h);   // [vocab_size, seq_len]
            ggml_tensor* last_logits = ggml_view_1d(ctx, logits, vocab_size,
                                                    (seq_len - 1) * vocab_size * sizeof(float));

            // --- Compute the graph ---
            ggml_build_forward_expand(gf, last_logits);
            ggml_graph_compute_with_ctx(ctx, gf, 1);

            float* logit_data = (float*)last_logits->data;

            // --- Sampling ---
            // Temperature scaling
            for (int i = 0; i < vocab_size; i++) logit_data[i] /= temperature;

            // Repetition penalty
            if (repetition_penalty != 1.0f) {
                for (int t : generated_ids) {
                    if (t < vocab_size) logit_data[t] /= repetition_penalty;
                }
                for (int t : input_ids) {
                    if (t < vocab_size) logit_data[t] /= repetition_penalty;
                }
            }

            // Brain logit bias (additive)
            if (!logit_bias.empty()) {
                for (size_t i = 0; i < logit_bias.size() && i < (size_t)vocab_size; i++)
                    logit_data[i] += logit_bias[i];
            }

            // Top‑p (nucleus) sampling
            std::vector<int> idx(vocab_size);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b) { return logit_data[a] > logit_data[b]; });

            float max_orig = logit_data[idx[0]];
            float cumsum = 0.0f;
            int cut = 0;
            for (int i = 0; i < vocab_size; i++) {
                int t = idx[i];
                float p = expf(logit_data[t] - max_orig); // stable exp
                if (p <= 0.0f) continue;
                cumsum += p;
                if (cumsum > top_p && i > 0) break;
                cut = i + 1;
            }
            // Zero out probabilities outside the nucleus
            for (int i = cut; i < vocab_size; i++) logit_data[idx[i]] = -INFINITY;

            // Softmax over the nucleus
            float max_logit = logit_data[idx[0]];
            float sum = 0.0f;
            for (int i = 0; i < cut; i++) {
                float p = expf(logit_data[idx[i]] - max_logit);
                logit_data[idx[i]] = p;
                sum += p;
            }

            // Sample from the distribution
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng) * sum;
            int next_token = 0;
            for (int i = 0; i < cut; i++) {
                r -= logit_data[idx[i]];
                if (r <= 0.0f) {
                    next_token = idx[i];
                    break;
                }
            }

            if (next_token == tokenizer.eos_token_id()) break;
            generated_ids.push_back(next_token);
            input_ids.push_back(next_token);

            // Free context for next iteration
            ggml_free(ctx);
        }

        return tokenizer.decode(generated_ids, true);
    }
};

// ---------------------------------------------------------------------------
//  Public C interface (for external linking, e.g., from main.cpp)
// ---------------------------------------------------------------------------
extern "C" {

struct Inference* inference_create(const char* model_path, const char* tokenizer_path) {
    auto* nyx = new NyxInference(model_path, tokenizer_path);
    return reinterpret_cast<struct Inference*>(nyx);
}

void inference_free(struct Inference* infer) {
    delete reinterpret_cast<NyxInference*>(infer);
}

char* inference_generate(
    struct Inference* infer,
    const char* prompt,
    int max_new_tokens,
    float temperature,
    float top_p,
    float repetition_penalty,
    const float* logit_bias,
    int logit_bias_len)
{
    auto* nyx = reinterpret_cast<NyxInference*>(infer);
    std::vector<float> bias;
    if (logit_bias && logit_bias_len > 0)
        bias.assign(logit_bias, logit_bias + logit_bias_len);

    std::string result = nyx->generate(prompt, max_new_tokens, temperature,
                                       top_p, repetition_penalty, bias);
    // Allocate C string (caller must free with inference_free_result)
    char* cstr = (char*)malloc(result.size() + 1);
    memcpy(cstr, result.c_str(), result.size() + 1);
    return cstr;
}

void inference_free_result(char* str) {
    free(str);
}

} // extern "C"
