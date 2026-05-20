#include "Loader.h"
#include "Model.h"
#include "ggml.h"

#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <unordered_map>
#include <algorithm>

// ---------------------------------------------------------------------------
//  Minimal GGUF reader – no external dependencies, follows GGUF v3 spec
// ---------------------------------------------------------------------------
namespace {

enum gguf_type : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_tensor_info {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4];
    gguf_type type;
    uint64_t offset;
};

// Simple binary reader helpers
static uint32_t read_u32(std::istream& s) {
    uint32_t v; s.read(reinterpret_cast<char*>(&v), sizeof(v)); return v;
}
static uint64_t read_u64(std::istream& s) {
    uint64_t v; s.read(reinterpret_cast<char*>(&v), sizeof(v)); return v;
}
static float read_f32(std::istream& s) {
    float v; s.read(reinterpret_cast<char*>(&v), sizeof(v)); return v;
}
static std::string read_string(std::istream& s) {
    uint64_t len = read_u64(s);
    std::string str(len, '\0');
    s.read(str.data(), len);
    return str;
}

static void skip_padding(std::istream& s, uint64_t offset) {
    uint64_t pos = s.tellg();
    if (pos > offset) throw std::runtime_error("Read past expected offset");
    if (pos < offset) s.seekg(offset, std::ios::beg);
}

// Align offset to `align` bytes
static uint64_t align_offset(uint64_t offset, uint64_t align) {
    return (offset + align - 1) / align * align;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  ModelLoader implementation
// ---------------------------------------------------------------------------
ModelLoader::ModelLoader(const std::string& path) : path_(path) {}

NyxTransformer ModelLoader::load() {
    std::ifstream file(path_, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open model file: " + path_);

    // --- Header ---
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x46554747) // 'GGUF' in little‑endian
        throw std::runtime_error("Not a valid GGUF file");

    uint32_t version = read_u32(file);
    if (version < 2 || version > 3)
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version));

    uint64_t n_tensors = read_u64(file);
    uint64_t n_kv = read_u64(file);

    // --- Read metadata (key‑value pairs) ---
    std::unordered_map<std::string, int32_t> int32_values;
    std::unordered_map<std::string, float> float_values;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = read_string(file);
        gguf_type type = static_cast<gguf_type>(read_u32(file));
        switch (type) {
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
                int32_values[key] = read_u32(file);
                break;
            case GGUF_TYPE_FLOAT32:
                float_values[key] = read_f32(file);
                break;
            case GGUF_TYPE_BOOL:
                int32_values[key] = read_u32(file); // treat as int
                break;
            case GGUF_TYPE_UINT8:  int32_values[key] = static_cast<uint8_t>(file.get()); break;
            case GGUF_TYPE_INT8:   int32_values[key] = static_cast<int8_t>(file.get()); break;
            case GGUF_TYPE_STRING: read_string(file); break;   // ignore strings for now
            default:
                throw std::runtime_error("Unsupported GGUF metadata type for key: " + key);
        }
    }

    // Extract architecture parameters
    NyxTransformer model;
    model.vocab_size    = int32_values.at("nyx.vocab_size");
    model.dim           = int32_values.at("nyx.dim");
    model.n_layers      = int32_values.at("nyx.n_layers");
    model.n_heads       = int32_values.at("nyx.n_heads");
    model.max_seq_len   = int32_values.at("nyx.max_seq_len");
    model.n_dense_layers = int32_values.count("nyx.n_dense_layers") ? int32_values["nyx.n_dense_layers"] : 1;
    model.n_activated_experts = int32_values.count("nyx.n_activated_experts") ? int32_values["nyx.n_activated_experts"] : 2;
    model.route_scale   = float_values.count("nyx.route_scale") ? float_values["nyx.route_scale"] : 1.0f;
    model.qk_nope_head_dim = int32_values.count("nyx.qk_nope_head_dim") ? int32_values["nyx.qk_nope_head_dim"] : 128;
    model.qk_rope_head_dim = int32_values.count("nyx.qk_rope_head_dim") ? int32_values["nyx.qk_rope_head_dim"] : 64;
    model.v_head_dim    = int32_values.count("nyx.v_head_dim") ? int32_values["nyx.v_head_dim"] : 128;
    model.kv_lora_rank  = int32_values.count("nyx.kv_lora_rank") ? int32_values["nyx.kv_lora_rank"] : 512;
    model.q_lora_rank   = int32_values.count("nyx.q_lora_rank") ? int32_values["nyx.q_lora_rank"] : 0;

    // --- Read tensor infos ---
    std::vector<gguf_tensor_info> tensors;
    tensors.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        gguf_tensor_info ti;
        ti.name = read_string(file);
        ti.n_dims = read_u32(file);
        for (uint32_t j = 0; j < ti.n_dims; j++) {
            ti.ne[j] = read_u64(file);
        }
        for (uint32_t j = ti.n_dims; j < 4; j++) {
            ti.ne[j] = 1; // broadcast
        }
        ti.type = static_cast<gguf_type>(read_u32(file));
        ti.offset = read_u64(file);
        tensors.push_back(ti);
    }

    // Align to tensor data start (usually 32‑byte aligned)
    uint64_t data_start = align_offset(file.tellg(), 32);
    skip_padding(file, data_start);

    // --- Helper to read a tensor ---
    auto read_tensor = [&](const std::string& name) -> ggml_tensor* {
        for (const auto& ti : tensors) {
            if (ti.name != name) continue;
            // Calculate number of elements
            uint64_t nelements = 1;
            for (int d = 0; d < 4; d++) nelements *= ti.ne[d];
            size_t size = nelements * sizeof(float);   // assume all tensors are float32 for now
            std::vector<float> data(nelements);

            // Seek to offset
            skip_padding(file, ti.offset);
            file.read(reinterpret_cast<char*>(data.data()), size);

            // Create a GGML tensor (from user‑provided context? We'll use a temporary global context)
            // In a real system, you'd have a ggml context; we'll allocate a new one.
            // For simplicity, we allocate a dedicated context per tensor and copy data.
            static struct ggml_init_params ctx_params = { 1024*1024, NULL, true }; // no alloc
            struct ggml_context* tmp_ctx = ggml_init(ctx_params);
            ggml_tensor* t = ggml_new_tensor_4d(tmp_ctx, GGML_TYPE_F32,
                                                ti.ne[0], ti.ne[1], ti.ne[2], ti.ne[3]);
            memcpy(t->data, data.data(), size);
            return t;
        }
        return nullptr;
    };

    // --- Map tensor names to NyxTransformer fields ---
    model.embed_weight = read_tensor("token_embd.weight");
    model.pos_emb_weight = read_tensor("pos_emb.weight");
    model.final_norm_weight = read_tensor("output_norm.weight");
    model.lm_head_weight = read_tensor("output.weight");

    // Layers
    model.layers.resize(model.n_layers);
    for (int l = 0; l < model.n_layers; l++) {
        auto& layer = model.layers[l];
        std::string prefix = "blk." + std::to_string(l) + ".";
        layer.wq      = read_tensor(prefix + "attn_q.weight");       // low‑rank A
        layer.wq_b    = read_tensor(prefix + "attn_q_b.weight");     // low‑rank B
        layer.wkv_a   = read_tensor(prefix + "attn_kv_a_mqa.weight");
        layer.wkv_b   = read_tensor(prefix + "attn_kv_b_mqa.weight");
        layer.wo      = read_tensor(prefix + "attn_output.weight");

        layer.ffn_w1  = read_tensor(prefix + "ffn_gate.weight");
        layer.ffn_w2  = read_tensor(prefix + "ffn_down.weight");
        layer.ffn_w3  = read_tensor(prefix + "ffn_up.weight");       // not used in current MLP, but kept

        if (l >= model.n_dense_layers) {
            // MoE layer: load experts
            int n_experts = int32_values["nyx.n_routed_experts"];
            layer.experts.resize(n_experts);
            for (int e = 0; e < n_experts; e++) {
                std::string exp_prefix = prefix + "experts." + std::to_string(e) + ".";
                layer.experts[e].w1 = read_tensor(exp_prefix + "w1.weight");
                layer.experts[e].w2 = read_tensor(exp_prefix + "w2.weight");
            }
            layer.gate_weight = read_tensor(prefix + "gate.weight");
        }
    }

    // Precomputed RoPE frequencies (complex values stored as pairs)
    model.freqs_cis = read_tensor("freqs_cis");

    return model;
}
