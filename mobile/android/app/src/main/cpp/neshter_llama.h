// Neshter's llama.cpp engine: any GGUF model, chat, per-head power and attention, and head cuts.
// Plain C++ (no JNI) so it can be built and tested on a desktop as well as on Android.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_context;
struct ggml_tensor;

namespace nl {

struct Info {
    std::string name;   // general.name, or the file name
    std::string arch;   // general.architecture (llama, qwen2, gemma3, ...)
    std::string desc;   // llama_model_desc: "qwen2 0.5B Q8_0"
    int n_layer = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int n_embd = 0;
    int n_ff = 0;       // <arch>.feed_forward_length (0 when the file does not say)
    int head_dim = 0;
    int n_vocab = 0;
    int n_ctx_train = 0;
    uint64_t size_bytes = 0;
    uint64_t n_params = 0;
    bool has_template = false;
    bool heads_supported = false;  // the graph exposes "kqv_out-N" with n_head equal slices (not MLA, not recurrent)
};

struct ChatMessage {
    std::string role;  // system, user, assistant
    std::string content;
};

struct Generation {
    std::string text;
    std::vector<int32_t> tokens;
    int n_prompt = 0;
    bool stopped_by_eog = false;
    double prompt_ms = 0;
    double gen_ms = 0;
};

struct Analysis {
    int n_tokens = 0;
    std::vector<float> power;   // [n_layer][n_head][n_tokens]: L2 norm of each head's output before the output projection
    std::vector<float> attn;    // [n_layer][n_head][n_tokens][n_tokens] when asked for; rows are queries
    int logprob_rows = 0;              // the last `logprob_rows` positions
    std::vector<float> logprobs;       // [logprob_rows][n_vocab] log-softmax, oldest position first
};

class Engine {
public:
    /// Loads a GGUF. On failure returns null and fills `error`.
    static std::unique_ptr<Engine> load(const std::string & path, int n_threads, int n_ctx, std::string & error);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    const Info & info() const { return info_; }

    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const;
    std::string piece(int32_t token) const;
    /// The model's own chat template (or a plain "User:/Assistant:" fallback) with the assistant turn opened.
    std::string chat_prompt(const std::vector<ChatMessage> & messages) const;

    /// head_keep: empty (no cut) or n_layer*n_head values, 1 keeps a head, 0 removes it.
    /// on_text receives each printable piece; returning false stops the generation.
    Generation generate(const std::vector<int32_t> & prompt, int max_new, const std::vector<float> & head_keep,
                        float temperature, uint32_t seed,
                        const std::function<bool(const std::string &)> & on_text,
                        const std::atomic<bool> * cancel);

    /// One pass over `tokens` (from an empty cache, at most one batch of tokens). logprob_rows: how many of the last
    /// positions get their next-token log-probabilities (0: none).
    Analysis analyze(const std::vector<int32_t> & tokens, const std::vector<float> & head_keep,
                     bool want_power, bool want_attn, int logprob_rows);
    int max_pass_tokens() const { return n_batch_; }

    /// Called by llama.cpp's scheduler for every graph node (see the .cpp).
    bool on_tensor(ggml_tensor * t, bool ask);

private:
    Engine() = default;
    bool decode(const std::vector<int32_t> & tokens, int pos0, int logit_rows);
    int32_t pick(float temperature, uint32_t & rng_state) const;

    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    Info info_;
    int n_ctx_ = 0;
    int n_batch_ = 512;
    std::string template_;

    // What the eval callback does during the current pass.
    const float * keep_ = nullptr;   // n_layer*n_head or null
    bool cut_any_ = false;
    bool want_power_ = false;
    bool want_attn_ = false;
    int pass_tokens_ = 0;            // tokens in this pass (power/attn capture)
    std::vector<float> * power_ = nullptr;
    std::vector<float> * attn_ = nullptr;
    int seen_kqv_ = 0;               // kqv_out tensors met in this pass (support check)
};

/// Short system description for logs: SIMD features the CPU backend found.
std::string system_info();

}  // namespace nl
