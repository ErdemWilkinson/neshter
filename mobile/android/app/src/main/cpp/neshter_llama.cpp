#include "neshter_llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

namespace nl {
namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

bool eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    return static_cast<Engine *>(user_data)->on_tensor(t, ask);
}

std::string meta(const llama_model * model, const char * key) {
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf, std::min<size_t>(n, sizeof(buf) - 1));
}

int layer_of(const char * name, size_t prefix) {
    return std::atoi(name + prefix);
}

/// Length of the longest prefix of `s` that ends on a whole UTF-8 character.
size_t utf8_complete(const std::string & s) {
    size_t i = s.size();
    // Look back at most 3 bytes for an unfinished lead byte.
    for (size_t back = 1; back <= 4 && back <= s.size(); ++back) {
        const unsigned char c = static_cast<unsigned char>(s[s.size() - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        if (back < need) i = s.size() - back;  // the last character is not finished yet
        break;
    }
    return i;
}

std::once_flag g_backend_once;

}  // namespace

std::string system_info() {
    return llama_print_system_info();
}

std::unique_ptr<Engine> Engine::load(const std::string & path, int n_threads, int n_ctx, std::string & error) {
    std::call_once(g_backend_once, [] { llama_backend_init(); });

    llama_model_params mp = llama_model_default_params();
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;  // the file stays on storage; the phone pages in what it needs
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (model == nullptr) {
        error = "Bu dosya bir GGUF modeli olarak açılamadı (bozuk, eksik ya da desteklenmeyen bir mimari).";
        return nullptr;
    }

    std::unique_ptr<Engine> e(new Engine());
    e->model_ = model;
    e->n_ctx_ = std::max(256, std::min(n_ctx, std::max(256, llama_model_n_ctx_train(model))));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(e->n_ctx_);
    cp.n_batch = static_cast<uint32_t>(e->n_batch_);
    cp.n_ubatch = static_cast<uint32_t>(e->n_batch_);
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    // Without flash attention the graph keeps "kq_soft_max-N" (the attention pattern) as its own tensor.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = eval_callback;
    cp.cb_eval_user_data = e.get();
    cp.no_perf = true;
    e->ctx_ = llama_init_from_model(model, cp);
    if (e->ctx_ == nullptr) {
        error = "Model açıldı ama çalışma alanı kurulamadı (bellek yetmemiş olabilir).";
        return nullptr;
    }

    Info & in = e->info_;
    char desc[256] = {0};
    llama_model_desc(model, desc, sizeof(desc));
    in.desc = desc;
    in.arch = meta(model, "general.architecture");
    in.name = meta(model, "general.name");
    if (in.name.empty()) {
        const size_t slash = path.find_last_of("/\\");
        in.name = slash == std::string::npos ? path : path.substr(slash + 1);
    }
    in.n_layer = llama_model_n_layer(model);
    in.n_head = llama_model_n_head(model);
    in.n_head_kv = llama_model_n_head_kv(model);
    in.n_embd = llama_model_n_embd(model);
    in.n_ff = std::atoi(meta(model, (in.arch + ".feed_forward_length").c_str()).c_str());
    in.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    in.n_ctx_train = llama_model_n_ctx_train(model);
    in.size_bytes = llama_model_size(model);
    in.n_params = llama_model_n_params(model);
    const char * tmpl = llama_model_chat_template(model, nullptr);
    e->template_ = tmpl == nullptr ? std::string() : std::string(tmpl);
    in.has_template = !e->template_.empty();
    in.head_dim = in.n_head > 0 ? in.n_embd / in.n_head : 0;

    // Probe: one token through the graph tells whether every layer has a per-head attention output we can read.
    if (in.n_head > 0) {
        const std::vector<int32_t> one = e->tokenize("a", true, false);
        if (!one.empty()) {
            e->analyze(one, {}, true, false, 0);
            in.heads_supported = e->seen_kqv_ == in.n_layer && in.head_dim > 0;
        }
    }
    return e;
}

Engine::~Engine() {
    if (ctx_ != nullptr) llama_free(ctx_);
    if (model_ != nullptr) llama_model_free(model_);
}

std::vector<int32_t> Engine::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    const llama_vocab * vocab = llama_model_get_vocab(model_);
    std::vector<int32_t> out(text.size() + 16);
    int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), out.data(),
                               static_cast<int32_t>(out.size()), add_special, parse_special);
    if (n < 0) {
        out.resize(static_cast<size_t>(-n));
        n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), out.data(),
                           static_cast<int32_t>(out.size()), add_special, parse_special);
    }
    out.resize(static_cast<size_t>(std::max(0, n)));
    return out;
}

std::string Engine::piece(int32_t token) const {
    const llama_vocab * vocab = llama_model_get_vocab(model_);
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::string big(static_cast<size_t>(-n), '\0');
        n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, true);
        big.resize(static_cast<size_t>(std::max(0, n)));
        return big;
    }
    return std::string(buf, static_cast<size_t>(n));
}

std::string Engine::chat_prompt(const std::vector<ChatMessage> & messages) const {
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    for (const ChatMessage & m : messages) chat.push_back({m.role.c_str(), m.content.c_str()});
    if (!template_.empty()) {
        std::string buf(4096, '\0');
        int32_t n = llama_chat_apply_template(template_.c_str(), chat.data(), chat.size(), true, buf.data(),
                                              static_cast<int32_t>(buf.size()));
        if (n > static_cast<int32_t>(buf.size())) {
            buf.resize(static_cast<size_t>(n));
            n = llama_chat_apply_template(template_.c_str(), chat.data(), chat.size(), true, buf.data(),
                                          static_cast<int32_t>(buf.size()));
        }
        if (n >= 0) {
            buf.resize(static_cast<size_t>(n));
            return buf;
        }
        // A template llama.cpp does not know: fall through to the plain form.
    }
    std::string out;
    for (const ChatMessage & m : messages) {
        const std::string who = m.role == "system" ? "System" : m.role == "assistant" ? "Assistant" : "User";
        out += who + ": " + m.content + "\n";
    }
    out += "Assistant:";
    return out;
}

bool Engine::decode(const std::vector<int32_t> & tokens, int pos0, int logit_rows) {
    llama_batch batch = llama_batch_init(n_batch_, 0, 1);
    bool ok = true;
    for (size_t start = 0; start < tokens.size() && ok; start += static_cast<size_t>(n_batch_)) {
        const size_t count = std::min(tokens.size() - start, static_cast<size_t>(n_batch_));
        batch.n_tokens = static_cast<int32_t>(count);
        for (size_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = static_cast<llama_pos>(pos0 + start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = start + i + static_cast<size_t>(logit_rows) >= tokens.size() ? 1 : 0;
        }
        ok = llama_decode(ctx_, batch) == 0;
    }
    llama_batch_free(batch);
    return ok;
}

int32_t Engine::pick(float temperature, uint32_t & rng) const {
    const float * logits = llama_get_logits_ith(ctx_, -1);
    const int n = info_.n_vocab;
    if (temperature <= 0.0f) {
        return static_cast<int32_t>(std::max_element(logits, logits + n) - logits);
    }
    const float mx = *std::max_element(logits, logits + n);
    std::vector<double> p(static_cast<size_t>(n));
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += (p[i] = std::exp((logits[i] - mx) / temperature));
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    double r = (rng / 4294967296.0) * sum;
    for (int i = 0; i < n; ++i) {
        r -= p[i];
        if (r <= 0) return i;
    }
    return n - 1;
}

Generation Engine::generate(const std::vector<int32_t> & prompt, int max_new, const std::vector<float> & head_keep,
                            float temperature, uint32_t seed,
                            const std::function<bool(const std::string &)> & on_text,
                            const std::atomic<bool> * cancel) {
    Generation g;
    g.n_prompt = static_cast<int>(prompt.size());
    if (prompt.empty()) return g;
    std::vector<int32_t> p = prompt;
    const int room = n_ctx_ - 1;
    if (static_cast<int>(p.size()) >= room) p.erase(p.begin(), p.end() - (room / 2));  // keep the end of a long chat
    max_new = std::max(0, std::min(max_new, room - static_cast<int>(p.size())));

    llama_memory_clear(llama_get_memory(ctx_), true);
    keep_ = head_keep.size() == static_cast<size_t>(info_.n_layer * info_.n_head) ? head_keep.data() : nullptr;
    cut_any_ = keep_ != nullptr && std::any_of(head_keep.begin(), head_keep.end(), [](float k) { return k < 1.0f; });
    want_power_ = want_attn_ = false;

    const auto t0 = clock_type::now();
    if (!decode(p, 0, 1)) {
        keep_ = nullptr;
        cut_any_ = false;
        return g;
    }
    g.prompt_ms = ms_since(t0);

    const llama_vocab * vocab = llama_model_get_vocab(model_);
    uint32_t rng = seed == 0 ? 0x9E3779B9u : seed;
    std::string pending;
    const auto t1 = clock_type::now();
    int pos = static_cast<int>(p.size());
    for (int i = 0; i < max_new; ++i) {
        if (cancel != nullptr && cancel->load()) break;
        const int32_t tok = pick(temperature, rng);
        if (llama_vocab_is_eog(vocab, tok)) {
            g.stopped_by_eog = true;
            break;
        }
        g.tokens.push_back(tok);
        char buf[256];
        const int32_t n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n > 0) pending.append(buf, static_cast<size_t>(n));
        const size_t ready = utf8_complete(pending);
        if (ready > 0) {
            const std::string out = pending.substr(0, ready);
            pending.erase(0, ready);
            g.text += out;
            if (on_text && !on_text(out)) break;
        }
        if (!decode({tok}, pos, 1)) break;
        ++pos;
    }
    if (!pending.empty()) g.text += pending;
    g.gen_ms = ms_since(t1);
    keep_ = nullptr;
    cut_any_ = false;
    return g;
}

Analysis Engine::analyze(const std::vector<int32_t> & tokens, const std::vector<float> & head_keep, bool want_power,
                         bool want_attn, int logprob_rows) {
    Analysis a;
    std::vector<int32_t> t = tokens;
    if (static_cast<int>(t.size()) > n_batch_) t.resize(static_cast<size_t>(n_batch_));  // one ubatch: positions line up
    a.n_tokens = static_cast<int>(t.size());
    if (t.empty()) return a;
    const size_t L = static_cast<size_t>(info_.n_layer), H = static_cast<size_t>(info_.n_head),
                 T = static_cast<size_t>(a.n_tokens);
    if (want_power) a.power.assign(L * H * T, 0.0f);
    if (want_attn) a.attn.assign(L * H * T * T, 0.0f);

    llama_memory_clear(llama_get_memory(ctx_), true);
    keep_ = head_keep.size() == L * H ? head_keep.data() : nullptr;
    cut_any_ = keep_ != nullptr && std::any_of(head_keep.begin(), head_keep.end(), [](float k) { return k < 1.0f; });
    want_power_ = want_power;
    want_attn_ = want_attn;
    pass_tokens_ = a.n_tokens;
    power_ = want_power ? &a.power : nullptr;
    attn_ = want_attn ? &a.attn : nullptr;
    seen_kqv_ = 0;

    const int want_rows = std::min(std::max(0, logprob_rows), a.n_tokens);
    logprob_rows = std::max(1, want_rows);  // llama.cpp wants at least one output
    const bool ok = decode(t, 0, logprob_rows);

    keep_ = nullptr;
    cut_any_ = want_power_ = want_attn_ = false;
    power_ = attn_ = nullptr;
    if (!ok) {
        a.n_tokens = 0;
        return a;
    }
    if (want_rows > 0) {
        const int n = info_.n_vocab;
        a.logprob_rows = want_rows;
        a.logprobs.resize(static_cast<size_t>(want_rows) * n);
        for (int r = 0; r < want_rows; ++r) {
            // outputs are numbered in batch order; the last `logprob_rows` tokens asked for one
            const float * logits = llama_get_logits_ith(ctx_, a.n_tokens - want_rows + r);
            float * dst = a.logprobs.data() + static_cast<size_t>(r) * n;
            const float mx = *std::max_element(logits, logits + n);
            double sum = 0;
            for (int i = 0; i < n; ++i) sum += std::exp(static_cast<double>(logits[i] - mx));
            const float lse = mx + static_cast<float>(std::log(sum));
            for (int i = 0; i < n; ++i) dst[i] = logits[i] - lse;
        }
    }
    return a;
}

bool Engine::on_tensor(ggml_tensor * t, bool ask) {
    const char * name = t->name;
    const bool is_kqv = std::strncmp(name, "kqv_out-", 8) == 0;
    const bool is_sm = std::strncmp(name, "kq_soft_max-", 12) == 0;
    if (ask) return (is_kqv && (cut_any_ || want_power_)) || (is_sm && want_attn_);
    if (!is_kqv && !is_sm) return true;
    if (t->type != GGML_TYPE_F32 || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) return true;

    const int H = info_.n_head;
    if (is_kqv) {
        const int il = layer_of(name, 8);
        ++seen_kqv_;
        if (il < 0 || il >= info_.n_layer || H <= 0 || t->ne[0] % H != 0) return true;
        const int64_t hd = t->ne[0] / H;
        const int64_t T = t->ne[1];
        for (int64_t j = 0; j < T; ++j) {
            float * row = reinterpret_cast<float *>(static_cast<char *>(t->data) + j * t->nb[1]);
            for (int h = 0; h < H; ++h) {
                float * x = row + h * hd;
                const float keep = keep_ != nullptr ? keep_[il * H + h] : 1.0f;
                if (keep != 1.0f) {
                    for (int64_t d = 0; d < hd; ++d) x[d] *= keep;
                }
                if (power_ != nullptr && j < pass_tokens_) {
                    double s = 0;
                    for (int64_t d = 0; d < hd; ++d) s += static_cast<double>(x[d]) * x[d];
                    (*power_)[(static_cast<size_t>(il) * H + h) * pass_tokens_ + j] = static_cast<float>(std::sqrt(s));
                }
            }
        }
    } else if (attn_ != nullptr) {
        const int il = layer_of(name, 12);
        if (il < 0 || il >= info_.n_layer) return true;
        // [n_kv, n_tokens, n_head]: one row of key weights per query and head; the cache was empty, so keys 0..T-1.
        const int64_t T = std::min<int64_t>(t->ne[1], pass_tokens_);
        const int64_t heads = std::min<int64_t>(t->ne[2], H);
        const int64_t K = std::min<int64_t>(t->ne[0], pass_tokens_);
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t q = 0; q < T; ++q) {
                const float * row =
                    reinterpret_cast<const float *>(static_cast<const char *>(t->data) + q * t->nb[1] + h * t->nb[2]);
                float * dst = attn_->data() + ((static_cast<size_t>(il) * H + h) * pass_tokens_ + q) * pass_tokens_;
                std::copy(row, row + K, dst);
            }
        }
    }
    return true;
}

}  // namespace nl
