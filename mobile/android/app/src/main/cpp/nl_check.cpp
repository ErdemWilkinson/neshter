// Desktop check for the engine: nl_check <model.gguf>. Prints facts and exits non-zero when one fails.
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "neshter_llama.h"

static int failures = 0;

static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) ++failures;
}

static double kl(const std::vector<float> & p_log, const std::vector<float> & q_log) {
    double s = 0;
    for (size_t i = 0; i < p_log.size(); ++i) s += std::exp(p_log[i]) * (p_log[i] - q_log[i]);
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: nl_check <model.gguf>\n");
        return 2;
    }
    const int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::string error;
    auto e = nl::Engine::load(argv[1], threads, 2048, error);
    if (!e) {
        std::fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }
    const nl::Info & in = e->info();
    std::printf("%s | %s | arch=%s layers=%d heads=%d kv=%d embd=%d head_dim=%d vocab=%d template=%d heads_ok=%d\n",
                in.name.c_str(), in.desc.c_str(), in.arch.c_str(), in.n_layer, in.n_head, in.n_head_kv, in.n_embd,
                in.head_dim, in.n_vocab, in.has_template, in.heads_supported);
    std::printf("%s\n", nl::system_info().c_str());
    check(in.heads_supported, "every layer exposes per-head attention output");

    const std::string prompt = e->chat_prompt({{"user", "What is the capital of France? Answer in one sentence."}});
    const std::vector<int32_t> toks = e->tokenize(prompt, true, true);
    std::printf("prompt tokens: %zu\n", toks.size());

    const size_t LH = static_cast<size_t>(in.n_layer) * in.n_head;
    const std::vector<float> ones(LH, 1.0f);

    nl::Analysis base = e->analyze(toks, {}, true, true, 1);
    nl::Analysis same = e->analyze(toks, ones, false, false, 1);
    check(base.n_tokens == static_cast<int>(toks.size()), "analysis covers every token");
    check(kl(base.logprobs, same.logprobs) < 1e-9, "keeping every head changes nothing");

    nl::Analysis rows4 = e->analyze(toks, {}, false, false, 4);
    const size_t V = static_cast<size_t>(in.n_vocab);
    double row_gap = 0;
    for (size_t i = 0; i < V; ++i) row_gap = std::max(row_gap, std::fabs(double(rows4.logprobs[3 * V + i]) - base.logprobs[i]));
    check(rows4.logprob_rows == 4 && row_gap < 1e-4, "the last of several log-prob rows is the final position");

    // Attention rows: sum to 1 and see no future token.
    const size_t T = static_cast<size_t>(base.n_tokens);
    double worst_sum = 0, worst_future = 0;
    for (size_t lh = 0; lh < LH; ++lh) {
        for (size_t q = 0; q < T; ++q) {
            const float * row = base.attn.data() + (lh * T + q) * T;
            double s = 0;
            for (size_t k = 0; k < T; ++k) {
                s += row[k];
                if (k > q) worst_future = std::max(worst_future, static_cast<double>(row[k]));
            }
            worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
        }
    }
    std::printf("attention: worst |row sum - 1| = %.2e, worst future weight = %.2e\n", worst_sum, worst_future);
    check(worst_sum < 1e-3 && worst_future < 1e-6, "attention rows are causal distributions");

    double pmax = 0, pmin = 1e30;
    for (float v : base.power) {
        pmax = std::max(pmax, static_cast<double>(v));
        pmin = std::min(pmin, static_cast<double>(v));
    }
    std::printf("head power range: %.4f .. %.4f\n", pmin, pmax);
    check(pmax > 0, "head power is measured");

    // One head of the middle layer, then a whole layer, then everything.
    std::vector<float> one_head = ones;
    one_head[static_cast<size_t>(in.n_layer / 2) * in.n_head] = 0.0f;
    std::vector<float> one_layer = ones;
    for (int h = 0; h < in.n_head; ++h) one_layer[static_cast<size_t>(in.n_layer / 2) * in.n_head + h] = 0.0f;
    const std::vector<float> none(LH, 0.0f);
    const double kl_head = kl(base.logprobs, e->analyze(toks, one_head, false, false, 1).logprobs);
    const double kl_layer = kl(base.logprobs, e->analyze(toks, one_layer, false, false, 1).logprobs);
    const double kl_all = kl(base.logprobs, e->analyze(toks, none, false, false, 1).logprobs);
    std::printf("KL of next token: one head %.5f, one layer %.5f, all heads %.5f\n", kl_head, kl_layer, kl_all);
    check(kl_head > 0 && kl_layer > kl_head && kl_all > kl_layer, "cuts change the output, bigger cuts more");

    nl::Analysis cut_power = e->analyze(toks, one_layer, true, false, 0);
    double cut_sum = 0;
    for (int h = 0; h < in.n_head; ++h)
        for (size_t j = 0; j < T; ++j)
            cut_sum += cut_power.power[(static_cast<size_t>(in.n_layer / 2) * in.n_head + h) * T + j];
    check(cut_sum == 0.0, "a cut head shows zero power");

    // Lens: the last row is the model's own output, so it must agree with the real prediction.
    for (const int at : {-1, static_cast<int>(toks.size()) / 2}) {
        const nl::Lens ln = e->lens(toks, at, 3);
        const int R = in.n_layer + 1;
        const bool shaped = ln.n_tokens == static_cast<int>(toks.size()) && static_cast<int>(ln.p_final.size()) == R;
        check(shaped, "lens has one row per layer plus the embedding");
        if (!shaped) continue;
        const nl::Analysis real = e->analyze(std::vector<int32_t>(toks.begin(), toks.begin() + ln.position + 1), {}, false, false, 1);
        const double p_real = std::exp(static_cast<double>(real.logprobs[static_cast<size_t>(ln.final_token)]));
        const int last = R - 1;
        std::printf("lens at %d: final '%s' p=%.4f (model %.4f); rank by row:", ln.position, e->piece(ln.final_token).c_str(),
                    ln.p_final[last], p_real);
        for (int r = 0; r < R; r += std::max(1, R / 8)) std::printf(" %d", ln.rank_final[r]);
        std::printf("\n");
        check(ln.top_ids[static_cast<size_t>(last) * ln.top_k] == ln.final_token && ln.rank_final[last] == 0,
              "the lens' last row predicts what the model predicts");
        check(std::fabs(ln.p_final[last] - p_real) < 2e-3, "the lens' last row has the model's probability");
    }

    std::string streamed;
    nl::Generation g = e->generate(toks, 48, {}, 0.0f, 0, [&](const std::string & s) {
        streamed += s;
        return true;
    }, nullptr);
    std::printf("answer (%zu tokens, prompt %.0f ms, %.1f tok/s): %s\n", g.tokens.size(), g.prompt_ms,
                g.tokens.empty() ? 0.0 : g.tokens.size() * 1000.0 / g.gen_ms, g.text.c_str());
    check(!g.text.empty() && streamed == g.text, "generation streams the whole answer");

    nl::Generation g2 = e->generate(toks, 48, {}, 0.0f, 0, nullptr, nullptr);
    check(g2.text == g.text, "greedy generation repeats exactly");
    nl::Generation g3 = e->generate(toks, 48, ones, 0.0f, 0, nullptr, nullptr);
    check(g3.text == g.text, "an all-keep mask leaves generation unchanged");
    nl::Generation g4 = e->generate(toks, 48, one_layer, 0.0f, 0, nullptr, nullptr);
    std::printf("answer with layer %d cut: %s\n", in.n_layer / 2, g4.text.c_str());

    std::printf(failures == 0 ? "RESULT: passed\n" : "RESULT: %d failed\n", failures);
    return failures == 0 ? 0 : 1;
}
