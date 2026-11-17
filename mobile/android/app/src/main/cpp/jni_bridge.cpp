// JNI side of dev.neshter.mobile.LlamaNative. Text crosses as UTF-8 byte arrays both ways: JNI's own string calls use
// "modified UTF-8", which mangles emoji and other 4-byte characters.
#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "neshter_llama.h"

namespace {

constexpr const char * TAG = "NeshterLlama";

struct Handle {
    std::unique_ptr<nl::Engine> engine;
    std::atomic<bool> cancel{false};
};

std::string g_last_error;

void log_to_android(ggml_log_level level, const char * text, void *) {
    const int prio = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
                     : level == GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN
                                                    : ANDROID_LOG_DEBUG;
    __android_log_write(prio, TAG, text);
}

std::string to_string(JNIEnv * env, jbyteArray bytes) {
    if (bytes == nullptr) return {};
    const jsize n = env->GetArrayLength(bytes);
    std::string s(static_cast<size_t>(n), '\0');
    env->GetByteArrayRegion(bytes, 0, n, reinterpret_cast<jbyte *>(s.data()));
    return s;
}

jbyteArray to_bytes(JNIEnv * env, const std::string & s) {
    jbyteArray out = env->NewByteArray(static_cast<jsize>(s.size()));
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte *>(s.data()));
    return out;
}

std::vector<int32_t> to_tokens(JNIEnv * env, jintArray arr) {
    std::vector<int32_t> out;
    if (arr == nullptr) return out;
    out.resize(static_cast<size_t>(env->GetArrayLength(arr)));
    env->GetIntArrayRegion(arr, 0, static_cast<jsize>(out.size()), reinterpret_cast<jint *>(out.data()));
    return out;
}

std::vector<float> to_floats(JNIEnv * env, jfloatArray arr) {
    std::vector<float> out;
    if (arr == nullptr) return out;
    out.resize(static_cast<size_t>(env->GetArrayLength(arr)));
    env->GetFloatArrayRegion(arr, 0, static_cast<jsize>(out.size()), out.data());
    return out;
}

jfloatArray from_floats(JNIEnv * env, const std::vector<float> & v) {
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(v.size()));
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(v.size()), v.data());
    return out;
}

std::string json_escape(const std::string & s) {
    std::string out;
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

Handle * handle(jlong h) { return reinterpret_cast<Handle *>(h); }

}  // namespace

extern "C" {

JNIEXPORT jbyteArray JNICALL Java_dev_neshter_mobile_LlamaNative_init(JNIEnv * env, jobject, jbyteArray lib_dir) {
    static std::once_flag once;
    const std::string dir = to_string(env, lib_dir);
    std::call_once(once, [&] {
        llama_log_set(log_to_android, nullptr);
        // libggml-cpu-<variant>.so files sit next to this library; the best one for this CPU is chosen here.
        ggml_backend_load_all_from_path(dir.c_str());
    });
    return to_bytes(env, nl::system_info());
}

JNIEXPORT jlong JNICALL Java_dev_neshter_mobile_LlamaNative_load(JNIEnv * env, jobject, jbyteArray path, jint threads,
                                                                  jint n_ctx) {
    std::string error;
    auto engine = nl::Engine::load(to_string(env, path), threads, n_ctx, error);
    if (!engine) {
        g_last_error = error;
        return 0;
    }
    auto * h = new Handle();
    h->engine = std::move(engine);
    return reinterpret_cast<jlong>(h);
}

JNIEXPORT jbyteArray JNICALL Java_dev_neshter_mobile_LlamaNative_lastError(JNIEnv * env, jobject) {
    return to_bytes(env, g_last_error);
}

JNIEXPORT void JNICALL Java_dev_neshter_mobile_LlamaNative_free(JNIEnv *, jobject, jlong h) {
    delete handle(h);
}

JNIEXPORT jbyteArray JNICALL Java_dev_neshter_mobile_LlamaNative_info(JNIEnv * env, jobject, jlong h) {
    const nl::Info & in = handle(h)->engine->info();
    std::string j = "{";
    j += "\"name\":\"" + json_escape(in.name) + "\"";
    j += ",\"arch\":\"" + json_escape(in.arch) + "\"";
    j += ",\"desc\":\"" + json_escape(in.desc) + "\"";
    j += ",\"n_layers\":" + std::to_string(in.n_layer);
    j += ",\"n_heads\":" + std::to_string(in.n_head);
    j += ",\"n_kv_heads\":" + std::to_string(in.n_head_kv);
    j += ",\"d_model\":" + std::to_string(in.n_embd);
    j += ",\"d_mlp\":" + std::to_string(in.n_ff);
    j += ",\"head_dim\":" + std::to_string(in.head_dim);
    j += ",\"vocab\":" + std::to_string(in.n_vocab);
    j += ",\"n_ctx_train\":" + std::to_string(in.n_ctx_train);
    j += ",\"size_bytes\":" + std::to_string(in.size_bytes);
    j += ",\"n_params\":" + std::to_string(in.n_params);
    j += std::string(",\"has_template\":") + (in.has_template ? "true" : "false");
    j += std::string(",\"heads_supported\":") + (in.heads_supported ? "true" : "false");
    j += "}";
    return to_bytes(env, j);
}

JNIEXPORT jintArray JNICALL Java_dev_neshter_mobile_LlamaNative_tokenize(JNIEnv * env, jobject, jlong h,
                                                                          jbyteArray text, jboolean add_special,
                                                                          jboolean parse_special) {
    const std::vector<int32_t> t = handle(h)->engine->tokenize(to_string(env, text), add_special, parse_special);
    jintArray out = env->NewIntArray(static_cast<jsize>(t.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(t.size()), reinterpret_cast<const jint *>(t.data()));
    return out;
}

JNIEXPORT jobjectArray JNICALL Java_dev_neshter_mobile_LlamaNative_pieces(JNIEnv * env, jobject, jlong h,
                                                                           jintArray tokens) {
    const std::vector<int32_t> t = to_tokens(env, tokens);
    jclass bytes_class = env->FindClass("[B");
    jobjectArray out = env->NewObjectArray(static_cast<jsize>(t.size()), bytes_class, nullptr);
    for (size_t i = 0; i < t.size(); ++i) {
        jbyteArray b = to_bytes(env, handle(h)->engine->piece(t[i]));
        env->SetObjectArrayElement(out, static_cast<jsize>(i), b);
        env->DeleteLocalRef(b);
    }
    return out;
}

/// roles and contents: parallel arrays of UTF-8 strings.
JNIEXPORT jbyteArray JNICALL Java_dev_neshter_mobile_LlamaNative_chatPrompt(JNIEnv * env, jobject, jlong h,
                                                                             jobjectArray roles,
                                                                             jobjectArray contents) {
    std::vector<nl::ChatMessage> msgs;
    const jsize n = env->GetArrayLength(roles);
    for (jsize i = 0; i < n; ++i) {
        auto r = static_cast<jbyteArray>(env->GetObjectArrayElement(roles, i));
        auto c = static_cast<jbyteArray>(env->GetObjectArrayElement(contents, i));
        msgs.push_back({to_string(env, r), to_string(env, c)});
        env->DeleteLocalRef(r);
        env->DeleteLocalRef(c);
    }
    return to_bytes(env, handle(h)->engine->chat_prompt(msgs));
}

JNIEXPORT void JNICALL Java_dev_neshter_mobile_LlamaNative_cancel(JNIEnv *, jobject, jlong h) {
    handle(h)->cancel.store(true);
}

/// sink: an object with `boolean onText(byte[])`. Returns the whole answer as UTF-8.
JNIEXPORT jbyteArray JNICALL Java_dev_neshter_mobile_LlamaNative_generate(JNIEnv * env, jobject, jlong h,
                                                                           jintArray prompt, jint max_new,
                                                                           jfloatArray keep, jfloat temperature,
                                                                           jint seed, jobject sink) {
    Handle * hd = handle(h);
    hd->cancel.store(false);
    jmethodID on_text = nullptr;
    if (sink != nullptr) {
        jclass cls = env->GetObjectClass(sink);
        on_text = env->GetMethodID(cls, "onText", "([B)Z");
    }
    const nl::Generation g = hd->engine->generate(
        to_tokens(env, prompt), max_new, to_floats(env, keep), temperature, static_cast<uint32_t>(seed),
        [&](const std::string & s) {
            if (on_text == nullptr) return true;
            jbyteArray b = to_bytes(env, s);
            const jboolean go = env->CallBooleanMethod(sink, on_text, b);
            env->DeleteLocalRef(b);
            return go == JNI_TRUE && !env->ExceptionCheck();
        },
        &hd->cancel);
    return to_bytes(env, g.text);
}

/// Returns [power, attn, logprobs ([rows][vocab])]; an entry is null when it was not asked for.
JNIEXPORT jobjectArray JNICALL Java_dev_neshter_mobile_LlamaNative_analyze(JNIEnv * env, jobject, jlong h,
                                                                            jintArray tokens, jfloatArray keep,
                                                                            jboolean power, jboolean attn,
                                                                            jint logprob_rows) {
    const nl::Analysis a =
        handle(h)->engine->analyze(to_tokens(env, tokens), to_floats(env, keep), power, attn, logprob_rows);
    jclass floats_class = env->FindClass("[F");
    jobjectArray out = env->NewObjectArray(3, floats_class, nullptr);
    if (power) env->SetObjectArrayElement(out, 0, from_floats(env, a.power));
    if (attn) env->SetObjectArrayElement(out, 1, from_floats(env, a.attn));
    if (logprob_rows > 0) env->SetObjectArrayElement(out, 2, from_floats(env, a.logprobs));
    return out;
}

}  // extern "C"
