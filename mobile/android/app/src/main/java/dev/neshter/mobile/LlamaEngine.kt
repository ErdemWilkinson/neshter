package dev.neshter.mobile

import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp

/**
 * A GGUF model through llama.cpp (src/main/cpp). Same answers as [Engine], measured on the model's own graph:
 * head power and cuts on each head's attention output before the output projection, attention from the softmax.
 */
class LlamaEngine private constructor(
    private val handle: Long,
    override val info: JSONObject,
    private val cancel: AtomicBoolean,
) : ModelEngine {
    private val L = info.getInt("n_layers")
    private val H = info.getInt("n_heads")
    private val V = info.getInt("vocab")
    private val headsOk = info.getBoolean("heads_supported")
    private val ones = FloatArray(L * H) { 1f }

    private class Base(val ids: IntArray, val keep: Int, val p: FloatArray, val plogp: Double)
    private var bases: List<Base> = emptyList()

    override fun close() = LlamaNative.free(handle)

    // --- helpers -------------------------------------------------------------------------------------------

    private fun encode(text: String, max: Int, special: Boolean = false): IntArray {
        val ids = LlamaNative.tokenize(handle, text.toByteArray(), true, special)
        if (ids.isEmpty()) throw BridgeError("invalid_input", "Boş istem: okunacak token yok.")
        if (ids.size > max) throw BridgeError("invalid_input", "Telefonda istem en fazla $max token olabilir (bu: ${ids.size}).")
        return ids
    }

    private fun pieces(ids: IntArray): List<String> = LlamaNative.pieces(handle, ids).map { String(it, Charsets.UTF_8) }
    private fun piece(id: Int): String = pieces(intArrayOf(id))[0]

    private fun needHeads() {
        if (!headsOk) throw BridgeError("unsupported", "Bu modelin mimarisinde head'ler ayrı okunamıyor; yalnız sohbet edilebilir.")
    }

    private fun checkCancel() {
        if (cancel.get()) throw BridgeError("cancelled", "İptal edildi.")
    }

    override fun mask(rows: List<List<Number>>?): FloatArray {
        if (rows == null) return ones
        if (rows.size != L || rows.any { it.size != H }) throw BridgeError("invalid_input", "Maske $L×$H olmalı.")
        return FloatArray(L * H) { rows[it / H][it % H].toFloat() }
    }

    private fun logprobs(ids: IntArray, mask: FloatArray, rows: Int): FloatArray =
        LlamaNative.analyze(handle, ids, mask, false, false, rows)[2] ?: throw BridgeError("internal", "Model çalıştırılamadı.")

    private fun argmax(x: FloatArray): Int {
        var best = 0
        for (i in x.indices) if (x[i] > x[best]) best = i
        return best
    }

    private fun topK(logp: FloatArray, k: Int): List<Pair<Int, Float>> =
        logp.indices.sortedByDescending { logp[it] }.take(k).map { it to exp(logp[it]) }

    private fun Json.tokenRow(id: Int, p: Float) = obj {
        key("token_id").num(id.toLong())
        key("text").str(piece(id))
        key("p").num(p)
    }

    // --- measurements --------------------------------------------------------------------------------------

    override fun activity(text: String): String {
        needHeads()
        val ids = encode(text, MAX_VIEW_TOKENS)
        val T = ids.size
        val power = LlamaNative.analyze(handle, ids, null, true, false, 0)[0]!!
        return Json.build {
            obj {
                key("tokens").strings(pieces(ids))
                key("power_raw").arr {
                    for (l in 0 until L) arr { for (h in 0 until H) floats(power, (l * H + h) * T, T) }
                }
            }
        }
    }

    override fun attention(text: String, layer: Int): String {
        needHeads()
        if (layer !in 0 until L) throw BridgeError("invalid_input", "Katman $layer aralık dışı (0..${L - 1}).")
        val ids = encode(text, MAX_VIEW_TOKENS)
        val T = ids.size
        val attn = LlamaNative.analyze(handle, ids, null, false, true, 0)[1]!!
        return Json.build {
            obj {
                key("tokens").strings(pieces(ids))
                key("layer").num(layer.toLong())
                key("patterns").arr {
                    for (h in 0 until H) arr { for (q in 0 until T) floats(attn, ((layer * H + h) * T + q) * T, T) }
                }
            }
        }
    }

    override fun lens(text: String, position: Int): String {
        if (!info.optBoolean("lens_supported")) {
            throw BridgeError("unsupported", "Bu modelin mimarisinde katman çıktıları okunamıyor; katman katman tahmin yok.")
        }
        val ids = encode(text, MAX_VIEW_TOKENS)
        val T = ids.size
        if (position !in -T until T) throw BridgeError("invalid_input", "position $position aralık dışı (istem $T token).")
        val f = LlamaNative.lens(handle, ids, position, 3)
        if (f.isEmpty()) throw BridgeError("internal", "Katman katman tahmin hesaplanamadı.")
        val pos = f[1].toInt()
        val y = f[2].toInt()
        val k = f[3].toInt()
        val rows = L + 1
        val stride = 2 * k + 2
        return Json.build {
            obj {
                key("tokens").strings(pieces(ids))
                key("position").num(pos.toLong())
                key("final").obj { key("token_id").num(y.toLong()); key("text").str(piece(y)) }
                key("layers").arr {
                    for (r in 0 until rows) obj {
                        val at = 4 + r * stride
                        key("layer").num((r - 1).toLong())
                        key("top").arr { for (j in 0 until k) tokenRow(f[at + j].toInt(), f[at + k + j]) }
                        key("p_final").num(f[at + 2 * k])
                        key("rank_final").num(f[at + 2 * k + 1].toLong())
                    }
                }
            }
        }
    }

    override fun klBase(prompts: List<String>) {
        needHeads()
        cancel.set(false)
        bases = prompts.map { text ->
            checkCancel()
            val ids = encode(text, MAX_KL_TOKENS)
            val keep = minOf(KL_POSITIONS, ids.size)
            val p = logprobs(ids, ones, keep)
            var plogp = 0.0
            for (i in p.indices) {
                val lp = p[i].toDouble()
                val e = exp(lp)
                plogp += e * lp
                p[i] = e.toFloat()
            }
            Base(ids, keep, p, plogp)
        }
    }

    override fun kl(mask: FloatArray): List<Double> {
        if (bases.isEmpty()) throw BridgeError("internal", "Önce klBase çağrılmalı.")
        cancel.set(false)
        return bases.map { b ->
            checkCancel()
            val q = logprobs(b.ids, mask, b.keep)
            var plogq = 0.0
            for (i in q.indices) plogq += b.p[i].toDouble() * q[i]
            (b.plogp - plogq) / b.keep
        }
    }

    override fun probe(text: String, mask: FloatArray): String {
        needHeads()
        val ids = encode(text, MAX_VIEW_TOKENS)
        val before = logprobs(ids, ones, 1)
        val after = logprobs(ids, mask, 1)
        val y = argmax(before)
        return Json.build {
            obj {
                key("answer").obj { key("token_id").num(y.toLong()); key("text").str(piece(y)) }
                key("p_before").num(exp(before[y]))
                key("p_after").num(exp(after[y]))
                key("top_before").arr { topK(before, 3).forEach { (i, p) -> tokenRow(i, p) } }
                key("top_after").arr { topK(after, 3).forEach { (i, p) -> tokenRow(i, p) } }
            }
        }
    }

    override fun generate(prompt: String, mask: FloatArray, maxNew: Int, onText: ((String) -> Unit)?): String {
        val ids = chatIds(listOf("user" to prompt), MAX_GEN_PROMPT_TOKENS)
        return run(ids, mask, maxNew, 0f, onText)
    }

    override fun chat(messages: List<Pair<String, String>>, mask: FloatArray, maxNew: Int, temperature: Float, onText: (String) -> Unit): String {
        if (messages.isEmpty()) throw BridgeError("invalid_input", "Mesaj yok.")
        // A long conversation keeps its newest turns: the oldest ones (after a system message) drop out first.
        var turns = messages
        var ids = chatIds(turns, Int.MAX_VALUE)
        while (ids.size > MAX_CHAT_TOKENS && turns.size > 1) {
            val first = if (turns[0].first == "system") 1 else 0
            if (turns.size - first <= 1) break
            turns = turns.take(first) + turns.drop(first + 1)
            ids = chatIds(turns, Int.MAX_VALUE)
        }
        if (ids.size > MAX_CHAT_TOKENS) {
            throw BridgeError("invalid_input", "Mesaj çok uzun: telefonda en fazla $MAX_CHAT_TOKENS token (bu: ${ids.size}).")
        }
        return run(ids, mask, maxNew, temperature.coerceIn(0f, 2f), onText)
    }

    private fun chatIds(turns: List<Pair<String, String>>, max: Int): IntArray {
        val text = LlamaNative.chatPrompt(
            handle, turns.map { it.first.toByteArray() }.toTypedArray(), turns.map { it.second.toByteArray() }.toTypedArray(),
        )
        val ids = LlamaNative.tokenize(handle, text, true, true)
        if (ids.size > max) throw BridgeError("invalid_input", "Telefonda istem en fazla $max token olabilir (bu: ${ids.size}).")
        return ids
    }

    private fun run(ids: IntArray, mask: FloatArray, maxNew: Int, temperature: Float, onText: ((String) -> Unit)?): String {
        cancel.set(false)
        val keep = if (mask === ones || !headsOk) null else mask
        val sink = LlamaNative.TextSink { bytes ->
            onText?.invoke(String(bytes, Charsets.UTF_8))
            !cancel.get()
        }
        val seed = (System.nanoTime() and 0x7fffffff).toInt()
        val out = String(LlamaNative.generate(handle, ids, maxNew, keep, temperature, seed, sink), Charsets.UTF_8)
        checkCancel()
        return out
    }

    companion object {
        const val MAX_VIEW_TOKENS = 128
        const val MAX_KL_TOKENS = 256
        const val MAX_GEN_PROMPT_TOKENS = 1024
        /** The conversation sent to the model; the rest of the 2048-token context is for the answer. */
        const val MAX_CHAT_TOKENS = 1280
        const val KL_POSITIONS = 16

        @Volatile private var systemLine: String? = null

        /** CPU backend variants are in the app's native library folder; the best one for this phone is loaded. */
        fun systemInfo(nativeLibDir: String): String =
            systemLine ?: String(LlamaNative.init(nativeLibDir.toByteArray()), Charsets.UTF_8).also { systemLine = it }

        fun open(file: File, nativeLibDir: String, threads: Int, cancel: AtomicBoolean): LlamaEngine {
            systemInfo(nativeLibDir)
            val h = LlamaNative.load(file.absolutePath.toByteArray(), threads, CONTEXT)
            if (h == 0L) throw BridgeError("invalid_input", String(LlamaNative.lastError(), Charsets.UTF_8))
            val info = JSONObject(String(LlamaNative.info(h), Charsets.UTF_8))
            return LlamaEngine(h, info, cancel)
        }

        /** Room for a 1024-token message and a long answer, small enough for phones with 4 GB. */
        private const val CONTEXT = 2048
    }
}
