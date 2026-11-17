package dev.neshter.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp

class BridgeError(val code: String, message: String) : Exception(message)

/**
 * The exported Qwen2 engine (mobile/export/export_qwen2.py) and the measurements the web UI asks for, in the
 * shapes the desktop server answers with. One forward pass gives everything (ORT does not prune unrequested
 * outputs), so each method runs as few passes as it can; text generation uses the engine's key/value cache.
 */
class Engine(
    model: ByteBuffer,
    private val meta: Meta,
    private val tok: Qwen2Tokenizer,
    threads: Int,
    private val cancel: AtomicBoolean,
) : ModelEngine {
    override val info: JSONObject get() = meta.raw
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = OrtSession.SessionOptions().use { o ->
        o.setIntraOpNumThreads(threads)
        env.createSession(model, o)
    }
    private val L = meta.nLayers
    private val H = meta.nHeads
    private val KV = meta.nKvHeads
    private val D = meta.headDim
    private val M = meta.dModel
    private val V = meta.vocab
    private val ones = FloatArray(L * H) { 1f }

    /**
     * KL baselines: per neutral prompt, its ids, the original model's probabilities p of its last `keep` tokens and
     * Σ p·log p over them. KL(p‖q) = Σ p·log p − Σ p·log q, so each measurement is one dot product (no exp per call).
     */
    private class Base(val ids: IntArray, val keep: Int, val p: FloatArray, val plogp: Double)
    private var bases: List<Base> = emptyList()

    override fun close() = session.close()

    // --- tensors -------------------------------------------------------------------------------------------

    private fun longs(xs: LongArray, vararg shape: Long) = OnnxTensor.createTensor(env, LongBuffer.wrap(xs), shape)

    private fun floats(xs: FloatArray, vararg shape: Long) = OnnxTensor.createTensor(env, FloatBuffer.wrap(xs), shape)

    private fun emptyPast() = floats(FloatArray(0), L.toLong(), KV.toLong(), 0, D.toLong())

    /**
     * One pass. `past` is the previous pass's result (its present_k/present_v) or null. The caller closes the
     * returned result.
     */
    private fun run(
        ids: IntArray,
        mask: FloatArray,
        keep: Int,
        outputs: Set<String>,
        lensHidden: FloatArray = FloatArray(0),
        lensTarget: Int = 0,
        past: OrtSession.Result? = null,
    ): OrtSession.Result {
        val made = ArrayList<OnnxTensor>()
        fun <T : OnnxTensor> own(t: T): T = t.also { made.add(it) }
        try {
            val pastK = past?.get("present_k")?.get() as OnnxTensor? ?: own(emptyPast())
            val pastV = past?.get("present_v")?.get() as OnnxTensor? ?: own(emptyPast())
            val inputs = mapOf(
                "input_ids" to own(longs(LongArray(ids.size) { ids[it].toLong() }, 1, ids.size.toLong())),
                "head_mask" to own(floats(mask, L.toLong(), H.toLong())),
                "keep" to own(longs(longArrayOf(keep.toLong()))),
                "lens_hidden" to own(floats(lensHidden, (lensHidden.size / M).toLong(), M.toLong())),
                "lens_target" to own(longs(longArrayOf(lensTarget.toLong()))),
                "past_k" to pastK,
                "past_v" to pastV,
            )
            return session.run(inputs, outputs)
        } catch (e: OutOfMemoryError) {
            throw BridgeError("oom", "Telefonun belleği bu istem için yetmedi; daha kısa bir metin dene.")
        } finally {
            made.forEach(OnnxTensor::close)
        }
    }

    private fun OrtSession.Result.floatsOf(name: String): FloatArray {
        val b = (get(name).get() as OnnxTensor).floatBuffer
        return FloatArray(b.remaining()).also { b.get(it) }
    }

    private fun OrtSession.Result.longsOf(name: String): LongArray {
        val b = (get(name).get() as OnnxTensor).longBuffer
        return LongArray(b.remaining()).also { b.get(it) }
    }

    // --- helpers -------------------------------------------------------------------------------------------

    fun encode(text: String, max: Int): IntArray {
        val ids = tok.encode(text)
        if (ids.isEmpty()) throw BridgeError("invalid_input", "Boş istem: okunacak token yok.")
        if (ids.size > max) throw BridgeError("invalid_input", "Telefonda istem en fazla $max token olabilir (bu: ${ids.size}).")
        return ids
    }

    override fun mask(rows: List<List<Number>>?): FloatArray {
        if (rows == null) return ones
        if (rows.size != L || rows.any { it.size != H }) throw BridgeError("invalid_input", "Maske $L×$H olmalı.")
        return FloatArray(L * H) { rows[it / H][it % H].toFloat() }
    }

    private fun argmax(x: FloatArray, from: Int, n: Int): Int {
        var best = from
        for (i in from until from + n) if (x[i] > x[best]) best = i
        return best - from
    }

    private fun topK(logp: FloatArray, from: Int, k: Int): List<Pair<Int, Float>> {
        val idx = IntArray(k) { -1 }
        for (v in 0 until V) {
            val x = logp[from + v]
            var j = k - 1
            if (idx[j] >= 0 && x <= logp[from + idx[j]]) continue
            while (j > 0 && (idx[j - 1] < 0 || x > logp[from + idx[j - 1]])) {
                idx[j] = idx[j - 1]
                j--
            }
            idx[j] = v
        }
        return idx.filter { it >= 0 }.map { it to exp(logp[from + it]) }
    }

    private fun checkCancel() {
        if (cancel.get()) throw BridgeError("cancelled", "İptal edildi.")
    }

    private fun Json.tokenRow(id: Int, p: Float) = obj {
        key("token_id").num(id.toLong())
        key("text").str(tok.piece(id))
        key("p").num(p)
    }

    // --- measurements --------------------------------------------------------------------------------------

    /** `power_raw` [layer][head][token], like `POST /api/activity` before the per-token scaling (done in TS). */
    override fun activity(text: String): String {
        val ids = encode(text, MAX_VIEW_TOKENS)
        val T = ids.size
        val power = run(ids, ones, 1, setOf("power")).use { it.floatsOf("power") }
        return Json.build {
            obj {
                key("tokens").strings(ids.map(tok::piece))
                key("power_raw").arr {
                    for (l in 0 until L) arr { for (h in 0 until H) floats(power, (l * H + h) * T, T) }
                }
            }
        }
    }

    /** `POST /api/attention`: [head][query][key] of one layer. */
    override fun attention(text: String, layer: Int): String {
        if (layer !in 0 until L) throw BridgeError("invalid_input", "Katman $layer aralık dışı (0..${L - 1}).")
        val ids = encode(text, MAX_VIEW_TOKENS)
        val T = ids.size
        val attn = run(ids, ones, 1, setOf("attn")).use { it.floatsOf("attn") }
        return Json.build {
            obj {
                key("tokens").strings(ids.map(tok::piece))
                key("layer").num(layer.toLong())
                key("patterns").arr {
                    for (h in 0 until H) arr { for (q in 0 until T) floats(attn, ((layer * H + h) * T + q) * T, T) }
                }
            }
        }
    }

    /** `POST /api/lens`: what the model would answer after every layer, row 0 the embedding (layer −1). */
    override fun lens(text: String, position: Int): String {
        val ids = encode(text, MAX_VIEW_TOKENS)
        val T = ids.size
        if (position !in -T until T) throw BridgeError("invalid_input", "position $position aralık dışı (istem $T token).")
        val pos = if (position < 0) T + position else position
        val keep = T - pos // the kept rows start at pos
        val (logp, resid) = run(ids, ones, keep, setOf("logprobs", "resid")).use { it.floatsOf("logprobs") to it.floatsOf("resid") }
        val y = argmax(logp, 0, V)
        val rows = L + 1
        val hidden = FloatArray(rows * M)
        for (r in 0 until rows) System.arraycopy(resid, (r * T + pos) * M, hidden, r * M, M)
        val out = run(intArrayOf(ids[0]), ones, 1, setOf("lens_p", "lens_id", "lens_p_target", "lens_rank_target"), hidden, y)
        out.use { o ->
            val p = o.floatsOf("lens_p")
            val id = o.longsOf("lens_id")
            val pT = o.floatsOf("lens_p_target")
            val rank = o.longsOf("lens_rank_target")
            val k = p.size / rows
            return Json.build {
                obj {
                    key("tokens").strings(ids.map(tok::piece))
                    key("position").num(pos.toLong())
                    key("final").obj { key("token_id").num(y.toLong()); key("text").str(tok.piece(y)) }
                    key("layers").arr {
                        for (r in 0 until rows) obj {
                            key("layer").num((r - 1).toLong())
                            key("top").arr { for (j in 0 until k) tokenRow(id[r * k + j].toInt(), p[r * k + j]) }
                            key("p_final").num(pT[r])
                            key("rank_final").num(rank[r])
                        }
                    }
                }
            }
        }
    }

    /** The original model's log-probs on the neutral prompts, kept for every `kl` call that follows. */
    override fun klBase(prompts: List<String>) {
        cancel.set(false)
        bases = prompts.map { text ->
            checkCancel()
            val ids = encode(text, MAX_KL_TOKENS)
            val keep = minOf(KL_POSITIONS, ids.size)
            val logp = run(ids, ones, keep, setOf("logprobs")).use { it.floatsOf("logprobs") }
            var plogp = 0.0
            for (i in logp.indices) {
                val lp = logp[i].toDouble()
                val p = exp(lp)
                plogp += p * lp
                logp[i] = p.toFloat() // reuse the array for p
            }
            Base(ids, keep, logp, plogp)
        }
    }

    /** Per prompt: KL(original ‖ masked), averaged over the last `keep` positions (Neşter's kl_effect). */
    override fun kl(mask: FloatArray): List<Double> {
        if (bases.isEmpty()) throw BridgeError("internal", "Önce klBase çağrılmalı.")
        cancel.set(false)
        return bases.map { b ->
            checkCancel()
            val q = run(b.ids, mask, b.keep, setOf("logprobs")).use { it.floatsOf("logprobs") }
            var plogq = 0.0
            for (i in q.indices) plogq += b.p[i].toDouble() * q[i]
            (b.plogp - plogq) / b.keep
        }
    }

    /** The answer card's hundred dots: the next token with and without the cut (metrics.probe_answer). */
    override fun probe(text: String, mask: FloatArray): String {
        val ids = encode(text, MAX_VIEW_TOKENS)
        val before = run(ids, ones, 1, setOf("logprobs")).use { it.floatsOf("logprobs") }
        val after = run(ids, mask, 1, setOf("logprobs")).use { it.floatsOf("logprobs") }
        val y = argmax(before, 0, V)
        return Json.build {
            obj {
                key("answer").obj { key("token_id").num(y.toLong()); key("text").str(tok.piece(y)) }
                key("p_before").num(exp(before[y]))
                key("p_after").num(exp(after[y]))
                key("top_before").arr { topK(before, 0, 3).forEach { (i, p) -> tokenRow(i, p) } }
                key("top_after").arr { topK(after, 0, 3).forEach { (i, p) -> tokenRow(i, p) } }
            }
        }
    }

    /** Greedy answer to a chat message (the desktop's compare: chat template, do_sample=False). */
    override fun generate(prompt: String, mask: FloatArray, maxNew: Int, onText: ((String) -> Unit)?): String {
        cancel.set(false)
        val ids = encode(meta.chatPrefix + prompt + meta.chatSuffix, MAX_GEN_PROMPT_TOKENS)
        val out = ArrayList<Int>()
        val outs = setOf("logprobs", "present_k", "present_v")
        var result = run(ids, mask, 1, outs)
        try {
            while (out.size < maxNew) {
                checkCancel()
                val next = argmax(result.floatsOf("logprobs"), 0, V)
                if (next in meta.eos) break
                out.add(next)
                val prev = result
                result = run(intArrayOf(next), mask, 1, outs, past = prev)
                prev.close()
            }
        } finally {
            result.close()
        }
        return tok.decode(out.toIntArray())
    }

    companion object {
        /** Attention is [24][14][T][T] floats per pass: on a phone the views stop at this many tokens. */
        const val MAX_VIEW_TOKENS = 128
        const val MAX_KL_TOKENS = 256
        const val MAX_GEN_PROMPT_TOKENS = 512
        const val KL_POSITIONS = 16
    }
}
