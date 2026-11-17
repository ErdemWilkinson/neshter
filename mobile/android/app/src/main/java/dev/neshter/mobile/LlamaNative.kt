package dev.neshter.mobile

/** JNI declarations for libneshter_llama.so (src/main/cpp/jni_bridge.cpp). Text crosses as UTF-8 bytes. */
object LlamaNative {
    init {
        System.loadLibrary("neshter_llama")
    }

    fun interface TextSink {
        /** One printable piece of the answer; false stops the generation. */
        fun onText(bytes: ByteArray): Boolean
    }

    /** Loads the CPU backend variants found in [libDir]; returns llama.cpp's system line (SIMD features). */
    external fun init(libDir: ByteArray): ByteArray
    external fun load(path: ByteArray, threads: Int, nCtx: Int): Long
    external fun lastError(): ByteArray
    external fun free(handle: Long)
    external fun info(handle: Long): ByteArray
    external fun tokenize(handle: Long, text: ByteArray, addSpecial: Boolean, parseSpecial: Boolean): IntArray
    external fun pieces(handle: Long, tokens: IntArray): Array<ByteArray>
    external fun chatPrompt(handle: Long, roles: Array<ByteArray>, contents: Array<ByteArray>): ByteArray
    external fun cancel(handle: Long)
    external fun generate(
        handle: Long, prompt: IntArray, maxNew: Int, keep: FloatArray?, temperature: Float, seed: Int, sink: TextSink?,
    ): ByteArray

    /** [n_tokens, position, final, k, then per row: k ids, k probs, p_final, rank]; empty when the model has no lens. */
    external fun lens(handle: Long, tokens: IntArray, position: Int, k: Int): FloatArray

    /** [power, attn, logprobs]: power [L][H][T], attn [L][H][T][T], logprobs [rows][vocab]; null when not asked for. */
    external fun analyze(
        handle: Long, tokens: IntArray, keep: FloatArray?, power: Boolean, attn: Boolean, logprobRows: Int,
    ): Array<FloatArray?>
}
