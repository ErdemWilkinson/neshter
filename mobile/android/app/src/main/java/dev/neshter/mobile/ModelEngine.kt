package dev.neshter.mobile

import org.json.JSONObject
import java.io.Closeable

/**
 * What the web UI can ask of a model on the phone. Two kinds answer: [Engine] (the bundled ONNX export) and
 * [LlamaEngine] (any GGUF file through llama.cpp). The JSON shapes are the desktop server's.
 */
interface ModelEngine : Closeable {
    /** Device info for the UI: name, n_layers, n_heads, ... (merged into the `info` reply). */
    val info: JSONObject

    fun mask(rows: List<List<Number>>?): FloatArray
    fun activity(text: String): String
    fun attention(text: String, layer: Int): String
    fun lens(text: String, position: Int): String
    fun klBase(prompts: List<String>)
    fun kl(mask: FloatArray): List<Double>
    fun probe(text: String, mask: FloatArray): String

    /** Greedy answer to one chat message; [onText] sees it as it is written. */
    fun generate(prompt: String, mask: FloatArray, maxNew: Int, onText: ((String) -> Unit)? = null): String

    /**
     * A conversation: (role, content) pairs ending with the user's turn. The default answers the last user message
     * on its own (the bundled export has no multi-turn template); [LlamaEngine] uses the model's chat template.
     */
    fun chat(messages: List<Pair<String, String>>, mask: FloatArray, maxNew: Int, temperature: Float, onText: (String) -> Unit): String {
        val last = messages.lastOrNull { it.first == "user" }?.second ?: throw BridgeError("invalid_input", "Mesaj yok.")
        return generate(last, mask, maxNew, onText)
    }
}
