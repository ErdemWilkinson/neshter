package dev.neshter.mobile

import org.json.JSONObject

/** meta.json, written by the exporter next to the engine. `raw` goes to the web UI as the device info. */
class Meta(val raw: JSONObject) {
    val nLayers = raw.getInt("n_layers")
    val nHeads = raw.getInt("n_heads")
    val nKvHeads = raw.getInt("n_kv_heads")
    val headDim = raw.getInt("head_dim")
    val dModel = raw.getInt("d_model")
    val vocab = raw.getInt("vocab")
    val eos: Set<Int> = raw.getJSONArray("eos").let { a -> (0 until a.length()).map(a::getInt).toSet() } +
        EXTRA_STOP
    val chatPrefix: String = raw.getString("chat_prefix")
    val chatSuffix: String = raw.getString("chat_suffix")

    companion object {
        /** Qwen2.5-Instruct's generation_config also stops at <|endoftext|>. */
        private const val EXTRA_STOP = 151643

        fun parse(text: String) = Meta(JSONObject(text))
    }
}
