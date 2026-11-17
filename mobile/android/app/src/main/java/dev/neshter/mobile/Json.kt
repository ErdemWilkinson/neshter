package dev.neshter.mobile

import org.json.JSONObject

/** Fast JSON for big number arrays (org.json is slow for a [24][14][T] tensor). Non-finite numbers become 0. */
class Json {
    private val b = StringBuilder(1 shl 14)

    fun raw(s: String) = apply { b.append(s) }

    fun str(s: String) = apply { b.append(JSONObject.quote(s)) }

    fun num(v: Float) = apply { if (v.isFinite()) b.append(v) else b.append('0') }

    fun num(v: Double) = apply { if (v.isFinite()) b.append(v) else b.append('0') }

    fun num(v: Long) = apply { b.append(v) }

    fun key(k: String) = apply {
        if (b.isNotEmpty() && b.last() != '{' && b.last() != '[') b.append(',')
        b.append(JSONObject.quote(k)).append(':')
    }

    fun obj(body: Json.() -> Unit) = apply { sep(); b.append('{'); body(); b.append('}') }

    fun arr(body: Json.() -> Unit) = apply { sep(); b.append('['); body(); b.append(']') }

    fun floats(src: FloatArray, from: Int, n: Int) = arr {
        for (i in 0 until n) {
            if (i > 0) b.append(',')
            num(src[from + i])
        }
    }

    fun strings(xs: List<String>) = arr {
        xs.forEachIndexed { i, s ->
            if (i > 0) b.append(',')
            str(s)
        }
    }

    private fun sep() {
        if (b.isNotEmpty() && b.last() != '{' && b.last() != '[' && b.last() != ':') b.append(',')
    }

    override fun toString() = b.toString()

    companion object {
        fun build(body: Json.() -> Unit): String = Json().apply(body).toString()
    }
}
