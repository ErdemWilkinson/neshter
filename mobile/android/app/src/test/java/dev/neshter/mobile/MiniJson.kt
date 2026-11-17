package dev.neshter.mobile

/** A minimal JSON reader for test fixtures (Android's org.json is a stub in JVM unit tests). */
class MiniJson(private val s: String) {
    private var i = 0

    fun parse(): Any? = value().also { ws() }

    private fun ws() {
        while (i < s.length && s[i].isWhitespace()) i++
    }

    private fun value(): Any? {
        ws()
        return when (val c = s[i]) {
            '{' -> obj()
            '[' -> arr()
            '"' -> str()
            't' -> { i += 4; true }
            'f' -> { i += 5; false }
            'n' -> { i += 4; null }
            else -> if (c == '-' || c.isDigit()) num() else error("JSON: '$c' at $i")
        }
    }

    private fun obj(): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        i++
        ws()
        if (s[i] == '}') return m.also { i++ }
        while (true) {
            ws()
            val k = str()
            ws()
            i++ // :
            m[k] = value()
            ws()
            if (s[i++] == '}') return m
        }
    }

    private fun arr(): List<Any?> {
        val a = ArrayList<Any?>()
        i++
        ws()
        if (s[i] == ']') return a.also { i++ }
        while (true) {
            a.add(value())
            ws()
            if (s[i++] == ']') return a
        }
    }

    private fun str(): String {
        val b = StringBuilder()
        i++ // "
        while (true) {
            val c = s[i++]
            when (c) {
                '"' -> return b.toString()
                '\\' -> when (val e = s[i++]) {
                    'n' -> b.append('\n')
                    't' -> b.append('\t')
                    'r' -> b.append('\r')
                    'b' -> b.append('\b')
                    'f' -> b.append('\u000C')
                    'u' -> { b.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                    else -> b.append(e)
                }
                else -> b.append(c)
            }
        }
    }

    private fun num(): Number {
        val start = i
        while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
        val t = s.substring(start, i)
        return if (t.any { it in ".eE" }) t.toDouble() else t.toLong()
    }
}
