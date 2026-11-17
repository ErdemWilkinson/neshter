package dev.neshter.mobile

import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * Qwen2's byte-level BPE, the same token ids as the Hugging Face tokenizer (checked against golden ids made by
 * Python in the unit test). Reads the three plain files the exporter writes next to the engine: vocab.txt (line =
 * id), merges.txt (rank = line), added.txt ("id<TAB>text<TAB>special").
 */
class Qwen2Tokenizer(vocab: List<String>, merges: List<String>, added: List<Added>) {
    data class Added(val id: Int, val content: String, val special: Boolean)

    private val encoder = HashMap<String, Int>(vocab.size * 2)
    private val pieces = arrayOfNulls<String>(maxOf(vocab.size, (added.maxOfOrNull { it.id } ?: 0) + 1))
    private val ranks = HashMap<String, Int>(merges.size * 2)
    private val addedByLength = added.sortedByDescending { it.content.length }
    private val addedById = added.associateBy { it.id }
    private val cache = object : LinkedHashMap<String, IntArray>(4096, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, IntArray>) = size > 20_000
    }

    init {
        vocab.forEachIndexed { i, piece ->
            encoder[piece] = i
            pieces[i] = piece
        }
        merges.forEachIndexed { i, m -> ranks[m] = i }
        for (a in added) pieces[a.id] = a.content
    }

    val size: Int get() = pieces.size

    /** Token ids of `text` (no special tokens are added: Qwen2 has no BOS). Special-token text becomes its id. */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        var start = 0
        while (start < text.length) {
            val (at, hit) = nextAdded(text, start)
            val end = at ?: text.length
            if (end > start) encodeOrdinary(text.substring(start, end), out)
            if (hit == null) break
            out.add(hit.id)
            start = end + hit.content.length
        }
        return out.toIntArray()
    }

    /** The token as Neşter shows it: its bytes decoded as UTF-8, broken sequences as U+FFFD. */
    fun piece(id: Int): String {
        val added = addedById[id]
        if (added != null) return added.content
        return String(bytesOf(pieces.getOrNull(id) ?: return ""), Charsets.UTF_8)
    }

    /** The text of a run of ids (special tokens left out), decoded as one byte string like the HF decoder. */
    fun decode(ids: IntArray, skipSpecial: Boolean = true): String {
        val buf = java.io.ByteArrayOutputStream()
        for (id in ids) {
            val added = addedById[id]
            when {
                added == null -> buf.write(bytesOf(pieces.getOrNull(id) ?: continue))
                added.special && skipSpecial -> Unit
                else -> buf.write(added.content.toByteArray(Charsets.UTF_8))
            }
        }
        return String(buf.toByteArray(), Charsets.UTF_8)
    }

    private fun nextAdded(text: String, from: Int): Pair<Int?, Added?> {
        var bestAt = -1
        var best: Added? = null
        for (a in addedByLength) { // longest first: at the same position the longer token wins
            val i = text.indexOf(a.content, from)
            if (i >= 0 && (bestAt < 0 || i < bestAt)) {
                bestAt = i
                best = a
            }
        }
        return if (best == null) null to null else bestAt to best
    }

    private fun encodeOrdinary(raw: String, out: MutableList<Int>) {
        val text = Normalizer.normalize(raw, Normalizer.Form.NFC)
        val m = SPLIT.matcher(text)
        while (m.find()) {
            val word = m.group()
            val ids = synchronized(cache) { cache[word] } ?: bpe(word).also { synchronized(cache) { cache[word] = it } }
            for (id in ids) out.add(id)
        }
    }

    private fun bpe(word: String): IntArray {
        val bytes = word.toByteArray(Charsets.UTF_8)
        var parts = ArrayList<String>(bytes.size)
        for (b in bytes) parts.add(BYTE_TO_CHAR[b.toInt() and 0xff].toString())
        while (parts.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestPair: String? = null
            for (i in 0 until parts.size - 1) {
                val r = ranks[parts[i] + " " + parts[i + 1]] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestPair = parts[i] + " " + parts[i + 1]
                }
            }
            if (bestPair == null) break
            val merged = ArrayList<String>(parts.size)
            var i = 0
            while (i < parts.size) {
                if (i < parts.size - 1 && parts[i] + " " + parts[i + 1] == bestPair) {
                    merged.add(parts[i] + parts[i + 1])
                    i += 2
                } else {
                    merged.add(parts[i])
                    i += 1
                }
            }
            parts = merged
        }
        return IntArray(parts.size) { encoder[parts[it]] ?: error("token yok: ${parts[it]}") }
    }

    private fun bytesOf(piece: String): ByteArray =
        ByteArray(piece.length) { (CHAR_TO_BYTE[piece[it]] ?: '?'.code).toByte() }

    companion object {
        // Unicode White_Space, written out: Android's regex engine (ICU) and the JVM's differ on what \s means.
        private const val WS = "\\t\\n\\u000B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000"

        /** Qwen2's pre-tokenizer split, with \s spelled out and the case-insensitive contractions expanded. */
        val SPLIT: Pattern = Pattern.compile(
            "'[sSſ]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]" +
                "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +
                "|\\p{N}" +
                "| ?[^$WS\\p{L}\\p{N}]+[\\r\\n]*" +
                "|[$WS]*[\\r\\n]+" +
                "|[$WS]+(?![^$WS])" +
                "|[$WS]+",
        )

        /** GPT-2's byte <-> printable character table. */
        private val BYTE_TO_CHAR = CharArray(256)
        private val CHAR_TO_BYTE = HashMap<Char, Int>(512)

        init {
            val direct = (('!'.code..'~'.code) + ('¡'.code..'¬'.code) + ('®'.code..'ÿ'.code)).toSet()
            var n = 0
            for (b in 0 until 256) {
                val c = if (b in direct) b else 256 + n++
                BYTE_TO_CHAR[b] = c.toChar()
                CHAR_TO_BYTE[c.toChar()] = b
            }
        }

        private fun lines(input: InputStream): List<String> =
            input.bufferedReader(Charsets.UTF_8).use(BufferedReader::readLines).let { all ->
                if (all.isNotEmpty() && all.last().isEmpty()) all.dropLast(1) else all
            }

        fun load(open: (String) -> InputStream): Qwen2Tokenizer {
            val added = lines(open("added.txt")).map { line ->
                val (id, text, special) = line.split('\t')
                Added(id.toInt(), text, special == "1")
            }
            return Qwen2Tokenizer(lines(open("vocab.txt")), lines(open("merges.txt")), added)
        }

        fun load(dir: File): Qwen2Tokenizer = load { File(dir, it).inputStream() }
    }
}
