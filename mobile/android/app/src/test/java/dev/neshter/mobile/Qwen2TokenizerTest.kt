package dev.neshter.mobile

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File

/** The Kotlin tokenizer against ids made by Hugging Face (mobile/export/make_golden.py). */
class Qwen2TokenizerTest {
    companion object {
        private var tok: Qwen2Tokenizer? = null

        @BeforeClass
        @JvmStatic
        fun setUp() {
            val dir = System.getProperty("neshter.modelDir")?.let(::File)
            if (dir != null && File(dir, "vocab.txt").exists()) tok = Qwen2Tokenizer.load(dir)
        }

        private data class Golden(val text: String, val ids: IntArray, val pieces: List<String>, val decoded: String)

        /** A tiny reader for the golden file (a flat array of objects with string and int-array fields). */
        private fun golden(): List<Golden> {
            val src = Qwen2TokenizerTest::class.java.getResource("/tokenizer_golden.json")!!.readText(Charsets.UTF_8)
            return MiniJson(src).parse().let { it as List<*> }.map { o ->
                val m = o as Map<*, *>
                Golden(
                    m["text"] as String,
                    (m["ids"] as List<*>).map { (it as Number).toInt() }.toIntArray(),
                    (m["pieces"] as List<*>).map { it as String },
                    m["decoded"] as String,
                )
            }
        }
    }

    @Test
    fun matchesHuggingFace() {
        val t = tok
        assumeTrue("model files not built (neshter.modelDir)", t != null)
        for (g in golden()) {
            assertArrayEquals("ids of ${g.text}", g.ids, t!!.encode(g.text))
            assertEquals("pieces of ${g.text}", g.pieces, g.ids.map(t::piece))
            assertEquals("decode of ${g.text}", g.decoded, t.decode(g.ids))
        }
    }

    @Test
    fun chatTemplateTokensAreSpecial() {
        val t = tok
        assumeTrue(t != null)
        val ids = t!!.encode("<|im_start|>user\nhi<|im_end|>")
        assertEquals(151644, ids.first())
        assertEquals(151645, ids.last())
        assertEquals("user\nhi", t.decode(ids))
    }
}
