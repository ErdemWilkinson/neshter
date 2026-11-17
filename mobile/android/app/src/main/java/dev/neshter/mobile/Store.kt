package dev.neshter.mobile

import java.io.File

/** The phone's stand-in for the server's drafts and prefs: one JSON file per key, written atomically. */
class Store(private val dir: File) {
    init {
        dir.mkdirs()
    }

    private fun file(key: String): File {
        require(KEY.matches(key)) { "geçersiz anahtar: $key" }
        return File(dir, "$key.json")
    }

    fun get(key: String): String? = file(key).takeIf(File::exists)?.readText(Charsets.UTF_8)

    fun put(key: String, json: String) {
        val target = file(key)
        val tmp = File(dir, "$key.json.tmp")
        tmp.writeText(json, Charsets.UTF_8)
        if (!tmp.renameTo(target)) {
            target.delete()
            check(tmp.renameTo(target)) { "yazılamadı: $key" }
        }
    }

    fun delete(key: String) {
        file(key).delete()
    }

    fun list(prefix: String): List<String> =
        (dir.listFiles() ?: emptyArray())
            .filter { it.name.startsWith(prefix) && it.name.endsWith(".json") }
            .sortedBy { it.name }
            .map { it.readText(Charsets.UTF_8) }

    companion object {
        private val KEY = Regex("[A-Za-z0-9._-]{1,128}")
    }
}
