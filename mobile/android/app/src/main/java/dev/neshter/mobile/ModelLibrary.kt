package dev.neshter.mobile

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileNotFoundException
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The models on the phone: the bundled ONNX export (when the APK or the app folder has it) and GGUF files the user
 * brought in. A GGUF lives in files/models/<id>.gguf with <id>.json beside it: what llama.cpp read from it when it
 * came in (layers, heads, name...) and its sha256, so the list needs no model to be opened.
 */
class ModelLibrary(private val ctx: Context, private val nativeLibDir: String) {
    private val dir = File(ctx.filesDir, "models").apply { mkdirs() }

    fun bundledAvailable(): Boolean =
        try {
            ctx.assets.openFd("model/engine.onnx").close()
            true
        } catch (_: FileNotFoundException) {
            File(ctx.getExternalFilesDir(null), "engine.onnx").isFile
        }

    fun ggufFile(id: String): File {
        if (!ID.matches(id)) throw BridgeError("invalid_input", "Geçersiz model kimliği: $id")
        val f = File(dir, "$id.gguf")
        if (!f.isFile) throw BridgeError("not_found", "Model bulunamadı: $id")
        return f
    }

    fun ggufInfo(id: String): JSONObject = JSONObject(File(dir, "$id.json").readText())

    /** Every model as {id, kind, name, size_bytes, info}. */
    fun list(bundledMeta: () -> JSONObject): JSONArray {
        val out = JSONArray()
        if (bundledAvailable()) {
            val m = bundledMeta()
            out.put(JSONObject().put("id", BUNDLED).put("kind", "onnx").put("name", m.getString("name")).put("size_bytes", m.optLong("size_bytes")).put("info", m))
        }
        dir.listFiles { f -> f.name.endsWith(".json") }?.sortedBy { it.name }?.forEach { side ->
            val id = side.name.removeSuffix(".json")
            if (!File(dir, "$id.gguf").isFile) return@forEach
            val info = JSONObject(side.readText())
            out.put(JSONObject().put("id", id).put("kind", "gguf").put("name", info.getString("name")).put("size_bytes", info.optLong("file_bytes")).put("info", info))
        }
        return out
    }

    /**
     * Copies a picked file in, checking that it starts with the GGUF magic, hashing it on the way, then opens it once
     * with llama.cpp so a file it cannot run is refused here and not later.
     */
    fun import(uri: Uri, cancel: AtomicBoolean, progress: (done: Long, total: Long) -> Unit): JSONObject {
        val resolver = ctx.contentResolver
        var display = "model.gguf"
        var total = -1L
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                display = c.getString(0) ?: display
                if (!c.isNull(1)) total = c.getLong(1)
            }
        }
        val base = display.removeSuffix(".gguf").removeSuffix(".GGUF").replace(Regex("[^A-Za-z0-9._-]+"), "-").trim('-', '.').take(80).ifEmpty { "model" }
        var id = base
        var n = 2
        while (File(dir, "$id.gguf").exists()) id = "$base-${n++}"

        val tmp = File(dir, "$id.part")
        val sha = MessageDigest.getInstance("SHA-256")
        try {
            val input = resolver.openInputStream(uri) ?: throw BridgeError("not_found", "Dosya açılamadı.")
            input.use { inp ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(1 shl 20)
                    var done = 0L
                    var first = true
                    var lastReport = 0L
                    while (true) {
                        if (cancel.get()) throw BridgeError("cancelled", "İptal edildi.")
                        val r = inp.read(buf)
                        if (r < 0) break
                        if (first) {
                            if (r < 4 || buf[0] != 'G'.code.toByte() || buf[1] != 'G'.code.toByte() || buf[2] != 'U'.code.toByte() || buf[3] != 'F'.code.toByte()) {
                                throw BridgeError("invalid_input", "Bu bir GGUF dosyası değil. Hugging Face'te adı .gguf ile biten bir dosya seç.")
                            }
                            first = false
                        }
                        out.write(buf, 0, r)
                        sha.update(buf, 0, r)
                        done += r
                        if (done - lastReport >= (16 shl 20)) {
                            lastReport = done
                            progress(done, total)
                        }
                    }
                    if (first) throw BridgeError("invalid_input", "Dosya boş.")
                }
            }
            val file = File(dir, "$id.gguf")
            if (!tmp.renameTo(file)) throw BridgeError("internal", "Dosya yerine taşınamadı.")
            val info = try {
                LlamaEngine.open(file, nativeLibDir, 2, AtomicBoolean(false)).use { it.info }
            } catch (e: BridgeError) {
                file.delete()
                throw e
            }
            info.put("id", id)
                .put("file_bytes", file.length())
                .put("sha256", sha.digest().joinToString("") { "%02x".format(it) })
                .put("family", info.optString("arch"))
                .put("source_name", display)
                // The file name ("SmolLM2-135M-Instruct-Q8_0") says more than general.name, which is often a training tag.
                .put("model_name", info.optString("name"))
                .put("name", display.removeSuffix(".gguf").removeSuffix(".GGUF"))
            File(dir, "$id.json").writeText(info.toString())
            return info
        } finally {
            tmp.delete()
        }
    }

    fun delete(id: String) {
        if (id == BUNDLED) throw BridgeError("invalid_input", "Uygulamayla gelen model silinemez.")
        ggufFile(id).delete()
        File(dir, "$id.json").delete()
    }

    companion object {
        const val BUNDLED = "bundled"
        private val ID = Regex("[A-Za-z0-9._-]{1,90}")
    }
}
