package dev.neshter.mobile

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * `window.NeshterNative` for the web UI (web/src/api/device/bridge.ts). `call` returns at once; the work runs on
 * one engine thread (copying a model in runs on its own) and the answer goes back through
 * `window.__neshterReply(id, ok, json)`. Long calls also send `window.__neshterEvent(id, json)` as they go.
 */
class NativeBridge(
    private val ctx: Context,
    private val web: WebView,
    /** Opens the system file picker; the callback gets the chosen document or null. */
    private val pickDocument: ((Uri?) -> Unit) -> Unit,
) {
    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "neshter-engine") }
    private val io = Executors.newSingleThreadExecutor { r -> Thread(r, "neshter-import") }
    private val main = Handler(Looper.getMainLooper())
    private val store = Store(ctx.filesDir.resolve("store"))
    private val cancel = AtomicBoolean(false)
    private val nativeLibDir = ctx.applicationInfo.nativeLibraryDir
    private val library = ModelLibrary(ctx, nativeLibDir)
    private val meta: Meta by lazy { Meta.parse(ctx.assets.open("model/meta.json").reader(Charsets.UTF_8).use { it.readText() }) }
    private var engine: ModelEngine? = null
    private var activeId: String? = null

    @JavascriptInterface
    fun call(id: String, method: String, argsJson: String) {
        if (method == "cancel") { // not queued behind the work it cancels
            cancel.set(true)
            reply(id, true, "null")
            return
        }
        val args = try {
            JSONObject(argsJson.ifEmpty { "{}" })
        } catch (e: Exception) {
            return reply(id, false, error("invalid_input", e.message))
        }
        // Copying a large file in must not hold up the model (or the other way round).
        val pool = if (method == "importModel") io else exec
        pool.execute {
            val started = System.nanoTime()
            val payload = try {
                handle(id, method, args)
            } catch (e: BridgeError) {
                return@execute reply(id, false, error(e.code, e.message))
            } catch (e: OutOfMemoryError) {
                return@execute reply(id, false, error("oom", "Telefonun belleği yetmedi."))
            } catch (e: Exception) {
                Log.e(TAG, "$method failed", e)
                return@execute reply(id, false, error("internal", e.message ?: e.toString()))
            }
            Log.d(TAG, "$method ${(System.nanoTime() - started) / 1_000_000} ms")
            reply(id, true, payload)
        }
    }

    fun close() {
        exec.execute { engine?.close(); engine = null }
        exec.shutdown()
        io.shutdownNow()
    }

    private fun handle(callId: String, method: String, a: JSONObject): String = when (method) {
        "info" -> info()
        "models" -> library.list { bundledInfo() }.toString()
        "importModel" -> importModel(callId)
        "deleteModel" -> { deleteModel(a.getString("id")); "null" }
        "load" -> load(a.optString("id", "").ifEmpty { null }).toString()
        "activity" -> eng().activity(a.getString("text"))
        "lens" -> eng().lens(a.getString("text"), a.optInt("position", -1))
        "attention" -> eng().attention(a.getString("text"), a.getInt("layer"))
        "klBase" -> { eng().klBase(a.getJSONArray("prompts").strings()); "null" }
        "kl" -> eng().let { e -> JSONArray(e.kl(e.mask(a.optJSONArray("mask")?.matrix()))).toString() }
        "probe" -> eng().let { e -> e.probe(a.getString("text"), e.mask(a.optJSONArray("mask")?.matrix())) }
        "generate" -> eng().let { e ->
            val stream = a.optBoolean("stream", false)
            JSONObject.quote(
                e.generate(a.getString("prompt"), e.mask(a.optJSONArray("mask")?.matrix()), a.optInt("max_new", 96),
                    if (stream) { text -> event(callId, JSONObject().put("text", text).toString()) } else null),
            )
        }
        "chat" -> eng().let { e ->
            val msgs = a.getJSONArray("messages").let { arr ->
                (0 until arr.length()).map { arr.getJSONObject(it).let { m -> m.getString("role") to m.getString("content") } }
            }
            JSONObject.quote(
                e.chat(msgs, e.mask(a.optJSONArray("mask")?.matrix()), a.optInt("max_new", 512), a.optDouble("temperature", 0.0).toFloat()) { text ->
                    event(callId, JSONObject().put("text", text).toString())
                },
            )
        }
        "storeGet" -> store.get(a.getString("key")) ?: "null"
        "storePut" -> { store.put(a.getString("key"), a.get("value").toString()); "null" }
        "storeDelete" -> { store.delete(a.getString("key")); "null" }
        "storeList" -> store.list(a.optString("prefix", "")).joinToString(",", "[", "]")
        else -> throw BridgeError("internal", "Bilinmeyen çağrı: $method")
    }

    private fun eng(): ModelEngine = engine ?: throw BridgeError("no_model", "Önce bir model aç.")

    private fun info(): String {
        val mem = ActivityManager.MemoryInfo().also { (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(it) }
        return JSONObject()
            .put("ram_bytes", mem.totalMem)
            .put("cpu_threads", Runtime.getRuntime().availableProcessors())
            .put("active", engine?.let { activeInfo() } ?: JSONObject.NULL)
            .toString()
    }

    /** The bundled export's meta.json as the UI's model info. */
    private fun bundledInfo(): JSONObject = JSONObject(meta.raw.toString())
        .put("id", ModelLibrary.BUNDLED).put("kind", "onnx").put("heads_supported", true).put("lens_supported", true)

    private fun activeInfo(): JSONObject {
        val id = activeId ?: throw BridgeError("no_model", "Önce bir model aç.")
        if (id == ModelLibrary.BUNDLED) return bundledInfo()
        // A GGUF carries no neutral prompts or license line of Neşter's: the bundled set is used for measuring.
        return library.ggufInfo(id)
            .put("kind", "gguf").put("lens_supported", false).put("license", JSONObject.NULL)
            .put("neutral", meta.raw.getJSONArray("neutral"))
    }

    private fun importModel(callId: String): String {
        cancel.set(false)
        val picked = CompletableFuture<Uri?>()
        main.post { pickDocument { picked.complete(it) } }
        val uri = picked.get() ?: throw BridgeError("cancelled", "Dosya seçilmedi.")
        val info = library.import(uri, cancel) { done, total ->
            event(callId, JSONObject().put("done", done).put("total", total).toString())
        }
        return info.toString()
    }

    private fun deleteModel(id: String) {
        if (id == activeId) {
            engine?.close()
            engine = null
            activeId = null
        }
        library.delete(id)
        if (store.get(ACTIVE_KEY) == JSONObject.quote(id)) store.delete(ACTIVE_KEY)
    }

    /** Opens [id] (or the last one opened, or the bundled one, or the first GGUF) in place of the open model. */
    private fun load(id: String?): JSONObject {
        val wanted = id
            ?: store.get(ACTIVE_KEY)?.let { JSONObject("{\"v\":$it}").getString("v") }
            ?: (if (library.bundledAvailable()) ModelLibrary.BUNDLED else null)
            ?: library.list { bundledInfo() }.let { if (it.length() > 0) it.getJSONObject(0).getString("id") else null }
            ?: throw BridgeError("not_found", "Telefonda model yok: Hugging Face'ten indirdiğin bir .gguf dosyasını ekle.")
        if (wanted == activeId && engine != null) return activeInfo()
        engine?.close()
        engine = null
        activeId = null
        val started = System.currentTimeMillis()
        val threads = threadsFor()
        engine = if (wanted == ModelLibrary.BUNDLED) {
            if (!library.bundledAvailable()) throw BridgeError("not_found", "Uygulamayla gelen model bu kurulumda yok.")
            val tok = Qwen2Tokenizer.load { ctx.assets.open("model/$it") }
            Engine(mapEngine(), meta, tok, threads, cancel)
        } else {
            LlamaEngine.open(library.ggufFile(wanted), nativeLibDir, threads, cancel)
        }
        activeId = wanted
        store.put(ACTIVE_KEY, JSONObject.quote(wanted))
        Log.i(TAG, "model $wanted ready in ${System.currentTimeMillis() - started} ms, $threads threads")
        return activeInfo()
    }

    /** Phones mix big and little cores; past four threads the little ones slow a pass down more than they help. */
    private fun threadsFor(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

    /**
     * engine.onnx, memory-mapped rather than copied: from the APK (stored uncompressed) or, in a build without the
     * model (neshter.bundleModel=false), from the app's own folder on shared storage.
     */
    private fun mapEngine(): MappedByteBuffer {
        try {
            val fd = ctx.assets.openFd("model/engine.onnx")
            return FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        } catch (_: FileNotFoundException) {
            val file = File(ctx.getExternalFilesDir(null), "engine.onnx")
            if (!file.isFile) throw BridgeError("not_found", "Model dosyası yok: engine.onnx dosyasını ${file.absolutePath} konumuna kopyala.")
            return RandomAccessFile(file, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }
        }
    }

    private fun reply(id: String, ok: Boolean, payload: String) {
        val js = "window.__neshterReply&&window.__neshterReply(${JSONObject.quote(id)},$ok,${JSONObject.quote(payload)})"
        main.post { web.evaluateJavascript(js, null) }
    }

    private fun event(id: String, payload: String) {
        val js = "window.__neshterEvent&&window.__neshterEvent(${JSONObject.quote(id)},${JSONObject.quote(payload)})"
        main.post { web.evaluateJavascript(js, null) }
    }

    private fun error(code: String, message: String?) =
        JSONObject().put("code", code).put("message", message ?: code).toString()

    private fun JSONArray.strings(): List<String> = (0 until length()).map(::getString)

    private fun JSONArray.matrix(): List<List<Number>> =
        (0 until length()).map { r -> getJSONArray(r).let { row -> (0 until row.length()).map { row.getDouble(it) } } }

    companion object {
        private const val TAG = "Neshter"
        private const val ACTIVE_KEY = "active-model"
    }
}
