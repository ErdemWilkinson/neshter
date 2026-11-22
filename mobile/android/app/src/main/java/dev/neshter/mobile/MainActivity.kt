package dev.neshter.mobile

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.res.AssetManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.WindowInsets
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.widget.FrameLayout
import android.window.OnBackInvokedDispatcher
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import java.io.IOException

/** One WebView with Neshter's web UI, served from the APK; the model behind it runs on the phone (NativeBridge). */
class MainActivity : Activity() {
    private lateinit var web: WebView
    private lateinit var bridge: NativeBridge
    private var picked: ((Uri?) -> Unit)? = null

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WebView.setWebContentsDebuggingEnabled((applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0)
        web = WebView(this).apply { setBackgroundColor(0xFF0B0A14.toInt()) }
        // A WebView ignores its own padding, so the frame around it takes the system-bar insets.
        val frame = FrameLayout(this).apply {
            setBackgroundColor(0xFF0B0A14.toInt())
            addView(web, FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT))
        }
        setContentView(frame)
        // Edge to edge (Android 15+): keep the page clear of the status bar, the navigation bar and the keyboard.
        frame.setOnApplyWindowInsetsListener { v, insets ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val i = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime() or WindowInsets.Type.displayCutout())
                v.setPadding(i.left, i.top, i.right, i.bottom)
            }
            insets
        }

        val loader = WebViewAssetLoader.Builder().addPathHandler("/app/", WebAssets(assets)).build()
        web.webViewClient = object : WebViewClientCompat() {
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                loader.shouldInterceptRequest(request.url)

            // Nothing outside the app's own pages opens inside it.
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean =
                request.url.host != WebViewAssetLoader.DEFAULT_DOMAIN
        }
        with(web.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
        }
        bridge = NativeBridge(this, web, ::pickDocument)
        web.addJavascriptInterface(bridge, "NeshterNative")
        if (savedInstanceState == null || web.restoreState(savedInstanceState) == null) web.loadUrl(START)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            onBackInvokedDispatcher.registerOnBackInvokedCallback(OnBackInvokedDispatcher.PRIORITY_DEFAULT) { back() }
        }
    }

    /** The system file picker (no storage permission needed): a GGUF the user downloaded, wherever it is. */
    private fun pickDocument(onPicked: (Uri?) -> Unit) {
        picked?.invoke(null) // a picker left open by an earlier call counts as cancelled
        picked = onPicked
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*")
        try {
            startActivityForResult(intent, PICK_MODEL)
        } catch (_: Exception) {
            picked = null
            onPicked(null)
        }
    }

    @Deprecated("Plain Activity: the result API needs androidx.activity, which this app does not use")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode != PICK_MODEL) return super.onActivityResult(requestCode, resultCode, data)
        val done = picked
        picked = null
        done?.invoke(if (resultCode == RESULT_OK) data?.data else null)
    }

    private fun back() {
        if (web.canGoBack()) web.goBack() else finish()
    }

    @Deprecated("Android 12 and older; 13+ uses the OnBackInvokedCallback above")
    override fun onBackPressed() = back()

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        web.saveState(outState)
    }

    override fun onDestroy() {
        bridge.close()
        web.destroy()
        super.onDestroy()
    }

    /** assets/web/ as https://appassets.androidplatform.net/app/, with the MIME types module scripts need. */
    private class WebAssets(private val assets: AssetManager) : WebViewAssetLoader.PathHandler {
        override fun handle(path: String): WebResourceResponse? {
            val p = path.ifEmpty { "index.html" }
            val type = TYPES[p.substringAfterLast('.', "").lowercase()] ?: "application/octet-stream"
            return try {
                WebResourceResponse(type, if (type.startsWith("text/") || type.endsWith("json")) "utf-8" else null, assets.open("web/$p"))
            } catch (_: IOException) {
                null
            }
        }
    }

    companion object {
        private const val START = "https://${WebViewAssetLoader.DEFAULT_DOMAIN}/app/index.html"
        private const val PICK_MODEL = 41
        private val TYPES = mapOf(
            "html" to "text/html", "js" to "text/javascript", "mjs" to "text/javascript", "css" to "text/css",
            "json" to "application/json", "svg" to "image/svg+xml", "png" to "image/png", "jpg" to "image/jpeg",
            "webp" to "image/webp", "woff2" to "font/woff2", "woff" to "font/woff", "ttf" to "font/ttf",
            "wasm" to "application/wasm", "ico" to "image/x-icon", "txt" to "text/plain",
        )
    }
}
