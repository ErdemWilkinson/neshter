import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val modelDir = rootProject.file(providers.gradleProperty("neshter.modelDir").get())
val webDir = rootProject.file(providers.gradleProperty("neshter.webDir").get())
val stagedAssets = layout.buildDirectory.dir("neshter-assets")
// false: the APK leaves engine.onnx out (about 80 MB instead of 780 MB) and the app reads it from its own folder,
// Android/data/dev.neshter.mobile/files/engine.onnx (adb push). For devices with little free space and quick reinstalls.
val bundleModel = providers.gradleProperty("neshter.bundleModel").orElse("true").get().toBoolean()
// llama.cpp source tree (a pinned release) for the GGUF engine in src/main/cpp.
val llamaSrc = rootProject.file(providers.gradleProperty("neshter.llamaSrc").get())

// The APK carries the engine and the web UI as assets: model/ (engine.onnx stored uncompressed so it can be
// memory-mapped) and web/ (the phone build of the React UI). The fp32 export next to the engine stays out.
val stageAssets by tasks.registering(Sync::class) {
    inputs.property("bundleModel", bundleModel)
    from(modelDir) {
        include("meta.json", "vocab.txt", "merges.txt", "added.txt", "LICENSE")
        if (bundleModel) include("engine.onnx")
        into("model")
    }
    from(webDir) { into("web") }
    into(stagedAssets)
    doFirst {
        check(File(modelDir, "engine.onnx").exists()) { "engine.onnx yok: önce mobile/export/export_qwen2.py çalıştır ($modelDir)" }
        check(File(webDir, "index.html").exists()) { "web arayüzü yok: web/ içinde npm run build:mobile ($webDir)" }
    }
}

android {
    namespace = "dev.neshter.mobile"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.neshter.mobile"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "0.2.0"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild {
            cmake {
                // Release flags in every build: a debug llama.cpp is too slow to talk to.
                arguments += listOf("-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_SRC=${llamaSrc.invariantSeparatorsPath}")
            }
        }
    }

    ndkVersion = providers.gradleProperty("neshter.ndkVersion").get()
    providers.gradleProperty("neshter.ndkPath").orNull?.let { ndkPath = it }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
            // CMake's thousands of intermediate files go next to the other build outputs, not into the source tree.
            buildStagingDirectory = file("${providers.gradleProperty("neshter.buildRoot").get()}/cxx")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Sideloaded test builds: signed with the debug key until there is a release key.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources { noCompress += listOf("onnx") }

    // llama.cpp picks its CPU backend (libggml-cpu-<armv8.2|armv9.2|...>.so) by listing the native library folder at
    // run time, so the libraries must be extracted there on install rather than read from inside the APK.
    packaging { jniLibs { useLegacyPackaging = true } }

    sourceSets["main"].assets.srcDir(stagedAssets)

    testOptions { unitTests.isReturnDefaultValues = true }
}

kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }

tasks.named("preBuild") { dependsOn(stageAssets) }

tasks.withType<Test>().configureEach { systemProperty("neshter.modelDir", modelDir.absolutePath) }

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.32.0")
    implementation("androidx.webkit:webkit:1.14.0")
    testImplementation("junit:junit:4.13.2")
}

// The staged folder is a plain source dir: tie asset merging to the staging task and to the bundling choice.
tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }.configureEach {
    dependsOn(stageAssets)
    inputs.property("bundleModel", bundleModel)
}
