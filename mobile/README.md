# Neşter for Android

The phone app runs Neşter's UI in a WebView, with the model running on the phone itself. It needs no server and no internet.

The app runs two kinds of model:

- **Your own GGUF.** On the Model screen, tap "GGUF modeli ekle" and pick a `.gguf` file you downloaded, for example from Hugging Face. The app copies it into its own folder, checks it and runs it with llama.cpp. Llama, Qwen, Gemma, Phi, SmolLM and other architectures llama.cpp knows all work. Small models (0.5–3 B parameters, Q4 or Q8) suit a phone.
- **The bundled model.** Qwen2.5-0.5B-Instruct, exported as a single ONNX engine. Weights are 8-bit; activations are fp32. The bundled model is optional: a build without it is the lite APK described below.

## What runs on the phone

| Feature | Bundled (ONNX) | GGUF (llama.cpp) |
| --- | --- | --- |
| Chat (multi-turn, the model's own chat template, streamed) | single turn | yes |
| 2D heatmap, hot spots, list | yes | yes |
| Prediction layer by layer (logit lens) | yes | not yet |
| Attention pattern | yes | yes |
| Measure the real effect (random controls, sequential permutation, the hundred dots) | yes | yes |
| Compare, and chat with the basket's head cuts | yes | yes |
| 3D connection graph, behaviour search, MLP neurons, test set, saving a model | no | no |

The features in the last row are hidden in the app. They need gradients or model files.

### How the GGUF engine sees heads

`android/app/src/main/cpp/neshter_llama.cpp` reads llama.cpp's own compute graph through its eval callback, with flash attention off:

- **Head power.** The callback reads `kqv_out-N`, which is each head's output before the output projection. The L2 norm of a head's slice is its power.
- **Cuts.** Cutting a head zeroes its slice in place, before the next node reads it.
- **Attention pattern.** The callback copies `kq_soft_max-N`.

When an architecture has no such per-head output (MLA, recurrent layers), the model still chats, and the head views say why they are missing.

The ARM build ships one CPU backend per instruction-set level: armv8.0, 8.2 (dotprod), 8.6 (i8mm), 9.0 and 9.2 (SVE2/SME). The app loads the best one for the phone when it starts. On arm64 it also uses KleidiAI and OpenMP.

`cpp/nl_check.cpp` tests the engine on a desktop (Linux or WSL):

```bash
cmake -S mobile/android/app/src/main/cpp -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_SRC=<llama.cpp>
cmake --build build --target nl_check -j
./build/nl_check model.gguf   # RESULT: passed
```

It checks the following:

- keeping every head changes nothing;
- attention rows are causal distributions;
- bigger cuts change the next token more;
- greedy generation repeats exactly.

Prompts in the views are limited to 128 tokens on the phone. The attention tensor is [24][14][T][T] per pass.

Real-effect measurements use 8 of the neutral prompts, 4 Turkish and 4 English. The desktop uses all 40.

## Pieces

- `export/export_qwen2.py`: Qwen2 as one ONNX graph.
  - Inputs: `input_ids`, `head_mask`, `keep`, `lens_hidden`, `lens_target`, `past_k`, `past_v`.
  - Outputs:
    - log-probs;
    - head power;
    - attention;
    - residuals;
    - logit-lens top-3;
    - present key/values (used for generation).
  - Also writes `meta.json` (name, sha256, neutral prompts, chat prefix and suffix).
  - Also writes `vocab.txt`, `merges.txt` and `added.txt` for the Kotlin tokenizer.
- `export/verify_engine.py`: compares the engine with Neşter's own measurements:
  - head power;
  - attention;
  - logit lens;
  - KL of a cut;
  - generation with the cache against generation without it.
- `export/make_golden.py`: Hugging Face token ids for the Kotlin tokenizer test.
- `android/`: Kotlin project with these parts:
  - `Qwen2Tokenizer`: byte-level BPE.
  - `Engine`: ONNX Runtime calls.
  - `NativeBridge`: `window.NeshterNative`.
  - `MainActivity`: the WebView.
- `../web/src/api/device/`: the same `Api` the desktop UI uses, answered through the bridge. The measurement statistics are in TypeScript (`stats.ts`) and covered by vitest.

## Build

The GGUF engine needs three things, all set in `android/gradle.properties` and `local.properties`:

- **llama.cpp source.** The pinned release is v1.0.0: `git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp`, set as `neshter.llamaSrc`.
- **Android NDK r29** (`29.0.14206865`), set as `neshter.ndkPath`.
- **CMake 3.31.6**, set as `cmake.dir` in `local.properties`.

Android Studio's SDK Manager installs the NDK and CMake. So does `sdkmanager --sdk_root=<dir> --package_file=<file>`, with one package per line: `ndk;29.0.14206865` and `cmake;3.31.6`.

```bash
# 1. engine (once per model; about 5 minutes, needs ~4 GB free on D:)
python mobile/export/export_qwen2.py <hf model folder> D:/neshter/mobile-build/qwen2.5-0.5b
python mobile/export/verify_engine.py <hf model folder> D:/neshter/mobile-build/qwen2.5-0.5b/engine.onnx   # "SONUÇ: geçti"

# 2. the phone build of the UI (relative URLs, into mobile/web-dist)
cd web && npm run build:mobile

# 3. APK (outputs go to D:/neshter/android-build, outside OneDrive)
cd mobile/android && ./gradlew :app:assembleDebug
```

The result is `D:/neshter/android-build/app/outputs/apk/debug/app-debug.apk`, about 780 MB with the engine inside. Paths can be changed in `android/gradle.properties`.

**Smaller APK (about 130 MB, with llama.cpp for both ABIs):** build with `./gradlew clean :app:assembleDebug -Pneshter.bundleModel=false`. This APK runs any GGUF you add in the app. To get the bundled Qwen as well, copy its engine to the phone:

```bash
adb push D:/neshter/mobile-build/qwen2.5-0.5b/engine.onnx /sdcard/Android/data/dev.neshter.mobile/files/engine.onnx
```

Use this when the phone has little free space. Installing a 780 MB APK briefly needs about twice its size. It also makes reinstalls quicker.

When switching between the two kinds of APK, include `clean`, because the asset packager keeps old files.

Tests:

- `./gradlew :app:testDebugUnitTest`: the tokenizer against the golden ids.
- `cd web && npx vitest run src/api/device`.

## Speed (emulator, x86_64, 4 threads)

| Step | Time |
| --- | --- |
| Open the model | 8–14 s |
| Heatmap | 0.4–2.5 s |
| Lens | 0.6 s |
| Attention | 0.8 s |
| Real effect (40 random controls × 8 prompts) | about 1 min 50 s |
| Compare (2 answers, key/value cache) | about 20 s for ~100 tokens each |

The table above is for the bundled Qwen2.5-0.5B (ONNX). The table below is for SmolLM2-135M-Instruct Q8_0 (GGUF, llama.cpp) on the same emulator:

| Step | Time |
| --- | --- |
| Add the file (copy 138 MB, hash, check) | about 1 s |
| Open the model | 0.25 s |
| Short chat answer (8 tokens) | first word 0.3 s, done 0.6 s |
| Heatmap | 0.13 s |
| One KL pass over 8 prompts | 1.5–2 s |

A real phone's ARM backend is faster than the emulator's x86 one. The numbers above were not measured on a real phone.

## Fidelity

These numbers are from `verify_engine.py`.

- Lens top guess matches on 25/25 layers.
- Cut KL is within 0.012 of the desktop.
- Head power (per-layer 0–1) differs by up to 0.14. The desktop's own bf16 run differs by 0.08 from fp32.
- The engine (and the phone) gives exactly the same tokens with and without the key/value cache.

Greedy answers can leave the desktop's path after a few words. This happens where two tokens are almost tied: for example, 0.05 apart in log-probability. The fp32 engine leaves the path at the same place, so 8-bit weights are not the cause.
