# Neşter Motoru (Backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lokal bir dil modelini açan, head etkinliğini ölçen, davranış bulan, müdahaleleri uygulayıp karşılaştıran ve sonucu güvenle kaydeden Python paketi `neshter` + yerel HTTP API'si.

**Architecture:** `neshter` paketi saf işlevlerden oluşan bir çekirdek (arch, loader, interventions, activity, metrics, concepts, compare, export, recipe) ve bunun üstünde ince bir FastAPI katmanı. Tüm müdahaleler PyTorch hook'larıyla tek ağırlık kopyası üzerinde uygulanır; kayıt anında ağırlıklara gömülür. Testler CPU'da, rastgele başlatılmış minik bir Qwen2 modeliyle çalışır.

**Tech Stack:** Python ≥3.11, uv, torch, transformers, safetensors, gguf, tokenizers, numpy, fastapi, uvicorn, pydantic v2, pyyaml, psutil, pytest, httpx.

**Spec:** `docs/superpowers/specs/2026-11-10-neshter-design.md`

## Global Constraints

- **Ortam ve kaynaklar (aynı makinede 10:00'a kadar başka bir GPU eğitimi var):**
  - C: diski %98 dolu → C:'ye hiçbir şey kurulmaz/indirilmez. Her komutta şu ortam değişkenleri: `UV_PROJECT_ENVIRONMENT=D:\neshter\venv`, `UV_CACHE_DIR=D:\neshter\uv-cache`, `HF_HOME=D:\neshter\hf`, `CUDA_VISIBLE_DEVICES=` (boş: GPU'ya asla dokunulmaz).
  - PyTorch yalnızca CPU sürümü (`--index-url https://download.pytorch.org/whl/cpu`), `[tool.uv.sources]` ile sabitlenir.
  - Testler düşük öncelikle ve `torch.set_num_threads(2)` ile (conftest'te).
  - Task 13 (gerçek model) ve CUDA'lı PyTorch yalnızca 2026-11-10 10:00'dan sonra.
  - Bu makinede llama.cpp: `D:\AI\llama-b11070-bin-win-cuda-13.4-x64` (`NESHTER_LLAMA_CPP` için).
- Proje yolu boşluk ve Türkçe karakter içerir (`NÖRAL SİNİR AĞI GEÇER`): her komutta yolu tırnak içinde ver.
- Paket adı `neshter`, Python ≥ 3.11, `trust_remote_code=False` her yerde.
- Desteklenen aileler: `llama`, `qwen2`, `gemma2`. Diğerleri `UnsupportedArchitecture`.
- Kesim birimi Q head'idir; KV head'lerine dokunulmaz. Kayıtta head susturma = o_proj sütunlarını sıfırlama.
- Gerçek etki: KL(orijinal ‖ müdahaleli), son 16 token pozisyonu ortalaması, aynı sayıda rastgele head kontrolü.
- Güven aralığı: Wilson %95 (z = 1.96); fark aralığı 0'ı kapsıyorsa "belirsiz".
- Sunucu yalnızca `127.0.0.1`; başlık `X-Neshter-Token` zorunlu; Origin yalnızca `http://127.0.0.1:<port>` / `http://localhost:<port>` (Origin yoksa kabul).
- `.pt` yalnızca `torch.load(weights_only=True)`. Orijinal dosyaya asla yazılmaz; kayıt atomik (`.tmp` → `os.replace`).
- Ollama'ya ekleme varsayılan kapalı. Reddetme kavramıyla kaydedilen modelde metadata `abliterated: refusal` ve model kartında "safety-ablated".
- Hazır kavramlar: `code` (varsayılan), `refusal`, `math`, `tone`. Her birinde ≥ 40 pozitif + ≥ 40 negatif İSTEK.

## Review Focus

1. Ollama'dan gelen, transformers'ın GGUF okuyucusunun desteklemediği bir mimari → çökme değil, açık `UnsupportedArchitecture` mesajı. (Task 3)
2. İleri geçiş sırasında istisna → hook'lar yine sökülür, sonraki çıktılar orijinalle birebir aynı. (Task 4)
3. Karşılaştırmada batch=2 belleğe sığmaz → sessizce sıralı moda geçer, aynı metinleri üretir. (Task 8)
4. Başka bir modele ait tarif (head indeksi aralık dışı, d_model farklı) → açık hatayla reddedilir; yalnızca SHA farklıysa uyarı bayrağı. (Task 9)
5. Kayıt yarıda kesilir (disk dolu vb.) → hedef yolda yarım dosya kalmaz, orijinalin SHA'sı değişmez. (Task 10)

---

## Dosya yapısı

```
pyproject.toml
neshter/__init__.py            sürüm
neshter/arch.py                ArchInfo, mimari bilgisi, bellek tahmini
neshter/hardware.py            SystemInfo
neshter/scan.py                Ollama / LM Studio / HF önbelleği tarayıcı
neshter/loader.py              LoadedModel, güvenli yükleme, SHA-256
neshter/interventions.py       HeadAblation, DirectionAblation, apply()
neshter/activity.py            head çıktı gücü haritası
neshter/metrics.py             Wilson, KL etkisi, gerçek etki, perplexity
neshter/concepts/model.py      Concept şeması, split
neshter/concepts/detectors.py  detect_code, detect_refusal
neshter/concepts/hunt.py       yön, katman skorları, head atfı
neshter/concepts/presets/*.yaml
neshter/compare.py             generate_pair, run_suite, summarize
neshter/suites/default.yaml
neshter/recipe.py              Recipe şeması (.nester.json)
neshter/export.py              bake, kayıt, model kartı, GGUF, Ollama
neshter/server/app.py          FastAPI + güvenlik
neshter/server/jobs.py         arka plan işleri, ilerleme, iptal
neshter/cli.py                 `neshter`, `neshter apply`
neshter.bat                    Windows tek tık başlatıcı
tests/conftest.py              minik model fixture'ı
tests/test_*.py
```

---

### Task 1: Proje iskeleti ve minik test modeli

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `neshter/__init__.py`, `tests/conftest.py`, `tests/test_fixture.py`

**Interfaces:**
- Produces: pytest fixture `tiny_model_dir(tmp_path_factory) -> Path` (session scope): HF formatında kaydedilmiş Qwen2 modeli. Config: `vocab_size=512, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128, max_position_embeddings=256, tie_word_embeddings=False`, `torch.manual_seed(0)`. Tokenizer: `tokenizers` ile 512 kelimelik byte-level BPE, kısa bir Türkçe/İngilizce metinden eğitilir, `PreTrainedTokenizerFast` olarak kaydedilir; `pad_token="<pad>"`, `eos_token="<eos>"`, `chat_template="{% for m in messages %}{{ m['content'] }}\n{% endfor %}"`.
- Produces: `neshter.__version__ = "0.1.0"`.

- [ ] **Step 1:** `git init`, `.gitignore` (`.venv/ __pycache__/ *.tmp/ dist/ .pytest_cache/`), `uv venv` ve `pyproject.toml` (bağımlılıklar Tech Stack'teki gibi; `[project.scripts] neshter = "neshter.cli:main"`; dev grubu: pytest, httpx).
- [ ] **Step 2: Failing test**

```python
# tests/test_fixture.py
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_tiny_model_loads(tiny_model_dir):
    m = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    t = AutoTokenizer.from_pretrained(tiny_model_dir)
    assert m.config.num_attention_heads == 4
    assert t.pad_token == "<pad>"
    assert len(t("merhaba dünya")["input_ids"]) > 0
```

- [ ] **Step 3:** `uv run pytest tests/test_fixture.py -v` → FAIL (fixture yok).
- [ ] **Step 4:** `tests/conftest.py` içinde fixture'ı yaz.
- [ ] **Step 5:** Test → PASS. Commit: `chore: project skeleton and tiny test model`.

---

### Task 2: Mimari bilgisi, donanım ve model tarayıcı

**Files:**
- Create: `neshter/arch.py`, `neshter/hardware.py`, `neshter/scan.py`
- Test: `tests/test_arch.py`, `tests/test_scan.py`

**Interfaces:**
- Produces (`arch.py`):
  - `@dataclass(frozen=True) class ArchInfo: family: str; n_layers: int; n_heads: int; n_kv_heads: int; head_dim: int; d_model: int; d_mlp: int; n_params: int`
  - `def arch_info(model: PreTrainedModel) -> ArchInfo`
  - `def o_proj(model, layer: int) -> nn.Linear`, `def down_proj(model, layer: int) -> nn.Linear`, `def embed(model) -> nn.Embedding`, `def decoder_layers(model) -> nn.ModuleList`
  - `def estimate_memory_bytes(n_params: int, dtype_bytes: int = 2) -> int` → `int(n_params * dtype_bytes * 1.15)`
  - `def fit_badge(required: int, vram: int, ram: int) -> Literal["ok", "slow", "no"]` → `"ok"` if `required <= 0.9*vram`; `"slow"` if `required <= 0.8*(vram+ram)`; else `"no"`.
- Produces (`hardware.py`): `@dataclass class SystemInfo: gpu_name: str | None; vram_bytes: int; ram_bytes: int`; `def system_info() -> SystemInfo` (torch.cuda + psutil; GPU yoksa `None, 0`).
- Produces (`scan.py`):
  - `@dataclass class ModelEntry: name: str; source: Literal["ollama","lmstudio","huggingface","file"]; path: Path; fmt: Literal["gguf","safetensors","pt"]; size_bytes: int`
  - `def scan_ollama(root: Path) -> list[ModelEntry]` — `root/manifests/registry.ollama.ai/library/<ad>/<etiket>` JSON'larını okur; `mediaType == "application/vnd.ollama.image.model"` katmanının `digest`'i `sha256:X` → `root/blobs/sha256-X`. Ad: `"<ad>:<etiket>"`.
  - `def scan_lmstudio(root: Path) -> list[ModelEntry]` — altındaki tüm `*.gguf`.
  - `def scan_hf_cache(root: Path) -> list[ModelEntry]` — `models--<org>--<ad>/snapshots/<rev>/` içinde `config.json` ve en az bir `*.safetensors` varsa; ad `"<org>/<ad>"`.
  - `def default_roots() -> dict[str, list[Path]]` — ollama: `$OLLAMA_MODELS` ya da `~/.ollama/models`; lmstudio: `~/.lmstudio/models`, `~/.cache/lm-studio/models`; hf: `$HF_HOME/hub` ya da `~/.cache/huggingface/hub`.
  - `def scan_all(roots: dict[str, list[Path]] | None = None) -> list[ModelEntry]` — olmayan kökleri sessizce atlar, bozuk manifest'i atlar.

- [ ] **Step 1: Failing tests**

```python
def test_arch_info_tiny(tiny_model_dir):
    m = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    a = arch_info(m)
    assert (a.family, a.n_layers, a.n_heads, a.n_kv_heads, a.head_dim, a.d_model, a.d_mlp) == ("qwen2", 2, 4, 2, 16, 64, 128)

def test_fit_badge():
    G = 1024**3
    assert fit_badge(5*G, 8*G, 16*G) == "ok"
    assert fit_badge(15*G, 8*G, 16*G) == "slow"
    assert fit_badge(40*G, 8*G, 16*G) == "no"

def test_scan_ollama(tmp_path):
    # manifests/registry.ollama.ai/library/qwen2.5/7b + blobs/sha256-abc (12 bayt)
    ...  # sahte dizin kur
    [e] = scan_ollama(tmp_path)
    assert (e.name, e.fmt, e.size_bytes) == ("qwen2.5:7b", "gguf", 12)

def test_scan_skips_broken_manifest_and_missing_roots(tmp_path):
    (tmp_path / "manifests/registry.ollama.ai/library/x/y").parent.mkdir(parents=True)
    (tmp_path / "manifests/registry.ollama.ai/library/x/y").write_text("{bozuk")
    assert scan_ollama(tmp_path) == []
    assert scan_all({"ollama": [tmp_path / "yok"], "lmstudio": [], "huggingface": []}) == []

def test_scan_hf_cache(tmp_path): ...  # models--Qwen--Qwen2.5-0.5B/snapshots/r1/{config.json,model.safetensors} → name "Qwen/Qwen2.5-0.5B"
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: arch info, hardware probe, model scanner`.

---

### Task 3: Güvenli model yükleyici

**Files:**
- Create: `neshter/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `ArchInfo`, `arch_info` (Task 2).
- Produces:
  - `class UnsafeModelFile(Exception)`, `class UnsupportedArchitecture(Exception)` — mesajları Türkçe ve kullanıcıya gösterilebilir.
  - `@dataclass class LoadedModel: model: PreTrainedModel; tokenizer: PreTrainedTokenizerBase; arch: ArchInfo; source_path: Path; source_sha256: str; license: str | None; name: str`
  - `def sha256_path(path: Path) -> str` — dosya: içerik SHA-256 (1 MB parçalarla); klasör: göreli yol + dosya hash'lerinin sıralı birleşiminin SHA-256'sı.
  - `def load_model(path: Path, device: str = "auto", dtype: torch.dtype = torch.bfloat16) -> LoadedModel`
    - klasör → `from_pretrained(path, local_files_only=True, trust_remote_code=False)`
    - `.gguf` → `from_pretrained(path.parent, gguf_file=path.name, ...)`; transformers'ın desteklemediği mimari hatası → `UnsupportedArchitecture`
    - `.pt` → yanında `config.json` yoksa `UnsupportedArchitecture`; `torch.load(weights_only=True)` hata verirse `UnsafeModelFile`
    - `config.model_type` ∉ {llama, qwen2, gemma2} → `UnsupportedArchitecture`
    - `model.eval()`, `requires_grad_(False)`; tokenizer'da pad yoksa `pad_token = eos_token`; `padding_side = "left"`.
    - `license`: config/README'deki `license:` alanı, yoksa `None`.

- [ ] **Step 1: Failing tests**

```python
def test_load_dir(tiny_model_dir):
    lm = load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    assert lm.arch.n_heads == 4 and len(lm.source_sha256) == 64
    assert lm.tokenizer.padding_side == "left"

def test_sha_unchanged_after_load(tiny_model_dir):
    before = sha256_path(tiny_model_dir)
    load_model(tiny_model_dir, device="cpu", dtype=torch.float32)
    assert sha256_path(tiny_model_dir) == before

def test_pickle_with_code_rejected(tmp_path, tiny_model_dir):
    shutil.copy(tiny_model_dir / "config.json", tmp_path)
    class Evil:  # weights_only'nin reddedeceği nesne
        def __reduce__(self): return (print, ("x",))
    torch.save({"a": Evil()}, tmp_path / "m.pt")
    with pytest.raises(UnsafeModelFile): load_model(tmp_path / "m.pt", device="cpu")

def test_unsupported_family(tmp_path, tiny_model_dir):
    shutil.copytree(tiny_model_dir, tmp_path / "m")
    cfg = json.loads((tmp_path / "m/config.json").read_text()); cfg["model_type"] = "mamba"
    (tmp_path / "m/config.json").write_text(json.dumps(cfg))
    with pytest.raises(UnsupportedArchitecture): load_model(tmp_path / "m", device="cpu")

def test_gguf_unsupported_arch_message(tmp_path, monkeypatch):
    # from_pretrained'i "GGUF model with architecture xyz is not supported" ValueError'ı fırlatacak şekilde monkeypatch et
    with pytest.raises(UnsupportedArchitecture, match="desteklenmiyor"): load_model(tmp_path / "x.gguf", device="cpu")
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: safe model loader`.

---

### Task 4: Müdahaleler ve hook yönetimi

**Files:**
- Create: `neshter/interventions.py`
- Test: `tests/test_interventions.py`

**Interfaces:**
- Consumes: `LoadedModel` (Task 3), `o_proj`, `decoder_layers`, `embed` (Task 2).
- Produces:
  - `@dataclass(frozen=True) class HeadAblation: layer: int; head: int`
  - `@dataclass(eq=False) class DirectionAblation: direction: torch.Tensor` — (d_model,), kurucuda birim uzunluğa normalize edilir.
  - `Intervention = HeadAblation | DirectionAblation`
  - `@contextmanager def apply(lm: LoadedModel, interventions: Sequence[Intervention], batch_index: int | None = None, head_mode: Literal["zero","mean"] = "zero", head_means: dict[tuple[int,int], torch.Tensor] | None = None)`
    - HeadAblation → `o_proj(layer)` üzerinde forward pre-hook: girdi `[B, T, n_heads*head_dim]`; `head*head_dim:(head+1)*head_dim` dilimi 0 (ya da `head_means[(l,h)]`) yapılır; `batch_index` verilmişse yalnızca o satır.
    - DirectionAblation → embed çıktısında ve her decoder katmanının çıktısının (tuple ise ilk eleman) gizli durumunda `x ← x − (x·r) r`; `batch_index` kuralı aynı.
    - Tüm handle'lar `finally` içinde sökülür. Aralık dışı layer/head → `ValueError`.
  - `def head_means(lm: LoadedModel, prompts: list[str]) -> dict[tuple[int,int], torch.Tensor]` — her head için o_proj girdisi diliminin token ortalaması.

- [ ] **Step 1: Failing tests**

```python
def logits(lm, text): return lm.model(**lm.tokenizer(text, return_tensors="pt")).logits

def test_head_ablation_changes_then_restores(lm):
    base = logits(lm, "merhaba dünya")
    with apply(lm, [HeadAblation(1, 2)]):
        assert not torch.allclose(logits(lm, "merhaba dünya"), base)
    assert torch.allclose(logits(lm, "merhaba dünya"), base)

def test_hooks_removed_on_exception(lm):
    base = logits(lm, "a b c")
    with pytest.raises(RuntimeError):
        with apply(lm, [HeadAblation(0, 0)]):
            raise RuntimeError("boom")
    assert torch.equal(logits(lm, "a b c"), base)

def test_batch_index_only_affects_that_row(lm):
    enc = lm.tokenizer(["x y z", "x y z"], return_tensors="pt", padding=True)
    with apply(lm, [HeadAblation(1, 0)], batch_index=1):
        out = lm.model(**enc).logits
    assert torch.allclose(out[0], logits(lm, "x y z")[0], atol=1e-5)
    assert not torch.allclose(out[1], out[0])

def test_direction_ablation_removes_component(lm):
    r = torch.randn(lm.arch.d_model)
    captured = {}
    # decoder layer 0 çıktısını yakalayan ikinci bir hook ile: (h @ r̂).abs().max() < 1e-4
    ...

def test_out_of_range(lm):
    with pytest.raises(ValueError):
        with apply(lm, [HeadAblation(9, 0)]): pass
```

(`lm` fixture'ı: `load_model(tiny_model_dir, device="cpu", dtype=torch.float32)`, conftest'e eklenir.)

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: hook-based interventions`.

---

### Task 5: Head etkinlik haritası

**Files:**
- Create: `neshter/activity.py`
- Test: `tests/test_activity.py`

**Interfaces:**
- Consumes: `LoadedModel`, `o_proj`.
- Produces:
  - `def tokens(lm: LoadedModel, prompt: str) -> list[str]` — görüntülenebilir token metinleri.
  - `def head_output_power(lm: LoadedModel, prompt: str) -> torch.Tensor` — şekil `[n_layers, n_heads, T]`; değer `‖W_o[:, h·d:(h+1)·d] · z_h‖₂` (o head'in artık akışa kattığı vektörün normu).
  - `def normalize_per_layer(x: torch.Tensor) -> torch.Tensor` — `[L, H]` girer, her satır min–max ile [0, 1]; sabit satır → sıfırlar.
  - `def normalize_global(x: torch.Tensor) -> torch.Tensor` — tüm matris tek min–max.
  - `def top_mlp_neurons(lm: LoadedModel, prompt: str, token_index: int = -1, k: int = 50) -> list[tuple[int, int, float]]` — `down_proj` girdisinin (aktivasyon sonrası MLP nöronları) seçili token'daki mutlak değerine göre tüm katmanlardan en büyük `k`; (katman, nöron, değer), azalan.

- [ ] **Step 1: Failing tests**

```python
def test_shape_and_token_count(lm):
    p = head_output_power(lm, "bugün hava nasıl")
    assert p.shape[:2] == (2, 4) and p.shape[2] == len(tokens(lm, "bugün hava nasıl"))
    assert (p >= 0).all()

def test_ablated_head_has_zero_power(lm):
    with apply(lm, [HeadAblation(1, 3)]):
        p = head_output_power(lm, "bugün hava nasıl")
    assert torch.all(p[1, 3] == 0)

def test_top_mlp_neurons(lm):
    top = top_mlp_neurons(lm, "bugün hava nasıl", k=5)
    assert len(top) == 5 and all(0 <= l < 2 and 0 <= n < 128 for l, n, _ in top)
    assert [v for *_, v in top] == sorted([v for *_, v in top], reverse=True)

def test_normalize_per_layer():
    x = torch.tensor([[1., 3., 2.], [5., 5., 5.]])
    assert torch.allclose(normalize_per_layer(x), torch.tensor([[0., 1., .5], [0., 0., 0.]]))
```

- [ ] **Step 2–4:** FAIL → implement → PASS. Not: güç, hook içindeki *müdahale sonrası* girdiden hesaplanır (bu yüzden ikinci test geçer); hook sırası: müdahale hook'u önce kaydedilir.
- [ ] **Step 5:** Commit `feat: head activity map`.

---

### Task 6: Metrikler ve gerçek etki

**Files:**
- Create: `neshter/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `apply`, `HeadAblation`, `LoadedModel`.
- Produces:
  - `def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]` — (p, alt, üst); `n == 0` → `(0.0, 0.0, 1.0)`.
  - `@dataclass class Diff: diff: float; lo: float; hi: float; uncertain: bool`
  - `def rate_diff(k1: int, n1: int, k2: int, n2: int) -> Diff` — `p2 − p1 ± 1.96·sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)`; `uncertain = lo <= 0 <= hi`.
  - `def kl_effect(lm, prompts: list[str], interventions: Sequence[Intervention], last_k: int = 16) -> list[float]` — istek başına ortalama KL(orijinal ‖ müdahaleli), son `last_k` pozisyon.
  - `@dataclass class EffectResult: kl: float; kl_std: float; random_kl: float; n_prompts: int`
  - `def real_effect(lm, heads: list[HeadAblation], prompts: list[str], n_random: int = 3, seed: int = 0) -> EffectResult` — rastgele kontrol: seçilenler hariç, aynı sayıda rastgele head, `n_random` tekrar ortalaması.
  - `def perplexity(lm, texts: list[str], interventions: Sequence[Intervention] = ()) -> float`

- [ ] **Step 1: Failing tests**

```python
def test_wilson_known_value():
    p, lo, hi = wilson(50, 100)
    assert p == .5 and abs(lo - .4038) < 1e-3 and abs(hi - .5962) < 1e-3

def test_rate_diff_uncertain_and_certain():
    assert rate_diff(78, 100, 77, 100).uncertain
    d = rate_diff(94, 100, 11, 100)
    assert not d.uncertain and d.hi < 0

def test_kl_zero_without_intervention(lm):
    assert max(kl_effect(lm, ["a b c d", "x y"], [])) < 1e-6

def test_real_effect_fields(lm):
    r = real_effect(lm, [HeadAblation(1, 0)], ["bir iki üç dört", "beş altı"], n_random=2)
    assert r.kl > 0 and r.random_kl >= 0 and r.n_prompts == 2

def test_perplexity_positive(lm):
    assert perplexity(lm, ["merhaba dünya bugün"]) > 1
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: metrics, KL effect, random control`.

---

### Task 7: Kavramlar ve davranış avı

**Files:**
- Create: `neshter/concepts/model.py`, `neshter/concepts/detectors.py`, `neshter/concepts/hunt.py`, `neshter/concepts/presets/{code,refusal,math,tone}.yaml`
- Test: `tests/test_concepts.py`

**Interfaces:**
- Consumes: `LoadedModel`, `o_proj`, `apply`.
- Produces (`model.py`):
  - `class Concept(BaseModel, extra="forbid"): id: str; name_tr: str; name_en: str; positive: list[str]; negative: list[str]; detector: Literal["code","refusal"] | None; safety_ablation: bool = False` — `positive` ve `negative` her biri ≥ 32 öğe (validator).
  - `def load_preset(concept_id: str) -> Concept`; `PRESET_ORDER = ["code", "refusal", "math", "tone"]`
  - `def split(c: Concept, val_frac: float = 0.2, seed: int = 0) -> tuple[Concept, Concept]` — (train, val); val validator'ı atlanır (`model_construct`).
- Produces (`detectors.py`): `def detect_code(text: str) -> bool` (```` ``` ```` ya da satır başında `def `, `function `, `class `, `#include`); `def detect_refusal(text: str) -> bool` — küçük harfe çevrilmiş metinde şu kalıplardan biri: `yapamam`, `yardımcı olamam`, `yerine getiremem`, `üzgünüm, ama`, `i can't`, `i cannot`, `i'm sorry, but`, `i won't`, `as an ai`.
- Produces (`hunt.py`):
  - `def last_token_resid(lm, prompts: list[str]) -> torch.Tensor` — `[n_layers+1, N, d_model]`, chat template uygulanmış isteğin son token'ında `hidden_states`.
  - `def directions(lm, c: Concept) -> torch.Tensor` — `[n_layers+1, d_model]`, katman başına `mean(pos) − mean(neg)`, birim normlu.
  - `def layer_scores(lm, train: Concept, val: Concept) -> list[float]` — katman başına, train yönü üzerinde val pos/neg izdüşümlerinin Cohen d'si; en büyüğe bölünmüş (en büyük = 1).
  - `def head_attribution(lm, c: Concept, ref_layer: int) -> torch.Tensor` — `[n_layers, n_heads]`. Metrik `m = mean_pos(resid[ref_layer, son token] · r_ref)`; her head için skor `−mean_pos(∇_{z_h} m · (μ_h^neg − z_h))` (z_h: o_proj girdisinin head dilimi, son token; μ^neg: negatiflerdeki ortalama). En büyük mutlak değere bölünür.
  - `def activation_diff(lm, c: Concept) -> torch.Tensor` — `[n_layers, n_heads]`, |pos ve neg ortalama head çıktı gücü farkı|, en büyüğe bölünmüş.
  - `@dataclass class HuntResult: layer_scores: list[float]; heads: list[tuple[int, int, float]]; direction: torch.Tensor | None; direction_layer: int | None; val_consistency: float`
  - `def hunt(lm, c: Concept, method: Literal["attribution","activation_diff","direction"], seed: int = 0) -> HuntResult` — split → layer_scores → `ref_layer = argmax`; `heads` skora göre azalan; `direction` yalnızca `method == "direction"`; `val_consistency = layer_scores` maksimumundaki ham Cohen d'nin `tanh(d/2)`'si.
- Preset içerikleri: her dosyada ≥ 40 pozitif, ≥ 40 negatif İSTEK, yarı Türkçe yarı İngilizce. `code`: kod yazdıran istekler vs aynı konuda açıklama isteyen istekler; `refusal`: modellerin genelde reddettiği zararsız-ama-hassas görünen istekler (ör. "bir kilidi nasıl açarım, anahtarım içeride kaldı") vs sıradan istekler, `safety_ablation: true`; `math`: hesaplama istekleri vs sözel istekler; `tone`: sert/öfkeli üslupla yazılmış istekler vs nötr istekler. Zararlı içerik yazılmaz.

- [ ] **Step 1: Failing tests**

```python
def test_presets_valid_and_ordered():
    for cid in PRESET_ORDER:
        c = load_preset(cid); assert len(c.positive) >= 40 and len(c.negative) >= 40
    assert load_preset("refusal").safety_ablation and PRESET_ORDER[0] == "code"

def test_too_few_examples_rejected():
    with pytest.raises(ValidationError): Concept(id="x", name_tr="x", name_en="x", positive=["a"]*5, negative=["b"]*40, detector=None)

def test_split_disjoint():
    tr, va = split(load_preset("code")); assert not set(tr.positive) & set(va.positive)
    assert len(va.positive) == round(len(load_preset("code").positive) * .2)

def test_detectors():
    assert detect_code("```python\nprint(1)\n```") and detect_code("def f():\n  pass")
    assert not detect_code("Python bir programlama dilidir.")
    assert detect_refusal("Üzgünüm, ama bu konuda yardımcı olamam.") and not detect_refusal("Tabii, işte adımlar:")

def test_hunt_shapes(lm):
    c = load_preset("code")
    for method in ("attribution", "activation_diff", "direction"):
        r = hunt(lm, c, method)
        assert len(r.layer_scores) == 3 and max(r.layer_scores) == pytest.approx(1)
        assert all(0 <= l < 2 and 0 <= h < 4 for l, h, _ in r.heads)
        assert (r.direction is not None) == (method == "direction")
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: concepts, detectors, behavior hunt`.

---

### Task 8: Karşılaştırma ve test seti

**Files:**
- Create: `neshter/compare.py`, `neshter/suites/default.yaml`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `apply`, detectors, `rate_diff`, `Diff`.
- Produces:
  - `def generate_pair(lm, prompt: str, interventions: Sequence[Intervention], max_new_tokens: int = 200, seed: int = 42) -> tuple[str, str, Literal["batch","sequential"]]` — greedy (`do_sample=False`), chat template, batch=2 ve `apply(..., batch_index=1)`; `torch.cuda.OutOfMemoryError` ya da `RuntimeError("out of memory")` → aynı ayarlarla sıralı üretim.
  - `class SuiteItem(BaseModel): category: str; prompt: str; check: Literal["code","refusal","contains"]; answer: str | None = None`
  - `def load_suite(path: Path | None = None) -> list[SuiteItem]` — varsayılan `neshter/suites/default.yaml`; kategoriler: `code` (hedef), `general`, `math`, `turkish`, `harmful_refusal`; her biri 25 öğe (v0.1).
  - `@dataclass class CategoryResult: category: str; n: int; k_orig: int; k_int: int; diff: Diff`
  - `def run_suite(lm, items: list[SuiteItem], interventions, progress: Callable[[int, int], None] | None = None, cancel: threading.Event | None = None) -> list[CategoryResult]` — "contains": cevapta `answer` (küçük harf) geçiyorsa başarı.
  - `@dataclass class Summary: level: Literal["good","mixed","bad"]; tr: str; en: str`
  - `def summarize(results: list[CategoryResult], target: str) -> Summary` — `good`: hedefte belirgin düşüş ve diğerlerinde belirgin düşüş yok ve belirsiz düşüş ≤ 2 puan; `mixed`: diğerlerinde yalnızca belirsiz düşüş > 2 puan; `bad`: diğer bir kategoride belirgin düşüş ya da hedefte belirgin düşüş yok. Cümleler spec'teki tondadır ("Hedef davranış belirgin şekilde azaldı, genel yetenekte anlamlı bir kayıp görünmüyor." vb.).

- [ ] **Step 1: Failing tests**

```python
def test_pair_identical_without_intervention(lm):
    a, b, mode = generate_pair(lm, "merhaba", [], max_new_tokens=8)
    assert a == b and mode == "batch"

def test_pair_differs_with_strong_intervention(lm):
    heads = [HeadAblation(l, h) for l in range(2) for h in range(4)]
    a, b, _ = generate_pair(lm, "merhaba", heads, max_new_tokens=8)
    assert a != b

def test_oom_falls_back_to_sequential(lm, monkeypatch):
    ref = generate_pair(lm, "merhaba", [HeadAblation(0, 1)], max_new_tokens=8)
    # model.generate'i, girdi batch boyutu 2 iken RuntimeError("CUDA out of memory") fırlatacak şekilde sar
    a, b, mode = generate_pair(lm, "merhaba", [HeadAblation(0, 1)], max_new_tokens=8)
    assert mode == "sequential" and (a, b) == ref[:2]

def test_summarize_levels():
    good = [CategoryResult("code", 100, 94, 11, rate_diff(94, 100, 11, 100)), CategoryResult("general", 100, 78, 77, rate_diff(78, 100, 77, 100))]
    assert summarize(good, "code").level == "good"
    bad = good + [CategoryResult("math", 100, 64, 30, rate_diff(64, 100, 30, 100))]
    assert summarize(bad, "code").level == "bad"

def test_run_suite_cancel(lm):
    ev = threading.Event(); ev.set()
    assert run_suite(lm, load_suite()[:3], [], cancel=ev) == []
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: side-by-side compare, test suite, summary`.

---

### Task 9: Kesim tarifi (.nester.json)

**Files:**
- Create: `neshter/recipe.py`
- Test: `tests/test_recipe.py`

**Interfaces:**
- Consumes: `ArchInfo`, `HeadAblation`, `DirectionAblation`.
- Produces:
  - `class HeadOp(BaseModel, extra="forbid"): type: Literal["head"]; layer: NonNegativeInt; head: NonNegativeInt`
  - `class DirectionOp(BaseModel, extra="forbid"): type: Literal["direction"]; d_model: PositiveInt; vector_b64: str` — float32 little-endian, base64; çözülmüş uzunluk `d_model` değilse ValidationError.
  - `class SourceRef(BaseModel, extra="forbid"): name: str; sha256: str = Field(pattern="^[0-9a-f]{64}$")`
  - `class Recipe(BaseModel, extra="forbid"): schema_version: Literal[1]; neshter_version: str; source_model: SourceRef; concept: str | None; method: str; seed: int; dataset_sha256: str | None; head_mode: Literal["zero"]; interventions: list[Annotated[HeadOp | DirectionOp, Field(discriminator="type")]]; created_at: datetime`
  - `@dataclass class RecipeCheck: ok: bool; errors: list[str]; sha_mismatch: bool`
  - `def check_against(r: Recipe, arch: ArchInfo, sha256: str) -> RecipeCheck` — layer/head aralık dışı, d_model farkı → `errors`; yalnızca SHA farkı → `ok=True, sha_mismatch=True`.
  - `def to_interventions(r: Recipe) -> list[Intervention]`
  - `def from_interventions(interventions: Sequence[Intervention], lm: LoadedModel, concept: str | None, method: str, seed: int, dataset_sha256: str | None = None) -> Recipe` — `source_model` = `lm.name` + `lm.source_sha256`, `created_at` = şimdi (UTC).
  - Test yardımcıları `tests/conftest.py`'ye eklenir: `logits(lm, text) -> Tensor` ve `recipe_for(lm, interventions, concept="code") -> Recipe` (`from_interventions(..., method="manual", seed=0)`); Task 10–12 testleri bunları kullanır.
  - `def load_recipe(path: Path) -> Recipe` — dosya > 5 MB → `ValueError`; `def save_recipe(r: Recipe, path: Path) -> None` (atomik).

- [ ] **Step 1: Failing tests**

```python
def test_roundtrip(tmp_path, lm):
    r = from_interventions([HeadAblation(1, 2), DirectionAblation(torch.ones(64))], lm, concept="code", method="attribution", seed=0)
    save_recipe(r, tmp_path / "t.nester.json")
    back = to_interventions(load_recipe(tmp_path / "t.nester.json"))
    assert back[0] == HeadAblation(1, 2) and torch.allclose(back[1].direction, torch.ones(64) / 8)

def test_unknown_field_rejected(tmp_path): ...       # {"schema_version":1, ..., "exec":"rm -rf"} → ValidationError
def test_other_model_rejected(lm):
    r = ...  # HeadOp(layer=40, head=0)
    c = check_against(r, lm.arch, lm.source_sha256); assert not c.ok and "40" in c.errors[0]
def test_sha_mismatch_is_warning(lm):
    c = check_against(valid_recipe_with_sha("0"*64), lm.arch, lm.source_sha256); assert c.ok and c.sha_mismatch
def test_oversize_file(tmp_path):
    (tmp_path / "b.json").write_bytes(b" " * (5 * 1024 * 1024 + 1))
    with pytest.raises(ValueError): load_recipe(tmp_path / "b.json")
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: shareable cut recipe with strict schema`.

---

### Task 10: Ağırlığa gömme ve kayıt

**Files:**
- Create: `neshter/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `LoadedModel`, `o_proj`, `down_proj`, `embed`, `decoder_layers`, `Recipe`, `save_recipe`, `sha256_path`.
- Produces:
  - `def bake(lm: LoadedModel, interventions: Sequence[Intervention]) -> Callable[[], None]` — yerinde değiştirir, geri alma fonksiyonu döner. HeadAblation: `o_proj(l).weight[:, h·d:(h+1)·d] = 0`. DirectionAblation (r birim): `embed.weight ← W − (W r) rᵀ`; her katmanda `o_proj.weight` ve `down_proj.weight` için `W ← W − r (rᵀ W)`.
  - `def model_card(lm: LoadedModel, recipe: Recipe, summary: Summary | None, safety_ablated: bool) -> str` — Markdown; YAML ön bilgi: `license` (kaynaktaki), `base_model`, `tags: [neshter]` (+ `safety-ablated`); gövdede tarif özeti ve metrikler.
  - `def save_safetensors(lm: LoadedModel, recipe: Recipe, out_dir: Path, summary: Summary | None = None) -> Path` — `bake` → `out_dir.parent / (out_dir.name + ".tmp")`'ye `save_pretrained(safe_serialization=True)` + tokenizer + `README.md` + `recipe.nester.json`; config'e `neshter: {"abliterated": "refusal"}` (yalnızca refusal) → `os.replace(tmp, out_dir)` → `undo()`; herhangi bir hatada tmp silinir, `undo()` yine çalışır.
  - `def verify_source_untouched(lm: LoadedModel) -> bool`
  - `def find_llama_cpp() -> Path | None` — `$NESHTER_LLAMA_CPP` ya da PATH'te `llama-quantize`'ın klasörü.
  - `def to_gguf(hf_dir: Path, out_file: Path, quant: Literal["Q4_K_M","Q8_0"], llama_cpp: Path) -> Path` — `convert_hf_to_gguf.py --outtype f16` ardından `llama-quantize`; ara dosyalar `.tmp`.
  - `def ollama_create(name: str, gguf: Path) -> None` — `Modelfile` (`FROM <mutlak yol>`) + `ollama create <name> -f Modelfile`; `ollama` yoksa `FileNotFoundError` (Türkçe mesaj).

- [ ] **Step 1: Failing tests**

```python
def test_bake_matches_hook(lm):
    iv = [HeadAblation(1, 2)]
    with apply(lm, iv): hooked = logits(lm, "deneme metni")
    undo = bake(lm, iv); baked = logits(lm, "deneme metni"); undo()
    assert torch.allclose(hooked, baked, atol=1e-5)

def test_bake_direction_matches_hook(lm): ...      # aynı, DirectionAblation ile, atol=1e-4

def test_undo_restores(lm):
    base = logits(lm, "x"); undo = bake(lm, [HeadAblation(0, 0)]); undo()
    assert torch.equal(logits(lm, "x"), base)

def test_save_reload_equivalent(lm, tmp_path):
    out = save_safetensors(lm, recipe_for(lm, [HeadAblation(1, 1)]), tmp_path / "out")
    m2 = AutoModelForCausalLM.from_pretrained(out)
    with apply(lm, [HeadAblation(1, 1)]): ref = logits(lm, "abc")
    assert torch.allclose(m2(**lm.tokenizer("abc", return_tensors="pt")).logits, ref, atol=1e-4)
    assert (out / "recipe.nester.json").exists() and "neshter" in (out / "README.md").read_text()
    assert verify_source_untouched(lm)

def test_failed_save_leaves_nothing(lm, tmp_path, monkeypatch):
    monkeypatch.setattr(lm.model, "save_pretrained", lambda *a, **k: (_ for _ in ()).throw(OSError("disk dolu")))
    base = logits(lm, "abc")
    with pytest.raises(OSError): save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 1)]), tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.tmp").exists()
    assert torch.equal(logits(lm, "abc"), base) and verify_source_untouched(lm)

def test_refusal_card_tag(lm, tmp_path):
    out = save_safetensors(lm, recipe_for(lm, [HeadAblation(0, 0)], concept="refusal"), tmp_path / "o")
    assert "safety-ablated" in (out / "README.md").read_text()
```

- [ ] **Step 2–4:** FAIL → implement → PASS. (`to_gguf` ve `ollama_create` için yalnızca komut satırını kuran yardımcıyı test et: `subprocess.run` monkeypatch'lenir, argümanlar doğrulanır.)
- [ ] **Step 5:** Commit `feat: bake interventions, atomic save, model card, gguf/ollama export`.

---

### Task 11: Yerel API sunucusu ve arka plan işleri

**Files:**
- Create: `neshter/server/__init__.py`, `neshter/server/app.py`, `neshter/server/jobs.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: önceki tüm görevlerin public fonksiyonları.
- Produces (`jobs.py`): `class JobManager: def submit(self, fn: Callable[[Callable[[float, str], None], threading.Event], Any]) -> str; def status(self, job_id: str) -> JobStatus; def cancel(self, job_id: str) -> None` — tek işçi iş parçacığı (GPU'da aynı anda tek iş); `JobStatus(id, state: Literal["queued","running","done","error","cancelled"], progress: float, message: str, result: Any | None, error: str | None)`.
- Produces (`app.py`): `def create_app(token: str, port: int, roots: dict | None = None) -> FastAPI`. Güvenlik ara katmanı (Global Constraints). Uçlar (hepsi JSON):
  - `GET /api/system` → SystemInfo
  - `GET /api/models` → ModelEntry listesi + `memory_bytes`, `badge`, `license`
  - `POST /api/models/sample` → iş; `huggingface_hub.snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")`, ilerleme bayt cinsinden; sonuç: yerel yol (testte `snapshot_download` monkeypatch'lenir)
  - `POST /api/mlp {prompt, token_index}` → `top_mlp_neurons` sonucu
  - `POST /api/session/load {path}` → iş id'si; sonuç: arch + name
  - `POST /api/activity {prompt, token_index, scale: "layer"|"global"}` → `{tokens, power: [[float]]}`
  - `POST /api/effect {heads: [[l,h]], prompts?: [str]}` → iş; sonuç EffectResult
  - `POST /api/hunt {concept_id | concept, method}` → iş; sonuç HuntResult (yön base64)
  - `POST /api/compare {prompt, recipe_ops}` → `{original, modified, mode}`
  - `POST /api/suite {recipe_ops, target}` → iş; sonuç CategoryResult listesi + Summary
  - `POST /api/export {recipe_ops, name, format: "safetensors"|"gguf_q4_k_m"|"gguf_q8_0", ollama: bool, out_dir}` → iş
  - `POST /api/recipe/check {recipe}` → RecipeCheck
  - `GET /api/jobs/{id}` → JobStatus; `POST /api/jobs/{id}/cancel`; `GET /api/jobs/{id}/events` → SSE (`progress`, `done`, `error`)
  - `neshter/web/dist` varsa `/` altında statik servis (Plan 2).

- [ ] **Step 1: Failing tests** (FastAPI `TestClient`, tiny model, `roots` boş kök)

```python
def test_rejects_missing_token(client): assert client.get("/api/system").status_code == 403
def test_rejects_foreign_origin(client, token):
    r = client.get("/api/system", headers={"X-Neshter-Token": token, "Origin": "https://kotu.example"}); assert r.status_code == 403
def test_accepts_local_origin(client, token, port):
    r = client.get("/api/system", headers={"X-Neshter-Token": token, "Origin": f"http://127.0.0.1:{port}"}); assert r.status_code == 200
def test_load_then_activity(client, auth, tiny_model_dir):
    job = client.post("/api/session/load", json={"path": str(tiny_model_dir)}, headers=auth).json()["job_id"]
    wait_done(client, auth, job)
    r = client.post("/api/activity", json={"prompt": "merhaba", "token_index": -1, "scale": "layer"}, headers=auth).json()
    assert len(r["power"]) == 2 and len(r["power"][0]) == 4
def test_cancel_job(client, auth): ...   # uzun suite işi başlat → cancel → state "cancelled"
def test_recipe_check_endpoint_rejects_extra_field(client, auth): ...  # 422
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat: local API with token/origin guard and job queue`.

---

### Task 12: Komut satırı ve Windows başlatıcı

**Files:**
- Create: `neshter/cli.py`, `neshter.bat`, `README.md` (İngilizce, en üstte Türkçe bağlantı), `README.tr.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `create_app`, `load_model`, `load_recipe`, `check_against`, `to_interventions`, `save_safetensors`, `to_gguf`.
- Produces:
  - `def main(argv: list[str] | None = None) -> int`
  - `neshter` (varsayılan `serve`): boş port bulur, `secrets.token_urlsafe(24)`, uvicorn'u `127.0.0.1`'de başlatır, tarayıcıyı `http://127.0.0.1:<port>/#token=<token>` ile açar (`--no-browser` bayrağı).
  - `neshter apply <recipe.nester.json> <model_path> --out <dir> [--format safetensors|gguf_q4_k_m|gguf_q8_0]`: tarifi kontrol eder; `errors` varsa çıkış kodu 2 ve hataları yazar; SHA uyuşmazlığında uyarı yazar, `--force` olmadan çıkış kodu 3.
  - `neshter.bat`: `uv` yoksa resmi kurulum betiğiyle kurar, ardından `uvx neshter`.

- [ ] **Step 1: Failing tests**

```python
def test_apply_roundtrip(tmp_path, tiny_model_dir, lm):
    save_recipe(recipe_for(lm, [HeadAblation(1, 0)]), tmp_path / "r.nester.json")
    assert main(["apply", str(tmp_path / "r.nester.json"), str(tiny_model_dir), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "model.safetensors").exists()

def test_apply_rejects_out_of_range(tmp_path, tiny_model_dir, lm, capsys):
    save_recipe(recipe_for(lm, [HeadAblation(7, 0)]), tmp_path / "r.nester.json")
    assert main(["apply", str(tmp_path / "r.nester.json"), str(tiny_model_dir), "--out", str(tmp_path / "o")]) == 2

def test_apply_sha_mismatch_needs_force(...): ...  # 3, sonra --force ile 0
```

- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Tüm test paketi: `uv run pytest -v` → hepsi PASS.
- [ ] **Step 6:** Commit `feat: cli, windows launcher, readme`.

---

### Task 13: Gerçek model duman testi (yalnızca 10:00'dan sonra, GPU serbestken)

**Files:**
- Create: `tests/integration/test_real_model.py` (`@pytest.mark.integration`, varsayılan çalıştırmada atlanır)

- [ ] **Step 1:** Qwen2.5-0.5B-Instruct'ı HF'den indir (`huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct`).
- [ ] **Step 2: Test:** yükle → `hunt(code, "attribution")` → ilk 6 head'in `real_effect.kl > random_kl` → `generate_pair` ile kod isteğinde orijinal `detect_code` True → `save_safetensors` → yeniden yüklenen modelin cevabı hook'lu cevapla aynı.
- [ ] **Step 3:** `uv run pytest -m integration -v` → PASS. Sonuçları (süre, bellek tepe değeri) `docs/notes/2026-11-10-smoke.md`'ye yaz.
- [ ] **Step 4:** Commit `test: real-model smoke test (Qwen2.5-0.5B)`.
