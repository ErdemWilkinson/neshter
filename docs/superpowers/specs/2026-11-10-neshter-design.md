# Neşter — Tasarım Spec'i (v0.1)

Tarih: 2026-11-10 · Durum: onaylandı (5 rollü ajan tartışması + Erdem'in 4 kararı)

## 1. Ne

Neşter, kullanıcının kendi bilgisayarındaki lokal dil modellerini (Ollama, LM Studio,
HuggingFace) açan, içindeki attention head'leri ve davranış yönlerini görselleştiren,
seçilen parçaları ölçüp susturan ve sonucu yeni bir model dosyası olarak kaydeden,
tamamen lokal çalışan açık kaynak bir araçtır. İçinde hiçbir bulut API'si yoktur.

## 2. Kullanıcı akışı

1. **Karşılama:** "Ne yapmak istiyorsun?" → "Bir davranışı azalt" ya da "İçine bak".
   "Uzman modu" bağlantısı sihirbazı atlar. "Modeli küçült" seçeneği YOK.
2. **Model seç:** taranan modeller; her satırda yapı (katman · Q/KV head · MLP boyutu),
   açınca gereken bellek, rozet (rahat / yavaş · RAM desteği / sığmaz), lisans.
   Boş durumda "Örnek modeli indir · Qwen2.5-0.5B".
3a. **Ağ haritası (İçine bak):** prompt + token şeridi; katman×head ısı haritası,
   varsayılan ölçüm "head çıktı gücü, katman içinde 0–1", "tahmini" etiketi.
   Seçilen head'ler için "Gerçek etkiyi ölç" (KL + rastgele kontrol). "Sustur ve sepete
   ekle" yalnızca ölçümden sonra aktif. Sadeleştirilmiş harita herkese görünür.
3b. **Davranış bul:** hazır kavramlar: Kod yazma (varsayılan), Reddetme, Matematik,
   Duygu tonu. "Halüsinasyon" ve "Türkçe" YOK. Örnekler İSTEK'tir (prompt), cevap değil;
   en az 32 + 32; %20 doğrulamaya ayrılır. Yöntemler: Tahmini etki (attribution,
   GPU'da önerilen), Etkinlik farkı (`activation_diff`, CPU'da varsayılan), Davranış yönü. Sonuçlar türe göre ayrı listelenir.
4. **Karşılaştır ve Kaydet:** düz cümleli sonuç özeti; orijinal ve susturulmuş cevap
   yan yana (sıcaklık 0, seed, tek ağırlık kopyası); metrikler: hedef davranış oranı
   + rastgele kontrol, KL, perplexity, test seti tablosu (n, ±%95 GA, fark).
   Kayıt: GGUF Q4_K_M / Q8_0 / safetensors bf16; Ollama'ya ekle (varsayılan KAPALI);
   .nester.json tarifi; model kartı + lisans her zaman.

## 3. Kesin kararlar

- **Kesim türleri v0.1:** (a) head susturma, (b) davranış yönü kaldırma
  (ağırlık ortogonalizasyonu). Her ikisi de tensör şeklini korur → GGUF uyumlu.
  Kalıcı silme (yapısal budama) ve 3B görünüm v0.1'de YOK.
- **Head susturma kaydı:** o_proj'un o head'e ait sütunları sıfırlanır (sıfır-ablation).
  Ortalama-ablation yalnızca analizde seçenektir; kaydedilen modelle aynı olan
  sıfır-ablation varsayılandır. (Ortalama değer bias olmadan ağırlığa gömülemez.)
- **GQA:** kesim birimi sorgu (Q) head'idir; KV head'lerine dokunulmaz.
- **Gerçek etki:** KL(orijinal ‖ susturulmuş), ayrılmış nötr istekler üzerinde son
  16 token pozisyonunda ortalama; aynı sayıda rastgele head ile kontrol.
- **Karşılaştırma:** batch=2, müdahale yalnızca batch index 1'e: yalnızca head müdahaleli
  tariflerde geçerlidir. Yön içeren tarifler değiştirilmiş cevabı pişirilmiş, kayıt
  yuvarlamalı ağırlıklarda üretir (önce orijinal kancasız, sonra pişir, sonra geri al);
  test seti ağırlıkları bir kez pişirir. Bellek yetmezse sıralı moda düşer.
- **Hedef davranış dedektörleri:** Kod yazma → cevapta ``` kod bloğu, satır başında
  `def `/`function `/`class `/`fn `/`func `/`public `/`import `/`#include`/`#!/`/`SELECT`
  ya da kod gibi `{`/`;` ile biten satır; Reddetme → Türkçe+İngilizce ret kalıpları
  listesi. Ayrıca "bozuk" sonucu: bozuk bir cevap asla ablasyon başarısı sayılmaz.
- **Güven aralığı:** oranlar için Wilson %95. Eşleşmiş fark için "belirsiz" kararı
  McNemar exact (iki yönlü) ile verilir (p ≥ 0,05 ise belirsiz); gösterilen aralık
  Newcombe eşleşmiş skor aralığıdır; hedef dışı kategorilerde Holm düzeltmesi uygulanır
  ve en küçük saptanabilir düşüş belirtilir. "İki oranın farkı ±1.96·SE" kullanılmaz.
- **Reddetme:** listede 2. sırada, varsayılan değil; ilk kullanımda tek seferlik bilgi
  kartı; kayıtta metadata `abliterated: refusal` + model kartında "safety-ablated".

## 4. Güvenlik

- Sunucu yalnızca 127.0.0.1'e bağlanır; başlangıçta rastgele oturum token'ı üretilir,
  her istekte `X-Neshter-Token` başlığı zorunlu; Origin `http://127.0.0.1:<port>`
  ya da `http://localhost:<port>` değilse 403.
- `.pt`: yalnızca `torch.load(weights_only=True)`; başarısızsa reddet + dönüştür önerisi.
  "Kaynağa güveniyorum" kapısı v0.1'de YOK.
- `trust_remote_code=False` her zaman.
- Orijinal dosya yalnızca okunur; kayıt öncesi ve sonrası SHA-256 karşılaştırılır.
- Kayıt atomik: `<ad>.tmp` → tamamlanınca `os.replace`.
- `.nester.json` içe aktarım: katı şema (pydantic, `extra="forbid"`), yol içermez.
- Telemetri yok.

## 5. Teknoloji

- Python ≥ 3.11, paket adı `neshter`, kurulum klondan `uv run --frozen neshter` + Windows `neshter.bat` (uv yoksa kurmadan önce sorar); `uvx` yalnızca kilitsiz, kullanıcının sorumluluğunda.
- torch, transformers (GGUF okuma `gguf_file=` ile), safetensors, gguf, numpy,
  fastapi, uvicorn, pydantic v2, pyyaml.
- GGUF yazma: kullanıcı makinesinde llama.cpp `convert_hf_to_gguf.py` + `llama-quantize`
  bulunursa; yoksa safetensors kaydedilir ve GGUF seçeneği "llama.cpp gerekli" der.
- Arayüz: React + Vite + TypeScript, i18n (tr, en; sistem diline göre). Ayrı plan.
- Desteklenen mimariler v0.1: Llama, Qwen2, Gemma2 (≤ 8B).

## 6. Kapsam dışı (v0.2+)

Tauri .exe paketi, WebGPU demo, HF Hub'a yükleme, SAE/Neuronpedia, MLP nöron haritası
(v0.1'de yalnızca "en güçlü 50" listesi), ONNX, yapısal budama, 3B görünüm.
