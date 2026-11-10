[English README](README.md)

# Neşter

Neşter, kendi bilgisayarındaki dil modellerini açar, hangi dikkat head'inin ve davranış yönünün ne
işe yaradığını gösterir, seçtiğin parçaları ölçüp susturmana izin verir ve sonucu yeni bir model
dosyası olarak kaydeder.

**%100 yereldir:** içinde bulut API'si, hesap ve **telemetri yoktur**.

## Neleri destekler

- **Model aileleri:** Llama, Qwen2 ve Gemma2 (8B parametreye kadar). Diğer mimariler reddedilir.
- **v0.1 kesim türleri:**
  - **Head susturma:** bir sorgu (Q) dikkat head'ini susturur (`o_proj` sütunları sıfırlanır).
  - **Yön ortogonalizasyonu:** bir davranış yönünü ağırlıklardan çıkarır.
- Gemma2'de yalnızca head susturma kaydedilebilir; yön kaldırma v0.1'de bu ailede kaydedilemez.
- İkisi de tensör şeklini korur, bu yüzden sonuç GGUF ile uyumludur. Yapısal budama v0.1'de yok.
- Bir kesimin gerçek etkisi, rastgele head kontrolüne karşı KL ıraksaması olarak ve Wilson %95
  aralıklarıyla ölçülür; aralık 0'ı kapsıyorsa sonuç "belirsiz" diye bildirilir.

## Kurulum ve çalıştırma

```
uvx neshter
```

Windows'ta `neshter.bat` dosyasını çift tıklayabilir ya da çalıştırabilirsin; `uv` yoksa
[uv](https://docs.astral.sh/uv/) kurulur, ardından `uvx neshter` çalışır. Neşter yerel bir sunucu
başlatır ve tarayıcını açar. Seçenekler: `--no-browser`, `--port N`.

Paylaşılan bir tarifi (`.nester.json`) arayüz olmadan uygulamak için:

```
neshter apply recipe.nester.json model/klasoru --out yeni/model/klasoru --format safetensors
```

`--format` değeri `safetensors` (varsayılan), `gguf_q4_k_m` ya da `gguf_q8_0` olabilir. Çıkış
kodları: `0` başarılı, `1` hata, `2` tarif geçersiz ya da modele uymuyor, `3` tarif başka bir model
için hazırlanmış (SHA-256 uyuşmuyor; yine de uygulamak için `--force`), `4` llama.cpp gerekli ama
bulunamadı.

## GGUF ve Ollama

GGUF çıktısı için bilgisayarında [llama.cpp](https://github.com/ggml-org/llama.cpp) olmalıdır
(`convert_hf_to_gguf.py` ve `llama-quantize`). `NESHTER_LLAMA_CPP` ortam değişkenini onun klasörüne
ayarla ya da `llama-quantize`'ı `PATH`'e ekle. Yoksa safetensors olarak kaydet; GGUF seçeneği
"llama.cpp gerekli" der.

Kaydedilen GGUF'u Ollama'ya eklemek isteğe bağlıdır ve **varsayılan olarak kapalıdır**. Açılırsa
Neşter senin yerine `ollama create` çalıştırır.

## Güvenlik modeli

- Sunucu yalnızca `127.0.0.1`'e bağlanır. Başka bir adresi dinleme seçeneği yoktur.
- Başlangıçta rastgele bir oturum belirteci üretilir ve her API isteğinde (`X-Neshter-Token`)
  zorunludur. Belirteç, tarayıcıların sunucuya hiç göndermediği URL parçasında (`#`) taşınır.
- `Origin` `http://127.0.0.1:<port>` ya da `http://localhost:<port>` olmalıdır (ya da hiç
  olmamalıdır); `Host` yerel bir adres olmalıdır. Bu, siteler arası ve DNS-rebinding isteklerini
  engeller.
- `.pt` dosyaları yalnızca `torch.load(weights_only=True)` ile yüklenir; olmazsa reddedilir.
- `trust_remote_code` hiçbir zaman açılmaz.
- Kaynak model asla değiştirilmez: salt okunur açılır, SHA-256 değeri kayıttan önce ve sonra
  karşılaştırılır. Kayıt atomiktir (`.tmp` klasörü, sonra `os.replace`) ve var olan bir klasörün
  üzerine yazmaz.
- Tarifler güvenilmeyen girdidir: katı şema, kod yok, dosya yolu yok.

## Sorumlu kullanım

Neşter'de hazır bir `refusal` (reddetme) kavramı vardır. Bu kavram kesilerek kaydedilen modellerin
güvenlik davranışı zayıfladığı için, bu modellerde `abliterated: refusal` metadata'sı ve model
kartında `safety-ablated` etiketi bulunur. Bu modelleri sorumlulukla kullan ve temel modelin
lisansına uy (örneğin Llama lisansı, türetilmiş modeller için adlandırma ve kullanım politikası
şartları getirir).

## Durum

v0.1 arka ucu hazır: model yükleme, tarama, head etkinlik haritaları, davranış bulma, karşılaştırma,
tarifler, dışa aktarma, yerel API ve komut satırı. Web arayüzü yakında (Plan 2).

## Lisans

License: TBD
