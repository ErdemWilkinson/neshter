# Neşter — 3B Bağ Görünümü ve Bağ Kesimi Tasarım Spec'i (Plan 3)

Tarih: 2026-11-10 · Durum: Erdem'in 6 kararı işlendi (2026-11-11) · Dayanak: Erdem'in 5 + 6 kararı
Üst belge: [2026-11-10-neshter-design.md](2026-11-10-neshter-design.md)

**2026-11-11 kararları (Erdem):**
1. Head kaynaklı okuma kesimi yalnızca head'in **ortalama yönünü** siler (rank 1), tüm yazma uzayını değil (§4.1).
2. KV grubunda paylaşılan V kanalı okuma kesimi **grup kesimi** olarak kaydedilebilir; etkilenen head'ler
   ve etkilenme miktarı 3B'de ve panelde açıkça gösterilir (§4.4, §8). Kesme ve kaydetme ayrı düğmelerdir;
   kaydedilmemiş kesimle çıkarken uyarı verilir (§8).
3. Sepette yalnızca keşif kesimleri varken "Kaydet": onay sorulur, onaylanırsa bu kesimler atılır ve
   karşılaştırma yeniden koşulur (§7).
4. Kavram grafiğinin çıkış düğümü davranış yönünün sinyalidir (öneri a).
5. Rastgele kontrol sayısı ayarlanabilir: varsayılan 20, en az 10, en çok **uygun havuzun tamamı** (§6).
6. 3B kütüphanesi @react-three/fiber; görsel dil fütüristik (§8). Ek istekler: 2B → 3B geçiş ekranı ve
   artık akımın "nehirler" olarak görünmesi (§8), yapay head ekleme (§13).

**Kapsam notu:** Bu spec, ana spec'in §3'teki "3B görünüm v0.1'de YOK" hükmünü ve §6'daki
"3B görünüm" ile "MLP nöron haritası" kapsam dışı maddelerini **v0.1 için geçersiz kılar**.
Sıra değişmez: önce v0.1.1 düzeltmeleri (audit-technical: fp32 CPU, QR ile çoklu yön, Gemma2
yön kaydı, katman eşli kontrol), sonra 2B arayüz (Plan 2), sonra bu. Herkese açık sürüm üçü
bitince çıkar. Bu spec v0.1.1 düzeltmelerinin yapıldığını varsayar.

## 1. Ne

"Ağ haritası" ekranına ısı haritasının yanında ikinci bir mod eklenir: **3B bağlar**. Tek bir
cevabın (ya da bir kavramın) **tahmini atıf grafiği** katman katman çizilir. Kullanıcı bir bağı
seçer, **gerçekten keserek ölçer**, sonra sepete ekler. Kayıtta yalnızca ağırlığa **birebir**
gömülebilen kesimler kabul edilir.

## 2. Grafik modeli

**Düğümler**
- `token`: girdi tokenleri (alt kat). Tek cevap grafiğinde pozisyon başına bir düğüm; kavram
  grafiğinde token kimliği (`token_id`) başına bir düğüm (pozisyonlar ve istekler toplanır).
- `head`: (katman, Q head). Pozisyonlar üzerinden toplanır (2B ısı haritasının birimiyle aynı).
- `neuron`: (katman, MLP nöronu = `down_proj` girdisinin bir kanalı). Pozisyonlar toplanır.
- `output`: tek cevap grafiğinde tahmin edilen token `y` (son pozisyon); kavram grafiğinde
  "kavram sinyali · K{ref}" (bkz. §5; bu düğüme giren kenarlar kesilemez, §4.2).

**Metrik M (grafiğin açıkladığı sayı)**
- Tek cevap: `M = logit_y − mean_v(logit_v)`, son pozisyonda, Gemma2 logit softcap'inden **önce**.
  `y` varsayılan olarak 1. tahmindir; kullanıcı ilk 5 tahminden birini seçebilir.
- Kavram: `m = resid[ref, son] · r_ref` (hunt ile aynı tanım, aynı `ref_layer`).

**Yazıcı katkısı `a_A`** (A'nın artık akıma yazdığı vektör, pozisyon başına): A sıfırlanınca
artık akıma yazılan değerin değişimi. Llama/Qwen2'de doğrusal: head için `W_O[:, h]·z_h`,
nöron için `W_down[:, n]·act_n`, token için gömme satırı (Gemma2'de `×√d`). Gemma2'de post-norm
yüzünden fark olarak hesaplanır: `post_norm(o) − post_norm(o − W_O[:, h]·z_h)`.

**Kenar türleri ve kanallar (v0.1)**

| Tür | Kaynak → hedef katman koşulu | Kanal |
|---|---|---|
| token→head | her katman | Q, V |
| token→neuron | her katman | resid |
| head→head | A.katman < B.katman | Q, V |
| neuron→head | A.katman < B.katman | Q, V |
| head→neuron | A.katman ≤ B.katman (aynı katmanda attention MLP'den önce) | resid |
| neuron→neuron | A.katman < B.katman | resid |
| düğüm→output | her yazıcı | logit |

**K bileşimi v0.1'de yok** (ne grafikte ne kesimde); UI "K-bileşimi gösterilmiyor" notunu
grafik başlığında gösterir. Q kanalı B'nin kendi pozisyonundaki okumadır; V kanalı B'nin
kaynak pozisyonlarındaki okumadır.

## 3. Tahmini ağırlıklar ("tahmini")

Tek ileri + tek geri geçiş, eager attention (desenler gerekir), `logits_to_keep=1`, fp32 tercih.
Normlar **dondurulmuş** kabul edilir (rms sabit; doğrusallaştırma) ve bu yüzden her şey "tahmini"dir.

- **Düğüm önemi:** `I_A = Σ_p ∇_{x_A[p]} M · x_A[p]` (gradyan × aktivasyon). `x_A`: head için
  `z_h` (o_proj girdisi dilimi), nöron için `act_n`, token için gömme vektörü. Pozitif = M'yi
  destekler. Bu, sıfır-ablation'ın doğrusal tahminidir (audit §5a `score_zero`).
- **Kenar ağırlığı (residual üzerinden sanal ağırlık × aktivasyon × gradyan):**
  `e(A→B, c) = Σ_p ⟨ (γ_B ⊙ g_B^c[p]) / rms_B[p] , a_A[p] ⟩`.
  `g_B^c[p]`, M'nin B'nin c kanalındaki ön-aktivasyonuna göre gradyanının B'nin okuma satırlarına
  geri izdüşümüdür:
  - nöron: `W_gate[n]ᵀ·∂M/∂gate_n + W_up[n]ᵀ·∂M/∂up_n`;
  - head Q: `W_q,hᵀ·∂M/∂q_h` (RoPE öncesi q; δ kancası q_proj çıktısında);
  - head V: `W_v,gᵀ·Σ_q A_h[q,p]·∂M/∂z_h[q]` (GQA'da v grubu paylaşıldığı için head başına
    gradyan, desen `A_h` ve `∂M/∂z_h`'den analitik kurulur);
  - output: `u_y = W_U[y] − mean_v W_U[v]`, `γ_final` ve son rms ile.
  `γ` Gemma2'de `1 + w`'dir. Tek token pozisyonu kısıtları (Q aynı pozisyon, V kaynak ≤ hedef,
  MLP aynı pozisyon) toplamda kendiliğinden sağlanır. `e > 0`: bağ M'yi artırır ("artıran").
- **Budama (varsayılan / sınır):** katman başına en güçlü `heads_per_layer = 6` (1–32) head ve
  `neurons_per_layer = 12` (0–64) nöron |I|'ya göre; sonra toplamda `max_nodes = 300` (≤ 600).
  Tüm token düğümleri (tek cevap: T ≤ 128; kavram: |I|'ya göre en güçlü 16 token kimliği) ve
  output her zaman tutulur. Tutulan düğümler arasındaki tüm geçerli (tür, kanal) çiftleri aday
  olur; `|e| ≥ 0.01·max|e|` olanlardan en büyük `max_edges = 500` (≤ 2000) tanesi kalır. Kenarı
  kalmayan ara düğümler atılır. Arayüz varsayılan olarak en güçlü 200 kenarı gösterir.
- **Bellek:** gradyan/yazıcı vektörleri yalnızca tutulan düğümler için, fp32. 7B, T=128, 300
  düğümde yaklaşık 1.1 GB; kenarlar katman çifti başına akışla (streaming) hesaplanır.

## 4. Kesim semantiği

### 4.1 İki kesim

| | **Yol kesimi** (`path`) | **Okuma kesimi** (`read`) |
|---|---|---|
| Anlamı | B, c kanalında `norm(x − a_A)` okur; A'nın başka her etkisi aynı kalır | B'nin c kanalındaki okuma satırlarından A'nın yazma uzayı silinir |
| Kanca | B'nin norm girdisi `x` ve aynı geçişteki `a_A` yakalanır; B'nin kanal çıktısı `W·norm_B(x − a_A) + b` ile değiştirilir | `out_B ← out_B − (n·Q)(R_B·Q)ᵀ` (`n`: norm çıktısı) |
| Ağırlığa gömme | **Gömülemez**: `rms(x)` A'yı içerdiği için hiçbir satır düzenlemesi eşdeğer değil | **Birebir**: `R_B ← R_B − (R_B·Q)·Qᵀ` |
| Yan etki | yok (tanım gereği tam yol) | B, bu uzaya **kim yazarsa yazsın** artık okumaz; rms hâlâ A'yı içerir |
| Rozet | `yalnızca keşif` | `kaydedilebilir` (bkz. 4.2 istisnaları) |

- `Q` = `γ_B ⊙ S_A` matrisinin QR ile ortonormal tabanı; `S_A` her kaynak için **rank 1**'dir:
  nöron → `W_down[:, n]`; token → `E[token_id]`; head → head'in **ortalama yazma yönü**
  `u_A = normalize(mean_p a_A[p])` (Erdem kararı 1). Ortalama, tek cevap grafiğinde isteğin
  pozisyonları, kavram grafiğinde train pozitiflerinin pozisyonları üzerinden alınır; `u_A` tarife
  vektör olarak yazılır (§7), böylece kesim veriden bağımsız tekrar üretilir. Head için ek gösterge
  **kapsama** = `Σ_p ⟨a_A[p], u_A⟩² / Σ_p ‖a_A[p]‖²` ("bu kesim head'in bu istekteki yazısının %K'sini
  kapsar"); %50 altında uyarı: "bağ tam kesilmiyor, yol kesimiyle karşılaştır". Gemma2'de yazıcı
  tarafına `γ_post` de çarpılır (`diag(γ_post)·S_A`).
  Taban **orijinal** ağırlıklardan hesaplanır; aynı B'ye birden çok okuma kesimi varsa tabanlar
  birleştirilip tek QR yapılır (sıralı izdüşüm yok; audit §1 ile aynı ilke).
- `R_B`: nöron için `W_gate[n]` ve `W_up[n]` satırlarının ikisi; head Q için `W_q`'nun h'ye ait
  head_dim satırı; output için `W_U[y]` satırı. Bias'lara (Qwen2 q/k/v) dokunulmaz.
- RoPE q_proj çıktısından sonra uygulanır ve doğrusaldır; satırdan silinen bileşen RoPE sonrası
  da sıfırdır → gömme kesin. Gemma2 softcap'leri sonradan geldiği için kesinliği bozmaz.
- **Yan etki göstergesi:** `side_effect = ‖R_B·Q‖²_F / ‖R_B‖²_F` ("B'nin okumasının %X'i
  silinir"). %50 üstünde uyarı. Kaynaklar rank 1 olduğu için bu oran genelde küçüktür; asıl
  yan etki §4.4'teki etkilenme önizlemesinde görünür.
- Yol kesimi her pozisyonda uygulanır ve girdiden bağımsız tanımlıdır (herhangi bir istekte
  çalışır). V kanalında kanca, yalnızca h head'inin `z_h[q]` değerine `Σ_p A_h[q,p](v'_p − v_p)`
  ekler (gruptaki diğer head'ler değişmez); output kanalında yalnızca `logit_y` yeniden hesaplanır.
- Token kesimleri (iki semantikte de) `token_id`'ye bağlıdır: o token **her pozisyonda ve her
  istekte** B için kesilir; okuma kesiminde gömmesi benzer tokenler de kısmen etkilenir. Tek cevap
  grafiğinde aynı token birden çok pozisyondaysa ipucu bunu söyler.

### 4.2 Kaydedilebilirlik tablosu (v0.1)

| Kenar | Yol kesimi | Okuma kesimi | Rozet |
|---|---|---|---|
| X→head, **Q** | keşif | tek head'e özel, birebir | kaydedilebilir |
| X→head, **V** | keşif (head'e özel, bkz. 4.1) | **grup kesimi**: `W_v`'nin g grubuna ait satırları düzenlenir → gruptaki tüm Q head'leri etkilenir (Erdem kararı 2) | kaydedilebilir · grup (N head) |
| X→neuron | keşif | birebir | kaydedilebilir |
| düğüm→output (bağımsız lm_head) | keşif | birebir | kaydedilebilir |
| düğüm→output (bağlı lm_head) | keşif | lm_head ayrılarak birebir (disk tahmini artar, yön kaydıyla aynı mekanizma) | kaydedilebilir |
| X→"kavram sinyali" | — (ölçüm sondası, model bileşeni değil) | — | kesilemez; "Yönü kaldır" (2B yön kesimi) önerilir |

- **Grup kesimi (V):** `R_B` = g grubunun `W_v` satırları (head_dim satır). Okuma kesimi ölçümü
  bake ile yapıldığı için ölçülen ΔM zaten **bütün grubun** etkisidir; panel bunu "grup kesimi:
  N head birlikte etkilenir" diye yazar. Gruptaki her head için ayrı satır §4.4'te.

### 4.3 Ölçüm hangi semantiği ölçtüğünü söyler

Ölçüm her zaman yol kesimini, kenar kaydedilebilirse ek olarak okuma kesimini ölçer. Her sonuç
`semantics: "path" | "read"` taşır ve panelde iki ayrı satır olarak görünür ("Yol kesimi: ΔM …" /
"Okuma kesimi (kaydedilecek olan): ΔM …"). Sepet kalemi hangi semantiği taşıdığını ve o semantiğin
ölçümünü saklar. Okuma kesimi ölçümleri "ölçtüğün kaydettiğindir" ilkesiyle `bake()` + `undo()`
üzerinden yapılır.

### 4.4 Etkilenme önizlemesi (her okuma kesiminde, kesmeden önce)

Okuma kesimi B'nin o yöndeki okumasını **kim yazarsa yazsın** siler. Kullanıcı neyi feda ettiğini
kesmeden önce görür (Erdem kararı 2):
- **Etkilenen okuyucular:** B; grup kesiminde gruptaki diğer N−1 head de. Her biri için tahmini
  okuma kaybı `side_effect` ve, grafikteki gelen kenarlar üzerinden, tahmini `Σ|Δe|`.
- **Etkilenen kaynaklar:** grafikte B'ye (ve grup kesiminde gruptaki head'lere) gelen her kenar
  A' için kesimden sonraki tahmini ağırlık `e' = e(A'→B) − ⟨(γ_B⊙g_B)/rms, P_Q a_A'⟩`; kaybı
  `|e − e'| / |e|` ≥ %10 olanlar "olası etkilenecek birimler" listesinde, büyükten küçüğe.
- Hepsi "tahmini" etiketlidir; gerçek etki ölçümle (§6) ve Karşılaştır'la görülür.
- Önizleme sunucuda hesaplanır (`/api/graph/preview_cut`, §7) ve hızlıdır (ileri geçiş yok;
  grafikteki saklı vektörlerle).

## 5. Grafik çıkarma algoritması ve maliyet

**Tek cevap ("İçine bak"):** 2B ısı haritasıyla aynı kodlama. (1) ileri+geri geçiş, δ kancaları:
o_proj girdisi, q_proj/v_proj/gate_proj/up_proj çıktıları, gömme çıktısı, norm girdileri (rms).
(2) tüm head/nöron/token için `I`. (3) budama. (4) aday kenarlar ve `e`. (5) budama, JSON.

**Kavram ("Davranış bul"dan):** hunt'ın split'i ve `ref_layer`'ı kullanılır; ileri geçiş
`ref_layer`'da durur (audit §7). Geçiş 1: train pozitiflerinden en çok `max_prompts = 16` istek
üzerinde `I` ortalaması → düğüm seçimi. Geçiş 2: aynı istekler üzerinde tutulan düğümler için `e`,
ortalama. Token düğümleri `token_id` başına toplanır. `ref_layer` sonrası katmanlar grafikte yoktur
(başlıkta yazar). Güvenlik kavramından (ör. refusal) gelen grafik `safety: true` taşır.

**Maliyet tahmini** (CPU fp32, AVX2, 4 iş parçacığı; audit §0 ölçümlerinden ölçeklenmiş, T≈64;
Plan 3'ün ilk görevi bu sayıları ölçüp düzeltmektir):

| Model | Tek cevap grafiği | Kavram grafiği (16 istek, 2 geçiş) | Bir kenar ölçümü (N=20, iki semantik) | fp32 RAM |
|---|---|---|---|---|
| 0.5B | 2–4 sn | ~1 dk | ~1–2 dk | ~2.5 GB |
| 1.5B | 6–12 sn | 3–5 dk | ~3–6 dk | ~7.5 GB |
| 7B | 30–60 sn | 15–30 dk | ~15–30 dk | ~36 GB |

bf16'yı yerel desteklemeyen CPU'da geri geçiş ~54× yavaştır: 7B tek cevap grafiği ~15–30 dk olur.
İş başlamadan önce süre tahmini (ETA) gösterilir; kullanıcı onaylar, iş iptal edilebilir.
Kabul ölçütü: 0.5B tek cevap grafiği referans makinede ≤ 10 sn.

## 6. Ölçüm

Kenar başına, seçilen semantik(ler) için:
- **Hedef etkisi ΔM:** tek cevap → kullanıcının isteğinde M'nin değişimi (n = 1, SE yok); kavram →
  ayrılmış val pozitiflerinde `m` değişiminin ortalaması ve SE'si.
- **Genel bozulma:** nötr setin sabit (seed'li) 16 isteğinde, son 16 pozisyonda KL (2B ile aynı
  tanım, "genel bozulma (KL)" adıyla).
- **Kontrol (rastgele ablation):** `n_random` **katman eşli rastgele kenar** aynı şekilde kesilir:
  A' A'nın katmanından ve türünden, B' B'nin katmanından ve türünden, aynı kanal ve aynı semantik;
  token kaynağı için başka bir pozisyon (tek cevap) ya da istek kümesinden başka bir token kimliği
  (kavram). Ölçülen kenarın kendisi hariç. **Havuz** = bu koşullara uyan tüm kenarların sayısı
  (grafikte görünmeyenler dahil; modelin yapısından sayılır). `n_random` varsayılan 20, en az 10,
  en çok havuzun tamamı (Erdem kararı 5); kaydırıcı her değerde ETA'yı gösterir, büyük değerler
  için "≈ {ETA}, sisteminize bağlı" uyarısı. Havuz 10'dan küçükse havuzun tamamı kullanılır ve
  sonuç "kontrol havuzu küçük (n = k)" der. Yerine koyarak çekme yoktur.
- **İstatistik (2B ile ortak fonksiyon, `metrics.control_stats`):** `p = (1 + #{|ΔM_rand| ≥ |ΔM|}) / (1 + N)`,
  yüzdelik (rastgelelerin |ΔM|'den küçük olan payı), rastgele dağılımın %5–95 bandı; aynı üçlü KL
  için de. `p > 0.05` ise "belirsiz: rastgele bağlardan ayırt edilemiyor". Oran ("×2.6") gösterilmez.
- **Kapı:** "Kes ve sepete ekle" yalnızca ilgili semantik ölçüldükten sonra açılır (2B ile aynı
  UI kapısı; sonuç belirsiz olsa da açılır, sonuç düz cümleyle yanında yazar).
- Tahmin ile ölçüm her zaman yan yana: "tahmini e" ve "ölçülen ΔM" ayrı sütunlar.

## 7. API

Tümü mevcut iş (job) + SSE altyapısıyla, token/Origin kurallarıyla, `extra="forbid"` şemalarla.

**`POST /api/graph`** → `{job_id}`. İstek:
```
GraphReq: mode: "prompt" | "concept"
          prompt: str | None            # mode=prompt (1..128 token)
          target_token_id: int | None   # None = 1. tahmin
          concept_id: str | None ; concept: Concept | None   # mode=concept, tam biri
          seed: int = 0 ; max_prompts: int = 16 (4..32)
          budget: {heads_per_layer: 6, neurons_per_layer: 12, max_nodes: 300, max_edges: 500}
```
Sonuç:
```
{graph_id, mode, estimated: true, safety: bool, dtype, elapsed_s, warnings: [str],
 metric: {kind: "logit_diff", token_id, text} | {kind: "concept_proj", ref_layer},
 tokens: [{pos, token_id, text}] | [{token_id, text, count}],
 nodes: [{id: "t:3" | "tid:1234" | "h:12:5" | "n:7:1203" | "out", kind, layer, index, score}],
 edges: [{id, src, dst, channel: "q"|"v"|"resid"|"logit", weight,
          badge: "kaydedilebilir"|"kaydedilebilir · grup"|"yalnızca keşif"|"kesilemez",
          control_pool_size,
          read: {available, bakeable, scope: "head"|"kv_group"|"neuron"|"logit", side_effect} | null,
          ops: {path: EdgePathOp | null, read: EdgeReadOp | null}}],
 pruning: {scored_nodes, kept_nodes, candidate_edges, kept_edges, edge_threshold}}
```
`graph_id` oturuma bağlıdır; model değişince geçersizdir (404 "grafik eskidi, yeniden çiz").
Sunucu en son 4 grafiği tutar.

**`GET /api/graph/{graph_id}/node/{node_id}`** → ayrıntı: tür, katman, `score`; head için son
pozisyondan en çok bakılan 5 kaynak token ve OV yazma uzayının logit-lens ilk 10 artırdığı /
bastırdığı token; nöron için istek(ler)de en etkin olduğu tokenler ve `W_U·(γ_f⊙W_down[:,n])`
ilk 10 artırılan/bastırılan token. Hepsi "tahmini (doğrudan etki)" etiketli.

**`POST /api/graph/measure`** → `{job_id}`. İstek: `{graph_id, edge_ids: [str] (1..8),
semantics: ["path","read"] | null (null = kullanılabilen hepsi), n_random: int | null (null = 20;
10 ≤ n ≤ pool_size, aksi 422), seed: 0}`. Grafik sonucu her kenar için `control_pool_size` taşır.
Sonuç: kenar başına `{edge_id, results: [{semantics, delta_metric, delta_metric_se | null,
kl_neutral, kl_neutral_se, n_target, n_neutral, control: {n, pool_size, delta_band:
[p5, p95], kl_band: [p5, p95], percentile, p_value}, uncertain}]}`. Kenarlar tek tek ölçülür
(ortak etki Karşılaştır'da).

**`POST /api/graph/preview_cut`** (iş değil, senkron, ≤ 1 sn): `{graph_id, edge_id}` →
`{scope: "head"|"kv_group"|"neuron"|"logit", coverage | null, readers: [{node_id, side_effect,
est_loss}], sources: [{edge_id, node_id, e, e_after, loss_frac}]}` (§4.4).

**Tarif şeması (strict, yol içermez):**
```
SrcRef = {kind:"token", token_id} | {kind:"neuron", layer, index}
       | {kind:"head", layer, head, d_model, vector_b64}   # u_A, float32 LE base64 (DirectionOp ile aynı denetim)
DstRef = {kind:"head", layer, head, channel:"q"|"v"} | {kind:"neuron", layer, index}
       | {kind:"logit", token_id}
EdgeReadOp = {type:"edge_read", src: SrcRef, dst: DstRef}   # channel "v" = grup kesimi
EdgePathOp = {type:"edge_path", src: SrcRef, dst: DstRef}   # head src'de vektör yok sayılır
```
`channel:"v"` okuma kesimi model kartında ve tarif özetinde "grup kesimi: K7 KV grubu 2
(head 4, 5, 6, 7)" diye açık yazılır.
- `Op` (tarif ve kayıt) = `HeadOp | DirectionOp | EdgeReadOp`.
  `ExploreOp` (compare, suite) = `Op | EdgePathOp`.
- `Recipe.schema_version: Literal[1, 2]`; yazıcı her zaman 2 yazar; 1 okunmaya devam eder ve
  `edge_read` içeremez. `ArchInfo`'ya `vocab_size` eklenir; `check_against` katman sırası,
  aralıklar ve `token_id < vocab_size` denetler.
- `.nester.json` asla `edge_path` içermez (şema düzeyinde: `Recipe.interventions` yalnızca `Op`).
- Bake sırası: head susturma → yön kaldırma → okuma kesimleri. Okuma tabanları orijinal
  ağırlıklardan; dokunulan tensörler ayrık (okuma kesimi yalnızca okuyucu satırlarına yazar).

**Kayıt ve güvenlik:**
- `/api/export` `Op` kabul eder; `edge_path` gelirse 422: "Yol kesimleri ağırlığa gömülemez;
  sepetten çıkarın." `bake()` da `EdgeReadOp` dışındaki kenar türlerinde hata verir (çift kilit).
- Sepette keşif kesimi varken "Kaydet"e basılırsa onay penceresi (Erdem kararı 3): "Şu N kesim
  yalnızca keşif içindir ve kaydedilemez: … Bunlar sepetten atılacak, karşılaştırma yeniden
  koşulacak. Devam?" Onaylanırsa kesimler atılır, karşılaştırma ve test seti **yeniden koşulur**,
  sonuç gösterilir ve kullanıcı kaydı **bir kez daha** onaylar (sayılar değiştiği için). Ekrandaki
  sayılar her zaman kaydedilecek modelin sayılarıdır. Sepette yalnızca keşif kesimi varsa pencere
  "kaydedilecek kesim kalmıyor" der ve kayıt başlamaz.
- Model kartı her kenar kesimini semantiğiyle yazar: "okuma kesimi: K3 nöron 1203 → K7 head 5 (Q).
  B, bu nöronun yazdığı yönü hiçbir kaynaktan okumaz."
- Güvenlik işareti: `safety: true` grafikten gelen her kenarın anahtarı (`src`, `dst`) oturumda
  `session.safety_edges`'e eklenir. Kayıtta tarifte bu kümeden bir `edge_read` varsa
  `abliterated: refusal` + "safety-ablated" zorunludur; istemci kaldıramaz (`_is_safety` genişler).

## 8. 3B arayüz

**Kütüphane: @react-three/fiber** (+ yalnızca `OrbitControls` için @react-three/drei), npm ile
sürümü kilitli, Vite ile paketlenir; çalışma anında CDN yok. Gerekçe: arayüz zaten React; seçim,
sepet ve ölçüm durumu 2B ile aynı store'dan okunur, ısı haritası ile eşleme bildirimseldir. Ağır
işler React dışında: düğümler tür başına bir `InstancedMesh`, kenarlar tek `LineSegments2`
(köşe başına renk/kalınlık). 3B modu tembel yüklenen ayrı bir parçadır (dynamic import); 2B
kullanıcısı three.js indirmez.

**Yerleşim:** y ekseni hesaplama sırası. Tokenler y=0'da cümle sırasıyla; katman ℓ'nin attention
düzlemi y=2ℓ+1, MLP düzlemi y=2ℓ+2; output en üstte. Head'ler düzlemde head numarasına göre
(ısı haritasının sütunlarıyla aynı sıra), nöronlar skora göre. Düzlemler soluk dikdörtgen,
etiket "K12 · attn" / "K12 · MLP".

**Görsel dil (fütüristik, Erdem kararı 6):** zemin `#0B0E13`, panel `#121722` yarı saydam cam
(backdrop blur), çizgi `#222A38`, metin `#E6EAF2` / `#A7B0C0`; başlıklar Space Grotesk, metin
Inter, sayılar JetBrains Mono. Artıran kenar `#2EE6C5`, bastıran `#9FC3FF`; kalınlık ve opaklık
|e| ile; düğümler ve güçlü kenarlar hafif ışıma (bloom, @react-three/postprocessing), seçili
kenarda akış yönünde ilerleyen ışık parçacıkları. Turuncu `#FF7A45` yalnızca sepetteki (kesik
çizgili) kenarlar içindir. Rozetler: `kaydedilebilir` teal çerçeve, `kaydedilebilir · grup`
teal + grup simgesi, `yalnızca keşif` amber `#FFD27A`, `kesilemez` gri. Grafik başlığında
"tahmini" etiketi her zaman görünür. Bloom ve parçacıklar `prefers-reduced-motion`'da ve
"düşük güç" ayarında kapanır; anlam hiçbir zaman yalnızca ışımaya bağlı değildir.

**Nehirler (artık akım):** her token pozisyonunun artık akımı, kendi sütununda alttan üste akan
yarı saydam bir şerit ("nehir") olarak çizilir; şerit genişliği o katmandaki `‖resid‖` ile (katman
içinde normalize), akış animasyonu yönü yukarı. Head ve nöronların yazdığı katkılar nehre giren,
okudukları nehirden çıkan kollar olarak görünür; kenarlar bu kolların üzerinden geçer. "Nehirler"
açılıp kapatılabilir (varsayılan açık). Kavram grafiğinde tutulan token kimlikleri başına birer nehir.
Grafik JSON'una `rivers: [{pos | token_id, norms: [float] (katman başına)}]` eklenir.

**2B → 3B geçiş:** "2B ısı haritası | 3B bağlar | Liste" anahtarında 2B'den 3B'ye geçerken ısı
haritası ızgarası (katman × head) yatırılır ve her hücre 3B'deki head düğümünün yerine uçar
(~800 ms); renkler korunur, böylece kullanıcı hangi hücrenin hangi düğüm olduğunu görür. Ardından
nehirler ve kenarlar belirir. Geri dönüş tersidir. Reduced-motion'da geçiş anında olur.

**Etkileşim:**
- Yörünge: sürükle döndür, tekerlek yakınlaştır, sağ tık kaydır; "Önden / Yandan / Üstten /
  Sıfırla" düğmeleri.
- Düğüme tık → sağ panelde ayrıntı (§7 node endpoint), "2B'de göster".
- Kenar üstüne gelme → ipucu: kaynak → hedef, kanal, tahmini e, rozet, yan etki. 6 px ekran
  eşiğiyle seçim; düğüm üstüne gelince gelen/giden kenarlar vurgulanır.
- Süzgeç: "Gösterilen bağ: en güçlü N" kaydırıcısı (20 … max_edges, varsayılan 200); tür ve
  kanal açma/kapama (token, head, nöron, Q, V); "yalnızca kaydedilebilir".
- Seçili kenar paneli: önce **etkilenme önizlemesi** (§4.4): "Bu kesim şunları da etkiler" listesi
  (okuyucular ve kaynaklar, tahmini kayıp yüzdesiyle); 3B'de etkilenecek düğümler kırmızımsı bir
  halkayla, etkilenecek kenarlar kayıp oranında soluklaşarak önizlenir. Grup kesiminde gruptaki N
  head bir çerçeveyle birlikte vurgulanır ve "grup kesimi: N head birlikte etkilenir" yazar. Sonra
  "Gerçek etkiyi ölç · ≈ {ETA} dk" (kontrol sayısı kaydırıcısıyla, §6) → sonuç satırları (§4.3) →
  "Kes · okuma kesimi" ve/veya "Kes · yol kesimi (yalnızca keşif)". Tek sepet vardır.
- **Kes ≠ Kaydet** (Erdem kararı 2): "Kes" yalnızca oturumdaki (kancalı) modeli değiştirir ve
  sepete ekler. Diske yazma yalnızca ayrı **"Modeli kaydet"** düğmesiyle olur (§7 onay akışı).
  Üst çubukta "Kaydedilmemiş N kesim" göstergesi.
- **Ayrılma uyarısı:** sepette kaydedilmemiş kesim varken model değiştirme, uygulamadaki "Kapat"
  ya da sekmeyi kapatma → uygulama içi pencere "Ayrılıyorsunuz, kesimlerinizi daha kaydetmediniz.
  Kaydet / Kaydetmeden çık / Vazgeç". Sekme kapatma ve yenilemede tarayıcının kendi `beforeunload`
  uyarısı çıkar (tarayıcılar özel metne izin vermez). Sepet, sekme kapansa bile `localStorage`'da
  tutulur ve yeniden açılınca "Önceki oturumdan kaydedilmemiş N kesim var: geri yükle?" diye sorulur
  (yalnızca aynı modelin SHA-256'sıyla). Komut satırında Ctrl+C'de sunucu da aynı uyarıyı basar.
  Bu davranış 2B sepeti için de geçerlidir (Plan 2).
- "2B ısı haritası | 3B bağlar | Liste" anahtarı aynı ekranda; seçim ortaktır: head seçimi iki
  modda da aynı hücreyi işaretler, nöron seçimi 2B'deki "MLP · en güçlü" listesinde görünür.

**Sınırlar:** ≤ 600 düğüm, ≤ 2000 kenar (sunucu sınırlarıyla aynı). `frameloop="demand"`
(yalnızca etkileşimde çizim). Hedef: 300 düğüm / 500 kenarda tümleşik GPU'da 60 fps. Etiketler
yalnızca odak/seçili düğüm ve katman adları için. WebGL2 yoksa otomatik Liste görünümü.

**Erişilebilirlik:**
- Klavye: Tab ile tuvale gir; ↑/↓ katman değiştir, ←/→ katman içinde düğüm, Enter seç,
  E ile gelen/giden kenarlar arasında dolaş, M ölç, +/− yakınlaştır, R sıfırla, L Liste.
  Odak halkası çizilir; odaktaki düğüm/kenar `aria-live` ile okunur.
- `prefers-reduced-motion`: otomatik dönme yok, kamera geçişleri anında, nabız animasyonu yok.
- **Liste görünümü** (2B yedek): kenar tablosu (kaynak, hedef, kanal, tahmini e, rozet, ölçüm
  sonucu), |e|'ye göre sıralı, aynı süzgeçler ve aynı düğmeler; tamamen klavyeyle kullanılır.

## 9. Hata ve kenar durumları

- Model yok → mevcut 409. İstek 128 token üstü → 422 "Grafik için en çok 128 token".
- `graph_id` bilinmiyor/eski → 404 "grafik eskidi, yeniden çiz".
- Tüm `|I|` ≈ 0 (ör. M doymuş) → boş grafik + "Bu cevap için belirgin bir yol bulunamadı".
- Bellek tahmini RAM'i aşarsa iş başlamadan 422 + "T'yi ya da max_nodes'u küçült".
- bf16 CPU → ETA uyarısı ve "fp32 ile yeniden aç" önerisi.
- V kenarı: rozet `kaydedilebilir · grup`, ipucu "v ağırlıkları KV grubundaki N head'le
  paylaşılıyor; kesim hepsini etkiler". GQA olmayan modelde (grup = 1 head) rozet düz `kaydedilebilir`.
- Head kaynağında `coverage < 0.5` → sarı uyarı "bağ tam kesilmiyor".
- `side_effect > 0.5` → sarı uyarı. Bağlı lm_head + output okuma kesimi → kayıt diskinde artış.
- Ölçüm iptal → kısmi sonuç saklanmaz. Kontrol havuzu < 10 → havuzun tamamı, sonuçta "n = k".
- `.nester.json` v2'yi eski Neşter açarsa katı şema reddeder (beklenen); bilinmeyen alan → 422.
- Eager attention yalnızca grafik ve V yol kesimi ölçümü sırasında açılır, sonra geri alınır.

## 10. Test yaklaşımı

Küçük rastgele llama, qwen2 ve gemma2 modelleri (CI'da zaten kuruluyor), CPU fp32:
1. **Kanca == bake:** her kaydedilebilir tür için (token→head Q, token→neuron, head→head Q,
   head→neuron, neuron→head Q, neuron→neuron, düğüm→logit bağımsız ve bağlı lm_head) kanca ile
   gömülü modelin logitleri `≤ 1e-5`; bf16 kayıt → yeniden yükleme → kancalı bf16 ile aynı token
   dizisi (mevcut head testleri gibi).
2. **Okuma kesimi B'nin okumasını sıfırlar:** `‖R_B'·(γ_B⊙S_A)‖ / ‖R_B·(γ_B⊙S_A)‖ ≤ 1e-6`; norm
   çıktısına `S_A` uzayında rastgele vektör eklenince B'nin ön-aktivasyonu değişmez; B dışındaki
   satırlar bit düzeyinde aynı.
3. **Yol kesimi doğru ölçülür:** kanca, iki geçişli bağımsız bir referansla (önce `a_A` ve `x`
   yakalanır, sonra B'nin kanal girdisi elle `norm_B(x − a_A)` ile yerleştirilir) `≤ 1e-5`
   eşleşir; aynı katmandaki diğer head'ler/nöronlar ve V'de gruptaki diğer head'ler değişmez.
4. **Tahmin tutarlılığı:** `a_A`'yı `ε` ile ölçekleyen yol kesiminin sonlu farkı `e`'ye %5 içinde
   (hunt'taki FD testi gibi).
5. **Şema ve kayıt:** `edge_path` export'ta 422, `bake()`'te hata; v1 tarif yüklenir; v2'de
   bilinmeyen alan, aralık dışı katman/token, yanlış boyutlu head vektörü reddedilir.
   Grup kesimi (`channel:"v"`) kanca == bake ve gruptaki tüm head'lerin değiştiği, grup dışı
   head'lerin bit düzeyinde aynı kaldığı test edilir.
6b. **Ortalama yön:** head kaynaklı okuma kesimi yalnızca `u_A` bileşenini siler (rank 1);
   `coverage` bilinen girdilerle; önizleme `e_after` değerleri kanca ile ölçülen değişimle aynı
   işarette ve %10 içinde (doğrusal kısım).
6c. **Kontrol havuzu:** `pool_size` modelin yapısından doğru sayılır; `n_random > pool_size` 422;
   havuz < 10 iken havuzun tamamı.
6. **Güvenlik:** `safety: true` grafikten kenar içeren kayıt, istemci `concept=None` dese de
   etiketlenir.
7. **İstatistik:** tüm kontrol kenarları katman ve tür eşli; `p` formülü ve yüzdelik bilinen
   girdilerle.
8. **Arayüz:** rozet ve kapı mantığı (ölçülmeden Kes kapalı; keşif kesimiyle Kaydet → onay →
   atma → yeniden karşılaştırma → ikinci onay), ayrılma uyarısı ve sepeti geri yükleme, etkilenme
   önizlemesi vurgusu, 2B→3B geçişi (reduced-motion'da anında), WebGL yokken Liste, klavye
   gezinmesi; sabit grafik JSON'uyla Playwright smoke.

## 11. Belge düzeltmeleri

`docs/ogrenme/08-nester-haritasi.md` "aynı katman içi bağlar kaydedilebilir, katmanlar arası
yalnızca keşif" diyor; bu yanlış (aynı katmandaki head→MLP arasında da norm var). Doğrusu:
kaydedilebilirlik katman uzaklığına değil **kesim semantiğine** bağlıdır (§4). 04 §4.7'deki
"gömülebilir ama etki yaklaşıktır" cümlesi de iki semantiğe göre yeniden yazılır. Ana spec
§3 ve §6'ya bu spec'e işaret eden birer satır eklenir.

## 12. Kapsam dışı (v0.2+)

Statik yalnızca-ağırlık grafiği; K bileşimi; head kaynağında tam yazma uzayı (rank > 1) okuma kesimi; pozisyon başına
head/nöron düğümleri; kesimlerden sonra grafiği yeniden çizme; ortalama-ablation ile kenar kesimi;
SAE / transcoder özellikleri ve süper-düğümler; çok tokenli cevabın tamamını açıklayan grafik;
norm eşli kontrol ve özgüllük oranı (audit §4 v0.2); grafiği resim/dosya olarak paylaşma;
WebGPU, VR.

## 13. Yapay head (yönlendirme head'i) — Erdem'in isteği

**İstek:** modele yapay bir head ekleyip onu bir şeyle ilişkilendirmek; model bu yüzden yapay olarak
"halüsinasyon" yapabilir, ama istenen davranış da böyle ortaya çıkarılabilir.

**Dürüst çerçeve (arayüzde de yazar):** Ağırlıkları sıfır olan boş bir head akışa hiçbir şey yazmaz.
Model onu kullanmayı hiç öğrenmediği için kesilen bilgi (ör. "hayır" cevapları) kendiliğinden oraya
**kaçmaz**. Yapay head'in bir etkisi olması için ağırlıklarını biz yazarız: **ne zaman** çalışacağını
(QK: hangi tetikleyiciye bakacağı) ve **ne yazacağını** (OV: hangi yönü itecek). Bu, bilinen
**yönlendirme** (steering) tekniğinin bir head biçimidir. Modeli bir davranışa zorlar ve gerçekten
uydurma cevaplara yol açabilir. Neşter bunu "keşif ve deney" aracı olarak sunar, ölçmeden kaydetmez.

**Tanım (katman ℓ'de, mevcut head'lere ek tek bir head):**
- Tetikleyici `t̂` (birim vektör): bir kavram yönü (hunt sonucu) ya da bir tokenin gömmesi.
- Hedef `ŵ`: bir kavram yönü (+ ya da −) ya da bir tokenin çıkış satırı `W_U[y]` (o tokeni öne iter).
- Güç `α` ve keskinlik `β`. Her pozisyon q için skorlar `s[q,p] = β·⟨n_q, t̂⟩·⟨n_p, t̂⟩` (n: ℓ'nin
  attention norm çıktısı, nedensel maske), `z[q] = Σ_p softmax(s[q,·])_p · ⟨n_p, t̂⟩`, yazılan
  `α · z[q] · ŵ`. Yani tetikleyici geçmişte göründükçe head hedef yönü akışa yazar.
- 3B'de ayrı renkte (mor) bir düğüm olarak katmanına eklenir; kenarları ve nehre girişi görünür.

**Ölçüm:** diğer kesimlerle aynı kapı. Hedef etkisi (tek cevapta ΔM, kavramda val pozitiflerinde
`m` değişimi), genel bozulma KL, ve kontrol: aynı `α`, `β`, aynı tetikleyiciyle **rastgele hedef
yönlü** `n_random` yapay head (§6 istatistiği).

**Kapsam:**
- **Plan 3 (v0.1):** yalnızca keşif, kancayla. Rozet `yalnızca keşif`. Sepete girer,
  karşılaştırılabilir; kayıtta §7 onay akışıyla atılır.
- **v0.2:** kaydetme. Head sayısını değiştirmek tensör şekillerini değiştirir (GQA'da ayrıca K/V
  grubu gerekir); transformers ve llama.cpp uyumu mimari başına doğrulanmadan kayıt açılmaz.
  Alternatif olarak "head nakli": önemi ölçülerek ~0 bulunan bir head'in ağırlıkları yapay head ile
  değiştirilir (şekil değişmez). İkisi de v0.2 spec'inde tasarlanır.
- Model kartı, yapay head içeren her kayıtta "yapay head ile yönlendirildi" der. Hedef yön bir
  güvenlik kavramından geliyorsa (ör. refusal'ı bastırma) güvenlik etiketi zorunludur.

**API (Plan 3):** `ExploreOp`'a `SteerHeadOp = {type:"steer_head", layer, trigger: Vec | {token_id},
target: Vec | {logit_token_id}, alpha, beta}` (`Vec` = `{d_model, vector_b64}`) eklenir;
`Op`'a (kayıt) eklenmez. `/api/graph/measure` yerine `POST /api/steer/measure` aynı sonuç şemasıyla.
