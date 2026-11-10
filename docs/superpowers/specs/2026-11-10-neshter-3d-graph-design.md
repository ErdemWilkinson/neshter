# Neşter — 3B Bağ Görünümü ve Bağ Kesimi Tasarım Spec'i (Plan 3)

Tarih: 2026-11-10 · Durum: Erdem'in 6 kararı + tartışma uzlaşısı ve A–F cevapları işlendi (2026-11-11) · Dayanak: Erdem'in 5 + 6 kararı, `.superpowers/sdd/2026-11-11-decisions-debate/`
Üst belge: [2026-11-10-neshter-design.md](2026-11-10-neshter-design.md)

**2026-11-11 kararları (Erdem):**
1. Head kaynaklı okuma kesimi yalnızca head'in **ortalama yönünü** siler (rank 1), tüm yazma uzayını değil (§4.1).
   *Tartışma sonrası: rank 1 kalır; yön ortalama yerine BOS hariç baskın tekil vektör.*
2. KV grubunda paylaşılan V kanalı okuma kesimi **grup kesimi** olarak kaydedilebilir; etkilenen head'ler
   ve etkilenme miktarı 3B'de ve panelde açıkça gösterilir (§4.4, §8). Kesme ve kaydetme ayrı düğmelerdir;
   kaydedilmemiş kesimle çıkarken uyarı verilir (§8).
3. Sepette yalnızca keşif kesimleri varken "Kaydet": onay sorulur, onaylanırsa bu kesimler atılır ve
   karşılaştırma yeniden koşulur (§7).
4. Kavram grafiğinin çıkış düğümü davranış yönünün sinyalidir (öneri a).
5. Rastgele kontrol sayısı ayarlanabilir: varsayılan 20, en az 10, en çok **uygun havuzun tamamı** (§6).
   *Tartışma sonrası: sıralı test, `N_max` varsayılan 39, en az 19 (m = 1), CPU tavanı 199 (10 ile p ≤ 0.05
   hiç görülemezdi).*
6. 3B kütüphanesi @react-three/fiber; görsel dil fütüristik (§8). Ek istekler: 2B → 3B geçiş ekranı ve
   artık akımın "nehirler" olarak görünmesi (§8), yapay head ekleme (§13).
   *Tartışma sonrası: B, C ve A cevaplarına göre biçimlendi (tek gövde nehri, sahte ışıma, önce sabit vektör).*

**2026-11-11 tartışma sonrası** (yöntem eleştirmeni, UX mimarı ve kapsam şüphecisinin uzlaşısı +
Erdem'in A–F cevapları, "hepsi tamam"):
- A. **Önce sabit yönlendirme vektörü** (kanca, keşif) temel çizgidir; yapay yönlendirme head'i yalnızca
  norm eşli sabit vektörü yenerse sunulur; "modelin kendi attention'ından geçen gerçek head" sonraki,
  mimari başına bir projedir (§13).
- B. **Tek gövde nehri:** genişlik = katman başına logit-lens yakınlığı (cevaba ne kadar yaklaşıldığı);
  seçilen token kendi kolunu açabilir (§8). Pozisyon başına nehir yok.
- C. Varsayılan **sahte toplamalı ışıma**; gerçek bloom yalnızca isteğe bağlı "Yüksek" kalite kademesinde;
  kalite yöneticisi (governor). 2B → 3B uçuş kalır ama yalnızca grafik işi bitince oynar, renk geçişli;
  reduced-motion'da anında (§8).
- D. Ctrl+C: taslak otomatik kaydedilir, terminale tek bilgi satırı; engelleyen uyarı yok (§8).
- E. İkinci onay = yenilenmiş Karşılaştır ekranı + olağan "Kaydet" tıklaması; ikinci pencere yok (§7).
- F. Düğme etiketi **"Kes (geçici)"**, alt satır "Dosya değişmez" (§8).
- Uzlaşı (yöntem): `u_A` = BOS hariç baskın tekil vektör (karar 1'in rank-1 ilkesi korunur);
  kapsama ve `edge_share` B'nin okuma uzayında; §4.4 önizleme formülü düzeltildi + toplam `ΔM_read_est`
  + veri ağırlıklı `read_loss`; Gemma2 head kaynağında `γ_post` iki kez çarpılmaz; token okuma kesimi
  ortalanmış gömmeyle; `e`'de rms terimi (§3, §4).
- Uzlaşı (istatistik, karar 5'i günceller): sıralı Besag–Clifford permütasyon kontrolü, `N_max` varsayılan
  39, en az 19 (yalnızca m = 1), CPU tavanı 199; birincil kontrol grafik kenarı katmanı; ölçüm toplu başına
  Holm; oturum sayacı + "hipotez"; sabit seed; KL kontrolü yalnızca betimsel bant (§6).
- Uzlaşı (güven): 3B herkese gösterilmeden önce kalibrasyon kapısı (§6.1); tarif vektörleri "doğrulanmamış
  yön" etiketli + refusal kosinüs denetimi (§7); yönlendirmede döngüsellik işareti (§13).
- Uzlaşı (arayüz): taslak ve tercihler sunucuda, model SHA-256'sına göre (rastgele port localStorage'ı
  bozar); token yalnızca sessionStorage'da; drei yerine three-stdlib; latin-ext fontlar pakette, CDN yok;
  CSP uyumlu derleme; grafik JSON'unda kenar başına vektör yok (§7, §8).

**Kapsam notu:** Bu spec, ana spec'in §3'teki "3B görünüm v0.1'de YOK" hükmünü ve §6'daki
"3B görünüm" ile "MLP nöron haritası" kapsam dışı maddelerini **v0.1 için geçersiz kılar**.
Sıra değişmez: önce v0.1.1 düzeltmeleri (audit-technical: fp32 CPU, QR ile çoklu yön, Gemma2
yön kaydı, katman eşli kontrol), sonra 2B arayüz (Plan 2), sonra bu. Herkese açık sürüm üçü
bitince çıkar. Bu spec v0.1.1 düzeltmelerinin yapıldığını varsayar.

**İç kilometre taşları** (herkese açık sürüm değil; sürüm yalnızca üçü bitince, Erdem kuralı):
- **M1** = Plan 2 (2B arayüz; sepet, keşif rafı, kayıt akışı durum makinesi, sunucu taslağı burada kurulur).
- **M2** = grafik görüntüleyici + ölçüm (tek cevap grafiği, Liste ve 3B görünüm, yol/okuma kesimi ölçümü,
  §6 kontrolleri) + kalibrasyon kapısı (§6.1).
- **M3** = kenar kesimi kaydı (şema v2, grup kesimi, etkilenme önizlemesi, `safety_edges`), kavram grafiği,
  yönlendirme (§13).

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
Birinci dereceden doğrusallaştırmadır; B'nin kendi norm'u dışındaki normlar ve desenler **dondurulmuş**
kabul edilir (bkz. kenar ağırlığı) ve bu yüzden her şey "tahmini"dir.

- **Düğüm önemi:** `I_A = Σ_p ∇_{x_A[p]} M · x_A[p]` (gradyan × aktivasyon). `x_A`: head için
  `z_h` (o_proj girdisi dilimi), nöron için `act_n`, token için gömme vektörü. Pozitif = M'yi
  destekler. Bu, sıfır-ablation'ın doğrusal tahminidir (audit §5a `score_zero`).
- **Kenar ağırlığı (residual üzerinden sanal ağırlık × aktivasyon × gradyan):**
  `e(A→B, c) = Σ_p ⟨ γ_B ⊙ g_B^c[p] , a_A[p]/rms_B[p] − x_B[p]·⟨x_B[p], a_A[p]⟩ / (d·rms_B[p]³) ⟩`
  (`x_B`: B'nin norm girdisi, `d = d_model`). İkinci terim B'nin rms Jacobian'ıdır (kenar başına bir ek
  iç çarpım); onsuz `e` yol kesiminin türevinden ~%8 sapar, A büyük ve `x`'e paralelse daha çok.
  `g_B^c[p]`, M'nin B'nin c kanalındaki ön-aktivasyonuna göre gradyanının B'nin okuma satırlarına
  geri izdüşümüdür:
  - nöron: `W_gate[n]ᵀ·∂M/∂gate_n + W_up[n]ᵀ·∂M/∂up_n`;
  - head Q: `W_q,hᵀ·∂M/∂q_h` (RoPE öncesi q; δ kancası q_proj çıktısında);
  - head V: `W_v,gᵀ·Σ_q A_h[q,p]·∂M/∂z_h[q]` (GQA'da v grubu paylaşıldığı için head başına
    gradyan, desen `A_h` ve `∂M/∂z_h`'den analitik kurulur);
  - output: `u_y = W_U[y] − mean_v W_U[v]`, `γ_final` ve son rms ile.
  `γ` Gemma2'de `1 + w`'dir. Tek token pozisyonu kısıtları (Q aynı pozisyon, V kaynak ≤ hedef,
  MLP aynı pozisyon) toplamda kendiliğinden sağlanır. `e > 0`: bağ M'yi tahminen artırır
  ("tahmini artıran").
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

- `Q` = `γ_B ⊙ S_A` matrisinin QR ile ortonormal tabanı (tek kesimde `q = normalize(γ_B ⊙ s_A)`);
  `S_A` her kaynak için **rank 1**'dir:
  - nöron → `W_down[:, n]` (Gemma2: `γ_post ⊙ W_down[:, n]`, çünkü post-norm'dan önce);
  - token → **ortalanmış** gömme `E[token_id] − mean_v E[v]` (Gemma2'de `×√d`). Gömmeler anizotropiktir;
    ortalanmamış `E[t]` silinirse ortak ortalama yön de **her token için** silinir. Önizleme
    `cos(E[t], mean_v E[v])`'yi gösterir;
  - head → head'in **baskın yazma yönü** `u_A` = `Σ_{p≥1} a_A[p]·a_A[p]ᵀ`'nin en büyük özvektörü
    (= yığılmış `a_A`'nın baskın sağ tekil vektörü), **BOS / pozisyon 0 hariç** (attention sink normu her
    şeyi ezer), işareti `⟨u_A, mean_p a_A[p]⟩ ≥ 0` olacak şekilde. Erdem kararı 1'in rank-1 ilkesi
    korunur; normalize(ortalama) yerine bu seçilir çünkü artık uzayındaki kapsamayı (`Σ_p⟨a_A[p], u⟩² / Σ_p‖a_A[p]‖²`) en büyükleyen yön budur ve
    işaret değiştiren (kopya, önceki-token) head'lerde ortalama ≈ 0 olup gürültüdür. Gemma2'de `a_A`
    zaten post-norm sonrası artık-akım uzayındadır; `γ_post` **tekrar çarpılmaz**.
  Yön, tek cevap grafiğinde isteğin pozisyonları, kavram grafiğinde train pozitiflerinin pozisyonları
  üzerinden alınır; `u_A` tarife vektör olarak yazılır (§7), böylece kesim veriden bağımsız tekrar üretilir.
  Head kaynağında, **(A, B) çifti başına** ve B'nin okuma uzayında iki gösterge (ikisi de "tahmini"):
  - `read_coverage_B = Σ_p ⟨q, γ_B⊙a_A[p]/rms_B[p]⟩² / Σ_p ‖γ_B⊙a_A[p]/rms_B[p]‖²`;
  - `edge_share = [Σ_p ⟨g_B[p], q⟩⟨q, γ_B⊙a_A[p]⟩/rms_B[p]] / e(A→B)` ("bu kesim tahmini e'nin %K'sini
    kaldırır"). `|edge_share| < 0.5` → uyarı: "bağ tam kesilmiyor, yol kesimiyle karşılaştır".
  Tanı amaçlı "ortalama payı" (`T·‖mean_p a_A‖² / Σ_p ‖a_A[p]‖²`) head'in çoğunlukla sabit bir kayma
  yazıp yazmadığını gösterir. %100 kapsamada bile okuma kesimi ≠ yol kesimi (rms hâlâ A'yı içerir).
  Taban **orijinal** ağırlıklardan hesaplanır; aynı B'ye birden çok okuma kesimi varsa tabanlar
  birleştirilip tek QR yapılır (sıralı izdüşüm yok; audit §1 ile aynı ilke).
- `R_B`: nöron için `W_gate[n]` ve `W_up[n]` satırlarının ikisi; head Q için `W_q`'nun h'ye ait
  head_dim satırı; output için `W_U[y]` satırı. Bias'lara (Qwen2 q/k/v) dokunulmaz.
- RoPE q_proj çıktısından sonra uygulanır ve doğrusaldır; satırdan silinen bileşen RoPE sonrası
  da sıfırdır → gömme kesin. Gemma2 softcap'leri sonradan geldiği için kesinliği bozmaz.
- **Yan etki göstergeleri:** `weight_share = ‖R_B·Q‖²_F / ‖R_B‖²_F` **"ağırlık payı"** adıyla, yalnızca
  bilgi olarak (rank 1'de ≈ 1/d, neredeyse her yön için küçüktür; tek başına uyarı vermez). Başlık sayısı
  veri ağırlıklı **okuma kaybı** `read_loss = Σ_p ‖R_B·Q·Qᵀ·n_B[p]‖² / Σ_p ‖R_B·n_B[p]‖²` (`n_B`: B'nin
  norm çıktısı, saklı aktivasyonlardan; ileri geçiş yok), "bu istek(ler)de B'nin okumasının %X'i silinir".
  Uyarı `read_loss`'a bağlıdır (varsayılan eşik %20; Plan 3 görev 1'de ayarlanır). Asıl yan etki §4.4'teki
  etkilenme önizlemesinde görünür.
- Yol kesimi her pozisyonda uygulanır ve girdiden bağımsız tanımlıdır (herhangi bir istekte
  çalışır). V kanalında kanca, yalnızca h head'inin `z_h[q]` değerine `Σ_p A_h[q,p](v'_p − v_p)`
  ekler (gruptaki diğer head'ler değişmez); output kanalında yalnızca `logit_y` yeniden hesaplanır.
- Token kesimleri (iki semantikte de) `token_id`'ye bağlıdır: o token **her pozisyonda ve her
  istekte** B için kesilir; okuma kesiminde ortalanmış gömmesi benzer tokenler de kısmen etkilenir. Tek cevap
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
- **Toplam tahmini etki (başlık sayısı):** kaynak saymadan, B'nin tüm okuması üzerinden
  `ΔM_read_est = −Σ_p ⟨g_B[p], q⟩·⟨q, n_B[p]⟩` ("tahmini toplam ΔM (okuma kesimi)"), yanında hedef
  kenarın kendi kaldırılan payı `edge_share·e(A→B)`. Aradaki fark bu istekteki tahmini **yan etkidir**
  (budanmış head/nöronlar, tüm token gömmeleri ve gömme ortalaması dahil; grafik listesi bunları göremez).
- **Etkilenen okuyucular:** B; grup kesiminde gruptaki diğer N−1 head de. Her biri için `read_loss` (§4.1),
  "ağırlık payı" ve, grafikteki gelen kenarlar üzerinden, tahmini **işaretli** `ΣΔe` (mutlak değerler
  yalnızca sıralama için). Grafikte olmayan grup üyesinde `g` yoktur: "grafikte değil: yalnız ağırlık payı"
  yazar, 0 yazmaz.
- **Etkilenen kaynaklar:** grafikte B'ye (ve grup kesiminde gruptaki head'lere) gelen her kenar A' için
  kesimden sonraki tahmini ağırlık
  `e'(A'→B) = e(A'→B) − Σ_p ⟨g_B[p], q⟩·⟨q, γ_B⊙a_A'[p]⟩ / rms_B[p]`
  (birden çok kesimde `q` sütunları üzerinden toplam). Eski `⟨(γ_B⊙g_B)/rms, P_Q a_A'⟩` biçimi yalnızca
  γ sabitken doğrudur (rastgele γ ile %73 hata ölçüldü). Listeye yalnızca `|e| ≥ 0.01·max|e|` (grafik
  eşiği) olan ve kaybı `|e − e'| / |e|` ≥ %10 olanlar girer, büyükten küçüğe.
- Hepsi "tahmini · bu istek(ler)de" etiketlidir (kesim veriden bağımsızdır, her girdide etki eder; birinci
  derece, dondurulmuş norm ve desen); gerçek etki ölçümle (§6) ve Karşılaştır'la görülür.
- Önizleme sunucuda hesaplanır (`/api/graph/preview_cut`, §7) ve hızlıdır (ileri geçiş yok;
  grafikteki saklı vektörler ve `n_B` ile).

## 5. Grafik çıkarma algoritması ve maliyet

**Tek cevap ("İçine bak"):** 2B ısı haritasıyla aynı kodlama. (1) ileri+geri geçiş, δ kancaları:
o_proj girdisi, q_proj/v_proj/gate_proj/up_proj çıktıları, gömme çıktısı, norm girdileri (rms).
(2) tüm head/nöron/token için `I`. (3) budama. (4) aday kenarlar ve `e`. (5) budama, JSON.

**Kavram ("Davranış bul"dan):** hunt'ın split'i ve `ref_layer`'ı kullanılır; ileri geçiş
`ref_layer`'da durur (audit §7). Geçiş 1: train pozitiflerinden en çok `max_prompts = 16` istek
üzerinde `I` ortalaması → düğüm seçimi. Geçiş 2: aynı istekler üzerinde tutulan düğümler için `e`,
ortalama. Token düğümleri `token_id` başına toplanır. `ref_layer` sonrası katmanlar grafikte yoktur
(başlıkta yazar). Güvenlik kavramından (ör. refusal) gelen grafik `safety: true` taşır.

**Maliyet tahmini** — bunlar **tahmindir**, Plan 3'ün 1. görevi ölçüp düzeltir; arayüzde söz olarak
basılmaz. Dayanak: smoke ölçümü (Qwen2.5-0.5B fp32, 4 iş parçacığı: ~0.65 sn / istek-ileri geçişi;
çalıştırmalar arası ~2× gürültü), parametreyle ölçekleme (1.5B ≈ 3.5×, 7B ≈ 18×), T≈64. Hobi
dizüstünde (2–4 yavaş çekirdek) 2–3× daha yavaş beklenir.

| Model | Tek cevap grafiği | Kavram grafiği (16 istek, 2 geçiş) | Bir kenar ölçümü, bir semantik (ΔM: N_max 39 · KL: 10–20 çekiş × 8 istek) | fp32 RAM |
|---|---|---|---|---|
| 0.5B | ~3–10 sn | ~1.5–3 dk | ~1.5–2.5 dk | ~2.5 GB |
| 1.5B | ~15–40 sn | ~8–15 dk | ~5–8 dk | ~7.5 GB |
| 7B | ~1–3 dk, **yalnızca 30+ GB RAM ile** | CPU'da gerçekçi değil | ~25–45 dk | ~30–36 GB |

- Kenar ölçümünde ΔM katmanı çekiş başına yalnızca hedef istekte 1 ileri geçiştir (N_max 39: 0.5B'de
  ≲ 26 sn, 1.5B'de ≲ 90 sn); etkisiz kenarlar 1–3 çekişte durur (§6). Pahalı kısım betimsel KL bandıdır.
  İki semantik ≈ 2×; yol kesimi eager attention ve çift yakalama yüzünden +%20–30. (Eski tasarım: N = 20
  sabit × 17 istek = 0.5B ~7–8 dk, 1.5B ~25 dk, 7B ~1.5–2 sa.)
- CPU desteği: grafik ve ölçüm ~1.5B'ye kadar; 7B "RAM ve sabır varsa çalışır, desteklenmez" (16 GB
  makineye sığmaz). bf16'yı yerel desteklemeyen CPU'da geri geçiş ~54× yavaştır: 7B tek cevap grafiği
  ~15–30 dk olur.
- İş başlamadan önce kaba ETA gösterilir, ilk ölçülen ileri geçişten sonra **canlı ETA**'ya döner;
  kullanıcı onaylar, iş iptal edilebilir. Kabul ölçütü: 0.5B tek cevap grafiği referans makinede ≤ 10 sn.

## 6. Ölçüm

Kenar başına, seçilen semantik(ler) için:
- **Hedef etkisi ΔM:** tek cevap → kullanıcının isteğinde M'nin değişimi (n = 1, SE yok; yanında
  "tek istek: genelleme iddiası yok"); kavram → ayrılmış val pozitiflerinde `m` değişiminin ortalaması ve SE'si.
- **Genel bozulma:** nötr setin sabit (seed'li) isteklerinde, son 16 pozisyonda KL (2B ile aynı tanım,
  "genel bozulma (KL)" adıyla). Hızlı profil 8 istek (varsayılan), tam profil 20; hedef ve kontroller
  **aynı** istek setinde (eşli).
- **Kontrol kenarları:** kontrol kenarı hedefle aynı tür, kanal ve semantikle kesilir. **Birincil**
  kontrol **grafik katmanı**dır: grafikte tutulan kenarlardan aynı (tür, kanal) ve katman eşli olanlar
  (yalnızca bu katman ≥ 39 üyeliyse); değilse **katman eşli düzgün havuz**: A' A'nın katmanından ve
  türünden, B' B'nin katmanından ve türünden (grafikte görünmeyenler dahil; modelin yapısından sayılır).
  İkisi birlikte koşulmaz (maliyet 2× olur); sonuç hangisinin kullanıldığını yazar. Ölçülen kenarın kendisi
  hariç; token kaynağı için başka bir pozisyon (tek cevap; **aynı `token_id`'yi taşıyan pozisyonlar hariç**,
  çünkü kesim `token_id`'ye bağlıdır) ya da istek kümesinden başka bir token kimliği (kavram). Head
  kaynaklı okuma kontrollerinde `u_A'` hedefle aynı kestiriciyle (BOS hariç baskın tekil vektör) bulunur.
  Yerine koyarak çekme yoktur. Bedava ek bilgi: hedefin `‖a_A‖`'sının havuz içindeki yüzdeliği.
- **Çekiş sayısı (Erdem kararı 5, uzlaşıyla güncellendi):** `N_max` varsayılan **39**, en az **19 yalnızca
  m = 1** (toplu ölçümde tek kenar) iken, m ≥ 2'de en az 39; CPU tavanı **199** (kaydırıcı en çok
  `min(havuz, 199)`; GPU'da "ileri" seçenekle daha fazlası). Holm altında en küçük p `0.05/m`'yi geçmeli ve
  `p_min = 1/(N_max+1)` olduğundan `N_max ≥ 20m − 1` gerekir (m = 2 → 39, m = 5 → 99, m = 8 → 159);
  kaydırıcı `p_min`'i ve bu `N_max`'ın desteklediği m'yi gösterir, yetmiyorsa önerir. Havuz ≤ tavan ise
  "havuzun tamamı" seçilebilir ve sonuç **"kesin sıra (havuzun tamamı)"** diye etiketlenir. Havuz < 19
  → tamamı kullanılır, p kararı gösterilmez: "kontrol havuzu küçük (n = k): anlamlılık gösterilemez".
  `N_max` ölçümden **önce** seçilir; sonucu görüp N artırmak yoktur.
- **Sıralı (Besag–Clifford) permütasyon testi (2B ile ortak fonksiyon, `metrics.control_stats`):** kontroller
  tek tek çekilir; `k` = `|ΔM_rand| ≥ |ΔM|` aşım sayısı. Karar tam `N_max` kuralıyla aynıdır:
  `p = (1 + k)/(1 + N_max) ≤ τ`. **Erken durma (yararsızlık):** `k`, eşiği (en gevşek Holm eşiği 0.05)
  artık sağlayamayacak kadar büyüdüğü anda çekiş durur ve sonuç **"p > τ, n çekişte durdu (k aşım)"**
  der (ya da Besag–Clifford `p = k/n`); asla yeniden hesaplanmış `(1+k)/(1+n)` gösterilmez. Başarı tam
  `N_max` ister; etkisiz kenar çoğunlukla 1–3 çekişte durur. Yüzdelik ve rastgele dağılımın %5–95 bandı
  ("yaklaşık") gösterilir. Oran ("×2.6") gösterilmez.
- **Çoklu karşılaştırma:** Holm **ölçüm toplu başına** (bir `/measure` isteği, ≤ 8 kenar); ham ve düzeltilmiş
  p yan yana ("bu toplu ölçümde m bağ için düzeltilmiş"). Oturum boyunca düzeltme yapılmaz (sabit aile yok);
  bunun yerine her p'nin yanında sayaç **"bu oturumda k bağ ölçüldü"** ve gezinerek seçilmiş her sonuçta
  **"hipotez"** etiketi. Seed grafik başına **sabittir** (sessiz yeniden çekiş yok); her ölçüm kaydedilir.
  Aynı seed'le aynı katmandaki kenarlar aynı kontrolleri paylaşır, p'leri ilişkilidir (belgelenir).
- **KL kontrolü yalnızca betimseldir:** 10–20 çekiş, hızlı profilin 8 isteğinde; yüzdelik ve %5–95 bandı,
  **p yok, Holm yok** ("hedef rastgelelerden daha mı bozucu" farklı bir sorudur).
- **Sonuç cümlesi:** `p ≤ τ` → **"bu bağ rastgele bir bağdan daha etkili"** ("davranışın nedeni" değil);
  aksi halde "belirsiz: rastgele bağlardan ayırt edilemiyor".
- **2B için:** `real_effect` sabit 30 çekiş yerine aynı sıralı teste geçer (`N_max` 39; maliyet aynı ya da az).
- **Kapı:** "Kes (geçici)" yalnızca ilgili semantik ölçüldükten sonra açılır (2B ile aynı UI kapısı; sonuç
  belirsiz olsa da açılır, sonuç düz cümleyle yanında yazar).
- Tahmin ile ölçüm her zaman yan yana: "tahmini e" ve "ölçülen ΔM" ayrı sütunlar.

### 6.1 Kalibrasyon kapısı (3B herkese gösterilmeden önce, go/no-go)

M2'nin sonunda, 2 gerçek modelde (Qwen2.5-0.5B, SmolLM2) model başına ~100 kenar, |e| ondalıklarına ve
kanallara (Q / V / resid / logit) **tabakalı** seçilir (yalnızca en güçlüler değil). Gerçek yol kesimi ölçülür
(`e`'nin tahmin ettiği şey); okuma kesimi ayrıca `edge_share·e` ile karşılaştırılır. Geçme ölçütleri
(2 model birlikte, kanal başına denetlenir):
- Spearman `ρ(e, ΔM_path) ≥ 0.5`;
- `|ΔM|`'si rastgele kontrolün p95'ini aşan kenarlarda işaret uyumu ≥ %80;
- en üst ondalık |e| kenarlarının ≥ %60'ı kontrolünü `p ≤ 0.05` ile yener.
Geçemeyen kanalın `e`'si yalnızca **sıra** ("ipucu") olarak gösterilir, büyüklük yok. Saçılım grafiği ve ρ
değerleri sonuç ne olursa olsun belgelerde ve sürüm notlarında yayımlanır.

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
          control_pool_size, control_kind: "graph"|"uniform",
          read: {available, bakeable, scope: "head"|"kv_group"|"neuron"|"logit", weight_share} | null,
          ops: {path: bool, read: bool}}],
 rivers: {trunk: [float] (katman başına logit-lens / kavram izdüşümü), norms: [float],
          branches: [{token, proj: [float]}]},   # §8; branches isteğe bağlı, seçilen token için
 pruning: {scored_nodes, kept_nodes, candidate_edges, kept_edges, edge_threshold}}
```
`graph_id` oturuma bağlıdır; model değişince geçersizdir (404 "grafik eskidi, yeniden çiz").
Sunucu en son 4 grafiği tutar.
- **Boyut:** grafik JSON'u kenar başına tam op ve `vector_b64` **taşımaz** (d=4096'da head vektörü ~22 KB;
  500 kenarda ~11 MB olurdu). Kenarlar düğümlere kimlikle başvurur; `u_A` sunucuda düğüm başına bir kez
  saklanır. Tam op (head kaynağında `vector_b64` ile) "Kes (geçici)" anında
  `GET /api/graph/{graph_id}/edge/{edge_id}/op?semantics=path|read` ile alınır ve sepete o girer.

**`GET /api/graph/{graph_id}/node/{node_id}`** → ayrıntı: tür, katman, `score`; head için son
pozisyondan en çok bakılan 5 kaynak token ve OV yazma uzayının logit-lens ilk 10 artırdığı /
bastırdığı token; nöron için istek(ler)de en etkin olduğu tokenler ve `W_U·(γ_f⊙W_down[:,n])`
ilk 10 artırılan/bastırılan token. Hepsi "tahmini (doğrudan etki)" etiketli.

**`POST /api/graph/measure`** → `{job_id}`. İstek: `{graph_id, edge_ids: [str] (1..8),
semantics: ["path","read"] | null (null = kullanılabilen hepsi), n_max: int | null (null = 39;
m = len(edge_ids) = 1 iken 19 ≤ n_max, m ≥ 2 iken 39 ≤ n_max; n_max ≤ min(pool_size, 199) CPU'da;
aksi 422), neutral_profile: "quick" | "full" = "quick"}`. Seed istekte yoktur: grafik başına sabittir.
Grafik sonucu her kenar için `control_pool_size` ve `control_kind` taşır.
Sonuç: `{session_measured_count, edges: [{edge_id, results: [{semantics, delta_metric,
delta_metric_se | null, kl_neutral, kl_neutral_se, n_target, n_neutral,
control: {kind: "graph"|"uniform", n_max, n_drawn, exceed, pool_size, exhaustive, stopped_early,
delta_band: [p5, p95], percentile, p_value | null, p_holm | null, norm_percentile},
kl_control: {n, band: [p5, p95], percentile}, verdict: "etkili"|"belirsiz"|"gösterilemez",
hypothesis: true}]}]}`. Kenarlar tek tek kesilip ölçülür (ortak etki Karşılaştır'da); Holm toplu içindedir.

**`POST /api/graph/preview_cut`** (iş değil, senkron, ≤ 1 sn): `{graph_id, edge_id}` →
`{scope: "head"|"kv_group"|"neuron"|"logit", delta_m_read_est, edge_removed, read_coverage | null,
edge_share | null, mean_share | null, token_mean_cos | null,
readers: [{node_id, in_graph, read_loss | null, weight_share, sum_delta_e | null}],
sources: [{edge_id, node_id, e, e_after, loss_frac}]}` (§4.4).

**Tarif şeması (strict, yol içermez):**
```
SrcRef = {kind:"token", token_id} | {kind:"neuron", layer, index}
       | {kind:"head", layer, head, d_model, vector_b64}   # u_A (§4.1), float32 LE base64 (DirectionOp ile aynı denetim)
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
  koşulacak (≈ {ETA}). Devam?" Onaylanırsa karşılaştırma ve test seti **yeniden koşulur**; bu sırada
  kesimler "atılacak" (üstü çizili) olarak sepette kalır, iş iptal olur ya da başarısız olursa hiçbir şey
  kaybolmaz; başarıda keşif rafına taşınır (taslağın parçası, geri alınabilir). **İkinci onay** (Erdem
  cevabı E) ayrı bir pencere değildir: kullanıcı yenilenmiş **Karşılaştır** ekranına "Sayılar yenilendi;
  N kesim atıldı" bandıyla iner ve olağan **"Modeli kaydet"** tıklaması ikinci onaydır. Hiçbir şey
  atılmadıysa tek onay olağan Kaydet tıklamasıdır. Ekrandaki sayılar her zaman kaydedilecek modelin
  sayılarıdır. Sepette yalnızca keşif kesimi varsa pencere "kaydedilecek kesim kalmıyor" der ve kayıt
  başlamaz. Sepet sürekli "Kaydedilecek: n · Yalnızca keşif: k" satırını gösterir, böylece pencere sürpriz değildir.
- **Tarif vektörleri:** bu oturumda üretilmemiş her head kaynağı vektörü (`vector_b64`) içe aktarmada,
  kartta ve tarif özetinde **"doğrulanmamış yön"** (dosyadan gelen) etiketi taşır. Oturumda önbellekte bir
  refusal yönü varsa her saklı vektörün ona kosinüsü denetlenir; eşiği aşan vektör güvenlik etiketini
  zorunlu kılar (`_is_safety` genişler). Önbellek yoksa içe aktarmada dakikalar süren bir hunt zorlanmaz;
  etiket "en iyi çaba"dır (README de söyler).
- Model kartı her kenar kesimini semantiğiyle yazar: "okuma kesimi: K3 nöron 1203 → K7 head 5 (Q).
  B, bu nöronun yazdığı yönü hiçbir kaynaktan okumaz."
- Güvenlik işareti: `safety: true` grafikten gelen her kenarın anahtarı (`src`, `dst`) oturumda
  `session.safety_edges`'e eklenir. Kayıtta tarifte bu kümeden bir `edge_read` varsa
  `abliterated: refusal` + "safety-ablated" zorunludur; istemci kaldıramaz (`_is_safety` genişler).

## 8. 3B arayüz

**Kütüphane: @react-three/fiber** + `OrbitControls` için **three-stdlib** (R3F `extend` ile), npm ile
sürümü kilitli, Vite ile paketlenir; çalışma anında CDN yok. **drei kullanılmaz**: `Text` (blob worker +
CDN font), `Environment`, `useGLTF` (CDN Draco) ve `useDetectGPU` (unpkg'den veri) CSP'yi ve "CDN yok"
kuralını bozar; etiketler DOM katmanıyla çizilir. Gerekçe: arayüz zaten React; seçim, sepet ve ölçüm
durumu 2B ile aynı Zustand store'larından okunur (`<Canvas>` ayrı bir React köküdür, Context geçmez),
ısı haritası ile eşleme bildirimseldir; düğüm kimlikleri kanonik (`h:L:H`, `n:L:i`, `t:p`, `tid:i`, `out`).
Ağır işler React dışında: düğümler tür başına bir `InstancedMesh`, kenarlar tek `LineSegments2`
(köşe başına renk/kalınlık). 3B modu tembel yüklenen ayrı bir parçadır (dynamic import); 2B
kullanıcısı three.js indirmez. `postprocessing` yalnızca "Yüksek" kademede, 3B parçası içinde dinamik yüklenir.

**CSP uyumu** (sunucunun CSP'si: `default-src 'self'`, `connect-src 'self'`, `img-src 'self' data:`):
satır içi `<script>`, `eval` / `new Function`, `blob:` / satır içi worker (`?worker&inline`) ve WASM
(Draco, KTX2) yok; fontlar `data:` URI olarak gömülmez (`build.assetsInlineLimit: 0`). Sunucu
`script-src 'self'; worker-src 'self'; font-src 'self'`'i açıkça yazar; test başlığı ve 127.0.0.1 dışı
istek olmadığını denetler. **Fontlar** pakettedir (`@fontsource-variable`, latin + **latin-ext**: Ğ ğ İ ı Ş ş);
Google Fonts ya da başka dış istek yok.

**Yerleşim:** y ekseni hesaplama sırası. Tokenler y=0'da cümle sırasıyla; katman ℓ'nin attention
düzlemi y=2ℓ+1, MLP düzlemi y=2ℓ+2; output en üstte. Head'ler düzlemde head numarasına göre
(ısı haritasının sütunlarıyla aynı sıra), nöronlar skora göre. Düzlemler soluk dikdörtgen,
etiket "K12 · attn" / "K12 · MLP".

**Görsel dil (fütüristik, Erdem kararı 6):** zemin `#0B0E13`, panel `#121722` yarı saydam cam
(backdrop blur), çizgi `#222A38`, metin `#E6EAF2` / `#A7B0C0`; başlıklar Space Grotesk, metin
Inter, sayılar JetBrains Mono. Artıran kenar `#2EE6C5`, bastıran `#9FC3FF`; kalınlık ve opaklık
|e| ile; düğümler ve güçlü kenarlar hafif ışıma, seçili kenarda akış yönünde ilerleyen ışık
parçacıkları (`LineMaterial` `dashOffset` animasyonu; seçili kenar + üzerine gelinen düğümün kenarları).
Turuncu `#FF7A45` yalnızca sepetteki (kesik çizgili) kenarlar içindir. Rozetler: `kaydedilebilir` teal
çerçeve, `kaydedilebilir · grup` teal + grup simgesi, `yalnızca keşif` amber `#FFD27A`, `kesilemez` gri.
Grafik başlığında "tahmini" etiketi her zaman görünür. Işıma ve parçacıklar `prefers-reduced-motion`'da
ve "Düşük güç" kademesinde kapanır; anlam hiçbir zaman yalnızca ışımaya bağlı değildir.

**Işıma ve kalite kademeleri (Erdem cevabı C):** varsayılan ışıma **sahte toplamalı ışıma**dır: düğüm başına
instanced, toplamalı karışımlı hale dörtgeni (doku tuvalde üretilir, ağ yok) ve seçili/güçlü kenarın altında
daha geniş, soluk ikinci bir çizgi. MSAA açık kalır. Gerçek bloom yalnızca isteğe bağlı "Yüksek" kademede.
`<Canvas flat>` ve ışıyan malzemelerde `toneMapped={false}` (ACES ton eşleme teal ve maviyi soldurur);
renkler CSS token'larından `readTokens()` ile okunur, 2B ve 3B tek kaynaktan.

| Kademe | Işıma | Nehir | Parçacık | DPR tavanı | Paneller |
|---|---|---|---|---|---|
| Yüksek (isteğe bağlı) | bloom | seçimde akış animasyonu | evet | 2 | blur |
| Dengeli (varsayılan) | sahte toplamalı | statik gradyan, kısa akış | yalnızca seçili kenar | 1.5 | yörüngede blur kapalı |
| Düşük güç | yok | düz şerit ya da kapalı | yok | 1 | düz renk |

**Kalite yöneticisi:** etkileşimin ilk 3 sn'sinde ortanca kare süresi ölçülür; ~24 ms üstü bir kademe
düşürür; seçim sunucu tercihlerine yazılır (aşağıda). El yazımı (~20 satır); drei `PerformanceMonitor` /
`detect-gpu` kullanılmaz (ağdan veri çeker). Animasyonlar tek bir `useAnimationClock` üzerinden yalnızca
bir şey oynarken `invalidate()` çağırır; 5 sn'den uzun her animasyon için başlıkta "Animasyonları durdur"
anahtarı (WCAG 2.2.2). Cam paneller (`backdrop-filter`) tuvalle çakışırken yörünge sırasında düz renge döner.

**Nehir (artık akım, Erdem cevabı B):** düğümler pozisyonlar üzerinden toplandığı için pozisyon başına
nehir yoktur. Varsayılan **tek gövde nehri**: M'nin okunduğu pozisyonun (tek cevapta son token) artık akımı,
alttan üste akan yarı saydam bir şerit. Şerit **genişliği** = katman başına **logit-lens yakınlığı**: o
katmandaki artık akımın cevap yönüne izdüşümü (tek cevapta `γ_final` ve son rms ile `⟨resid_ℓ, u_y⟩`,
kavramda `resid_ℓ · r_ref`), yani M'nin katman katman nasıl biriktiği. Ham `‖resid‖` isteğe bağlı ikinci
kanaldır (BOS'un dev normu yüzünden genişlik değildir). Head ve nöronlar gövdeye, doğrudan katkıları
`|e(düğüm → out)|` ile orantılı kollarla bağlanır (grafikte zaten olan ağırlıklar). Kullanıcı bir token
seçerse o pozisyon (kavramda token kimliği) gövdeden ayrılan **kendi kolunu** açar (aynı izdüşüm, o pozisyonda).
Tümü tek mesh, özel shader (katman başına genişlik dizisi), tek çizim çağrısı. "Nehir" açılıp kapatılabilir
(varsayılan açık); akış animasyonu kalite kademesine bağlıdır. Grafik JSON'unda `rivers` (§7).

**2B → 3B geçiş (Erdem cevabı C):** "2B ısı haritası | 3B bağlar | Liste" anahtarında 3B seçilince grafik
bir **iştir**; iş bitene kadar ısı haritası ilerleme kartı ve ETA ile kalır. Uçuş **yalnızca grafik sonucu
geldikten sonra** oynar: ısı haritası ızgarası (katman × head) yatırılır ve tutulan head'lerin hücreleri
3B'deki düğümlerin yerine uçar (~800 ms); tutulmayan hücreler yerinde söner ve küçülür. Hücreler ısı
haritası renginde (çıkış gücü, [0,1]) kalkar, indikten sonra işaretli `I` kodlamasına (teal / mavi) **renk
geçişiyle** döner. Ardından nehir ve kenarlar belirir. Geri dönüş tersidir. Uygulama: hücreler tek
`InstancedMesh`, başlangıçlar ısı haritasının `CellLayout.cellRect()`'inden, ara değer vertex shader'da tek
`uTransition` ile. `prefers-reduced-motion` ya da "Animasyonları durdur" açıkken geçiş **anında** olur.

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
  "Gerçek etkiyi ölç · ≈ {ETA} dk" (`N_max` kaydırıcısıyla, §6) → sonuç satırları (§4.3) →
  "Kes (geçici) · okuma kesimi" ve/veya "Kes (geçici) · yol kesimi (yalnızca keşif)". Tek sepet vardır.
- **Kes ≠ Kaydet** (Erdem kararı 2, cevap F): düğme etiketi **"Kes (geçici)"**, alt satırı **"Dosya
  değişmez"**. Yalnızca oturumdaki (kancalı) modeli değiştirir ve sepete ekler (sunucu kesimler için
  durumsuzdur; sepet tek doğruluk kaynağıdır). Diske yazma yalnızca ayrı **"Modeli kaydet (n kesim)"**
  düğmesiyle olur (§7 onay akışı). Üst çubukta "Kaydedilmemiş N kesim" göstergesi.
- **Taslak (uzlaşı):** port her açılışta rastgele olduğu için tarayıcı kökeni değişir ve `localStorage`
  yeniden açılışta bulunamaz. Bu yüzden sepet taslağı ve tercihler **sunucuda**, model SHA-256'sına göre
  tutulur: `GET/PUT/DELETE /api/draft/{sha256}`, `GET/PUT /api/prefs` (`~/Neshter/.state/`, atomik
  yazım, token korumalı, katı şema, boyut tavanı). Taslak değişiklikte (500 ms gecikmeli) ve
  `visibilitychange: hidden`'da yazılır, kapanışta değil. `/api/session` `source_sha256` döner; geri yükleme
  önerisi model yüklenip SHA bilinince "Önceki oturumdan kaydedilmemiş N kesim var: geri yükle?" diye
  gelir, `recipe/check`'ten geçer. `localStorage` en çok aynı sekme yenilemesi için önbellektir. **Token**
  yalnızca bellekte ve `sessionStorage`'da (URL'den `replaceState` ile silinir), asla `localStorage`'da;
  `source_path` saklanmaz. Model çıktısı metni taslağa yazılmaz.
- **Ayrılma uyarısı:** sepette kaydedilmemiş kesim varken model değiştirme ya da uygulamadaki "Kapat"
  (gerçek, token korumalı `POST /api/shutdown`) → uygulama içi pencere "Ayrılıyorsunuz, kesimlerinizi daha
  kaydetmediniz. Kaydet / Kaydetmeden çık / Vazgeç". Sekme kapatma ve yenilemede `beforeunload` yalnızca
  kirli durumda kaydedilir (tarayıcılar özel metne izin vermez). **Ctrl+C (Erdem cevabı D):** sunucu
  kapanışı engellemez; taslak zaten sunucuda olduğundan terminale tek satır basar: "N kaydedilmemiş kesim
  taslağa yazıldı; sonraki açılışta geri yüklenir." Bu davranış 2B sepeti için de geçerlidir (Plan 2 / M1).
- "2B ısı haritası | 3B bağlar | Liste" anahtarı aynı ekranda; seçim ortaktır: head seçimi iki
  modda da aynı hücreyi işaretler, nöron seçimi 2B'deki "MLP · en güçlü" listesinde görünür.

**Sınırlar:** ≤ 600 düğüm, ≤ 2000 kenar (sunucu sınırlarıyla aynı). `frameloop="demand"`
(yalnızca etkileşimde ya da bir animasyon oynarken çizim). Hedef: 300 düğüm / 500 kenarda tümleşik GPU'da
60 fps (Plan 3 görev 1'de gerçek bir Intel iGPU'da ölçülür). Etiketler
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
- Head kaynağında `|edge_share| < 0.5` → sarı uyarı "bağ tam kesilmiyor".
- `read_loss` eşiği aşarsa (§4.1) → sarı uyarı. Bağlı lm_head + output okuma kesimi → kayıt diskinde artış.
- Ölçüm iptal → kısmi sonuç saklanmaz. Kontrol havuzu < 19 → havuzun tamamı, sonuçta "n = k, anlamlılık
  gösterilemez". Grafik katmanı < 39 üyeliyse düzgün havuza düşülür (§6).
- Sunucu yeniden başlatılmış (eski token, her çağrı 403) → tek ekran "Oturum bitti, sunucu yeniden
  başlatılmış"; taslak varsa "kesimlerin taslakta".
- WebGL2 yok, `webglcontextlost` ya da yazılım işleyici (SwiftShader / llvmpipe) → otomatik Liste,
  "Yine de 3B'yi dene" seçeneğiyle.
- `.nester.json` v2'yi eski Neşter açarsa katı şema reddeder (beklenen); bilinmeyen alan → 422.
- Eager attention yalnızca grafik ve V yol kesimi ölçümü sırasında açılır, sonra geri alınır.

## 10. Test yaklaşımı

Küçük rastgele llama, qwen2 ve gemma2 modelleri (CI'da zaten kuruluyor), CPU fp32:
1. **Kanca == bake:** her kaydedilebilir tür için (token→head Q, token→neuron, head→head Q,
   head→neuron, neuron→head Q, neuron→neuron, düğüm→logit bağımsız ve bağlı lm_head) kanca ile
   gömülü modelin logitleri `≤ 1e-5`; bf16 kayıt → yeniden yükleme → kancalı bf16 ile aynı token
   dizisi (mevcut head testleri gibi). Gemma2'de head→head Q durumu ayrıca: head kaynağında `γ_post`
   tekrar çarpılmaz, nöron kaynağında çarpılır. Token kaynağında `S_A` ortalanmış gömmedir.
2. **Okuma kesimi B'nin okumasını sıfırlar:** `‖R_B'·(γ_B⊙S_A)‖ / ‖R_B·(γ_B⊙S_A)‖ ≤ 1e-6`; norm
   çıktısına `S_A` uzayında rastgele vektör eklenince B'nin ön-aktivasyonu değişmez; B dışındaki
   satırlar bit düzeyinde aynı.
3. **Yol kesimi doğru ölçülür:** kanca, iki geçişli bağımsız bir referansla (önce `a_A` ve `x`
   yakalanır, sonra B'nin kanal girdisi elle `norm_B(x − a_A)` ile yerleştirilir) `≤ 1e-5`
   eşleşir; aynı katmandaki diğer head'ler/nöronlar ve V'de gruptaki diğer head'ler değişmez.
4. **Tahmin tutarlılığı:** `a_A`'yı `ε` ile ölçekleyen yol kesiminin sonlu farkı (merkezi, iki `ε`)
   rms terimli `e`'ye (§3) %2 içinde (hunt'taki FD testi gibi). rms terimi olmayan eski `e` bu testi
   geçemezdi (~%8 sapma); terimi düşüren bir değişiklik testi kırar. Bu test doğrusallaştırmanın
   tutarlılığını doğrular, gerçek ablation'la uyumu değil (o, §6.1 kalibrasyonudur).
5. **Şema ve kayıt:** `edge_path` export'ta 422, `bake()`'te hata; v1 tarif yüklenir; v2'de
   bilinmeyen alan, aralık dışı katman/token, yanlış boyutlu head vektörü reddedilir.
   Grup kesimi (`channel:"v"`) kanca == bake ve gruptaki tüm head'lerin değiştiği, grup dışı
   head'lerin bit düzeyinde aynı kaldığı test edilir.
6. **Baskın yön ve önizleme:** head kaynaklı okuma kesimi yalnızca `u_A` bileşenini siler (rank 1);
   `u_A` BOS hariç baskın tekil vektördür ve işareti `⟨u_A, mean a⟩ ≥ 0`; `read_coverage_B`,
   `edge_share`, `read_loss`, `ΔM_read_est` bilinen girdilerle. Önizleme `e_after` değerleri dondurulmuş
   norm ve dondurulmuş desenli bir referansa `≤ 1e-5` eşit; γ sabit olmayan bir durumda eski formülden
   ayrıldığı da test edilir. Gerçek kancayla yalnızca **işaret** uyumu denetlenir (softmax/SiLU doğrusal değil).
7. **Kontrol havuzu ve çekiş:** `pool_size` modelin yapısından doğru sayılır; aynı `token_id`'li
   pozisyonlar dışlanır; grafik katmanı ≥ 39 değilse düzgün havuz; `n_max` sınırları (m = 1 → 19,
   m ≥ 2 → 39, ≤ min(havuz, 199)) dışı 422; havuz < 19 iken havuzun tamamı ve p kararı yok.
8. **Güvenlik:** `safety: true` grafikten kenar içeren kayıt, istemci `concept=None` dese de
   etiketlenir.
9. **İstatistik:** tüm kontrol kenarları katman ve tür eşli; `p` formülü ve yüzdelik bilinen
   girdilerle. Sıralı test: erken durmalı karar her dizide tam `N_max` kararıyla aynı; yararsızlıkta
   "p > τ, n çekişte durdu" döner, `(1+k)/(1+n)` dönmez. Holm toplu içinde (`metrics.holm`); seed sabitken
   iki ölçüm aynı kontrolleri çeker; KL kontrolünde p alanı yoktur; oturum sayacı artar.
10. **Arayüz:** rozet ve kapı mantığı (ölçülmeden "Kes (geçici)" kapalı; keşif kesimiyle Kaydet → onay →
   "atılacak" işaretli yeniden karşılaştırma → iptalde hiçbir şey kaybolmaz → yenilenmiş Karşılaştır +
   olağan Kaydet tıklaması, ikinci pencere yok), ayrılma uyarısı; sunucu öldürülüp yeniden açılınca (yeni
   port) aynı SHA'da taslağın geri gelmesi; Ctrl+C'de terminal satırı; önyüklemeden sonra hiçbir
   `localStorage` değerinde token yok; CSP ihlali ya da 127.0.0.1 dışı istek yok (fontlar dahil); etkilenme
   önizlemesi vurgusu; 2B→3B geçişi yalnızca grafik işi bittikten sonra (reduced-motion'da anında); kalite
   kademesi düşüşü; WebGL yokken Liste; klavye gezinmesi; sabit grafik JSON'uyla Playwright smoke.
11. **Grafik JSON boyutu:** 500 kenarlı sabit grafikte JSON'da `vector_b64` geçmez; edge op uç noktası head
   kaynağında vektörü döner.

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
norm eşli kontrol (bedava `‖a_A‖` yüzdeliği dışında) ve özgüllük oranı (audit §4 v0.2); gradyan
ağırlıklı `u_A` (A→B kenarından en çok payı kaldıran yön); pozisyon başına nehirler; yönlendirme
head'inin modelin kendi attention'ından geçen gerçek head olarak kurulması (§13 aşama 2); grafiği
resim/dosya olarak paylaşma; WebGPU, VR.

## 13. Yönlendirme: sabit vektör, sonra yapay head — Erdem'in isteği

**İstek:** modele yapay bir head ekleyip onu bir şeyle ilişkilendirmek; model bu yüzden yapay olarak
"halüsinasyon" yapabilir, ama istenen davranış da böyle ortaya çıkarılabilir.

**Dürüst çerçeve (arayüzde de yazar):** Ağırlıkları sıfır olan boş bir head akışa hiçbir şey yazmaz.
Model onu kullanmayı hiç öğrenmediği için kesilen bilgi (ör. "hayır" cevapları) kendiliğinden oraya
**kaçmaz**. Yapay head'in bir etkisi olması için ağırlıklarını biz yazarız: **ne zaman** çalışacağını
(QK: hangi tetikleyiciye bakacağı) ve **ne yazacağını** (OV: hangi yönü itecek). Bu, bilinen
**yönlendirme** (steering) tekniğinin bir head biçimidir. Modeli bir davranışa zorlar ve gerçekten
uydurma cevaplara yol açabilir. Neşter bunu "keşif ve deney" aracı olarak sunar, ölçmeden kaydetmez.
Arayüz etiketi: "yönlendirme (tetiklemeli, head biçiminde): model bunu öğrenmedi; yazdığı yönü biz
seçtik". Çıktı değişirse bu kaldıracın çalıştığını gösterir, modelin bir şeyi "bildiğini" değil; araçta
doğruluk/halüsinasyon ölçüsü yoktur, yönlendirilmiş model için "güvenli" ya da "daha iyi" denmez.

**Sıra (Erdem cevabı A):** önce sabit vektör temel çizgidir; head ancak onu yenerse sunulur; gerçek head
sonraki, mimari başına bir projedir.

**Aşama 0 — Sabit yönlendirme vektörü (M3, temel çizgi; kanca, yalnızca keşif):**
- Katman ℓ'nin artık akımına her pozisyonda (BOS hariç) `α·ŵ` eklenir (ActAdd/CAA biçimi; tek parametre).
- Hedef `ŵ` (birim): bir kavram yönü (+ ya da −) ya da bir tokenin çıkış satırı `W_U[y]` (o tokeni öne
  iter). `α`, nötr sette katman ℓ'nin ortanca `‖resid_ℓ‖`'sinin katı olarak verilir (katmanlar ve modeller
  arasında taşınabilir birim).
- Llama/Qwen2 o_proj'unda bias olmadığı için ağırlığa gömülemez: rozet `yalnızca keşif`.

**Aşama 1 — Yapay yönlendirme head'i (kanca, yalnızca keşif; aşama 0'ı yenerse sunulur):**
- Katman ℓ'de, mevcut head'lere ek tek bir head. Tetikleyici `t̂` (birim): bir kavram yönü (hunt sonucu)
  ya da bir tokenin (ortalanmış) gömmesi. Hedef `ŵ` aşama 0'daki gibi. Güç `α` (aynı birim) ve keskinlik
  `β` (nötr sette tetikleyici std'si başına).
- Her pozisyon q için skorlar `s[q,p] = β·⟨n_q, t̂⟩·⟨n_p, t̂⟩` (n: ℓ'nin attention norm çıktısı, nedensel
  maske, **BOS anahtarlardan hariç**) ve skoru 0 olan bir **boş anahtar** (sink yuvası, değeri 0);
  `z[q] = Σ_p softmax(s[q,·])_p · relu(⟨n_p, t̂⟩)`, yazılan `α · z[q] · ŵ`. Boş anahtar ve relu sayesinde
  tetikleyici yoksa yazı ≈ 0'dır; yalnızca tetikleyici geçmişte göründükçe head hedef yönü akışa yazar.
- **Sunulma kapısı:** ölçüm paneli aynı katmanda, **eşit ortalama yazma normlu** sabit vektörü (aşama 0)
  yan yana gösterir. Head yalnızca eşit ya da daha düşük nötr KL'de daha çok hedef etkisi verirse "değer"
  diye işaretlenir; vermezse panel "sabit vektör yeterli" der ve head sepete önerilmez.
- 3B'de ayrı renkte (mor) bir düğüm olarak katmanına eklenir (sabit vektör, gövde nehrine giren tek kol);
  kenarları ve nehre girişi görünür.

**Ölçüm (iki aşamada da):** diğer kesimlerle aynı kapı ve §6 istatistiği (sıralı test, `N_max`, Holm, KL
betimsel). Hedef etkisi **çıktılarda** ölçülür (tek cevapta ΔM, kavramda val'de logit farkı / test seti
doğruluğu); kavramda `m` yalnızca "probe değeri" olarak gösterilir. **Döngüsellik işareti:** `ŵ` ölçülen
kavram yönüyle aynıysa ya da `cos(ŵ, r_ref) > 0.5` ve `ℓ < ref_layer` ise sonuç "döngüsel: ölçülen, yazılan
yönün kendisi" diye işaretlenir ve `m` üzerinden karar verilmez. Kontrol: aynı `α` (head'de aynı `β`, aynı
tetikleyici) ile **norm eşli rastgele hedef yön**, aynı türden (hedef token satırıysa rastgele `W_U[y']`).

**Kapsam:**
- **M3 (Plan 3):** aşama 0 ve aşama 1, yalnızca keşif, kancayla. Rozet `yalnızca keşif`. Sepete girer,
  karşılaştırılabilir; kayıtta §7 onay akışıyla atılır.
- **Aşama 2 — Gerçek head ve kaydetme (sonraki proje, mimari başına):** aşama 1'in tanımı gerçek bir
  Llama/Qwen2/Gemma2 head'inin hesapladığı şey değildir (RoPE rank-1 q·k'yi döndürür; Gemma2'de sorgu
  ölçeği, softcap 50, kayan pencere ve birleştirilmiş çıktının post-norm'u). Kaydedilebilir olması için head
  **modelin kendi attention fonksiyonundan** geçen gerçek ağırlıklarla kurulur (`W_Q, W_K` rank-1 satırları
  en düşük frekanslı RoPE çiftinde, `W_V` satırı `t̂`, `W_O` sütunu `α·ŵ`); o zaman "kanca == bake" test
  edilebilir değişmezdir. Head sayısını değiştirmek tensör şekillerini değiştirir (GQA'da ayrıca K/V grubu
  gerekir); transformers ve llama.cpp uyumu mimari başına doğrulanmadan kayıt açılmaz. Alternatif olarak
  "head nakli": önemi ölçülerek ~0 bulunan bir head'in ağırlıkları yapay head ile değiştirilir (şekil
  değişmez). İkisi de ayrı bir spec'te tasarlanır; Plan 3'e girmez.
- **Aşama 3 — "Yapay head'i eğit" (aşama 2'den sonra):** yalnızca yeni head'in ağırlıkları eğitilir, model
  donuk kalır. Başlangıç: OV yazma tarafı sıfır, QK/V okuma tarafı küçük rastgele (tamamen sıfır head'in
  gradyanı da sıfırdır, hiç öğrenemez; bu başlangıçla ilk adımda ağı bozmaz ama öğrenebilir: ilk adımda
  yalnız `W_O` öğrenir, sonra hepsi; LoRA'daki B = 0 deseni). Daha iyisi aşama 2'nin elle kurulmuş head'inden
  küçük `α` ile başlamaktır. Kayıp: hedef tamamlamalarda CE (ya da davranış ölçüsü) + nötr sette özgün
  modele KL cezası; `ℓ < ref_layer` iken `m`'ye karşı eğitmek `r_ref`'i doğrudan yazarak çözülür ve hiçbir
  şey kanıtlamaz (Goodhart). Tutulan değerlendirme §6 kapısıyla. Veri: kavramın pozitif/negatif örnekleri.
  CPU'da çalışır; süre Plan'da ölçülür. Model kartı "eğitilmiş yapay head" der. Bu bir küçük fine-tuning'dir
  ve öyle etiketlenir.
- **Elle tanımlama:** Plan 3'te tetikleyici ve hedef kavram/token listesinden seçilir; ileri kullanıcı
  için ayrıca bir yön vektörü dosyadan yapıştırılabilir (`.npy` değil, yalnızca float32 base64 ya da
  JSON sayı listesi; boyut `d_model` denetlenir). Ham ağırlık matrisi elle yazılmaz. Yapıştırılan vektör
  "doğrulanmamış yön" etiketi taşır ve önbellekte refusal yönü varsa ona kosinüsü denetlenir (§7).
- Model kartı, yönlendirme içeren her kayıtta "yapay head ile yönlendirildi" (ya da "sabit vektörle
  yönlendirildi") ve hedef kavramı, `α`'yı, katmanı yazar. Hedef yön bir güvenlik kavramından geliyorsa
  (ör. refusal'ı bastırma) güvenlik etiketi zorunludur.

**API (Plan 3):** `ExploreOp`'a `SteerVectorOp = {type:"steer_vector", layer, target: Vec | {logit_token_id},
alpha}` ve `SteerHeadOp = {type:"steer_head", layer, trigger: Vec | {token_id}, target: Vec |
{logit_token_id}, alpha, beta}` (`Vec` = `{d_model, vector_b64}`) eklenir; `Op`'a (kayıt) eklenmez.
`/api/graph/measure` yerine `POST /api/steer/measure` aynı sonuç şemasıyla; `steer_head` sonucu ayrıca
eşit normlu `steer_vector` temel çizgisini (`baseline: {delta_metric, kl_neutral}`) ve `circular: bool` taşır.
