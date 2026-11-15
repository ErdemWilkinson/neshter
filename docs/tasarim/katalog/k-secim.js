/* Seçtiklerin: Erdem'in işaretlerinde seçtiği parçalardan kurulan yeni Neşter taslağı.
   Noktalı ısı haritası (isi-02 biçimi + isi-01 renkleri), yüz nokta + manşet (olc-06 + olc-05), lens merdiveni (diğer cevaplarla),
   dikkat matrisi (dkt-01) ve neon, hareketli 3B bağlar (uc-00 + Neşter'in kamera ve büyüme hareketleri). */
(function () {
  const K = window.Katalog, N = window.Nester, el = N.el;
  K.SINIFLAR.unshift({ id: 'secim', ad: 'Seçtiklerin', tarif: 'İşaretlerinde beğendiğin parçalar bir araya geldi. Her kartta “şu ankiyle kıyasla” bugünkü Neşter’le yan yana açar.', genis: true });
  const bul = (id) => K.varyantlar.find((v) => v.id === id);

  K.ekle({ id: 'sec-01', sinif: 'secim', kiyas: 'isk-00', ad: 'Yeni keşif ekranı',
    tarif: 'Solda noktalı ısı haritası ve altında dikkat matrisi; sağda yüz nokta ölçümü (eski olasılık üstü çizili, yenisi büyük, neyin kapalı olduğu yazılı) ve diğer cevapların oranlarıyla lens merdiveni.',
    ipucu: 'noktaya tıkla: head seç · seçim bütün panellerde ve öbür kartlarda değişir',
    dom(kap, ctx) {
      const f = el('div', { class: 'mini', style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:8px;align-items:start' });
      kap.append(f);
      const pan = (ad) => { const p = el('div', { class: 'p' }); p.append(el('b', { text: ad })); return p; };
      const sol = el('div', { style: 'display:grid;gap:8px;min-width:0' }), sag = el('div', { style: 'display:grid;gap:8px;min-width:0' });
      const a = pan('Isı haritası · noktalar'), b = pan('Dikkat matrisi'), c = pan('Gerçek etki'), d = pan('Katman katman tahmin');
      sol.append(a, b); sag.append(c, d); f.append(sol, sag);
      const dkt = bul('dkt-01');
      return ((fs) => () => fs.forEach((g) => g()))([
        ctx.mini.tuval(a, 300, (x, w, h) => { x.font = '10px ' + ctx.css('--m'); x.fillStyle = ctx.css('--soluk'); x.textAlign = 'right'; const S = ctx.S(); for (const l of [0, Math.floor(S.L / 2), S.L - 1]) x.fillText('K' + l, 24, (l + 0.75) * (h / S.L)); return ctx.noktaIsi(x, 28, 0, w - 28, h); }),
        ctx.mini.tuval(b, 260, (x, w, h) => dkt.ciz(x, w, h, ctx)),
        ctx.mini.tuval(c, 240, (x, w, h) => K.yuzNokta(x, w, h, ctx)),
        ctx.mini.tuval(d, 360, (x, w, h) => K.lensMerdiven(x, w, h, ctx)),
      ]);
    } });

  // Neon bağlar: Neşter'in drawFlow.ts sahnesinin kopyası (kavisli nehirler, akan ışıklar, halkalı head'ler, token satırı,
  // cevap halkası, kesimde kopan nehir), taban noktalı harita, üstüne hafif neon parıltı.
  const PAL = { bg: '#0b0e13', line: '#222a38', t1: '#e6eaf2', t2: '#a7b0c0', inc: '#2ee6c5', focus: '#9fc3ff', kes: '#ff7a45', isik: '#c8fff5', sec: '#ff3b30' };
  const MONO = '"JetBrains Mono", ui-monospace, monospace', RZ = 0.62, ADIM = 32;
  const c01 = (v) => Math.max(0, Math.min(1, v)), eo = (v) => 1 - Math.pow(1 - c01(v), 3);

  function neonAkis(x, w, h, ctx, kam, t) {
    const S = ctx.S(), G = N.bagGrafi(S), L = S.L, H = S.H, sec = ctx.secili(), rn = ctx.rampa('viridis'), sn = t / 1000;
    const toks = S.istem.tokenlar, T = toks.length;
    x.fillStyle = PAL.bg; x.fillRect(0, 0, w, h); x.lineCap = 'round'; x.lineJoin = 'round';
    const mix = c01((1.45 - kam.pitch) / 0.5);
    const heatA = 1 - eo(mix * 1.5), flowA = eo((mix - 0.35) / 0.65), nodeA = eo((mix - 0.3) / 0.5), ringA = eo((mix - 0.2) / 0.5);
    const s = Math.min(w * 0.84 / H, h * 0.78 / ((L + 2.6) * RZ)) * (0.78 + 0.22 * Math.sin(kam.pitch));
    const P = ctx.P(kam, w, h, s, 4), kpx = Math.max(0.6, Math.min(1.4, s / 42));
    const hx = (hh) => hh - (H - 1) / 2, lz = (l) => (l - (L - 1) / 2) * RZ;
    const zTok = lz(0) - 1.3 * RZ, zCev = lz(L - 1) + 1.3 * RZ, yay = Math.max(H - 1, Math.min(1.3 * (T - 1), 2 * (H - 1)));
    const tokX = (i) => (T <= 1 ? 0 : -yay / 2 + (yay * i) / (T - 1));

    // düğümler ve dünya konumları
    const dm = new Map(G.dugumler.map((n) => [n.id, n])), grafHead = new Set(G.dugumler.filter((n) => n.tur === 'head').map((n) => n.id));
    const konum = (n) => (n.tur === 'head' ? [hx(n.h), 0, lz(n.l)] : n.tur === 'cikti' ? [0, 0, zCev] : [0, 0, zTok]);
    const kaynak = (b) => { const a = dm.get(b.k); if (a.tur !== 'girdi') return konum(a); const d = dm.get(b.h); return [tokX(((d.h || 0) * 7 + (d.l || 0)) % T), 0, zTok]; };
    // akan pay: kesilen (hedefi seçili) bağ 0, düğüm gelen payın kesilmeyen oranını iletir
    const sira = [...G.dugumler].sort((a, b) => (a.tur === 'girdi' ? -1 : a.tur === 'cikti' ? 1e9 : a.l) - (b.tur === 'girdi' ? -1 : b.tur === 'cikti' ? 1e9 : b.l));
    const oran = new Map(), canli = new Map(), maxW = Math.max(...G.baglar.map((b) => b.w), 1e-6);
    for (const n of sira) { let hep = 0, kalan = 0; for (const b of G.baglar) if (b.h === n.id) { const v = sec.has(b.h) ? 0 : b.w * (oran.get(b.k) ?? 1); canli.set(b, v); hep += b.w; kalan += v; } oran.set(n.id, hep > 0 ? kalan / hep : 1); }

    // satır çizgileri (cevaba yaklaştıkça parlak) ve etiketler
    const yarim = H / 2 + 0.2, satirPx = Math.abs(P(0, 0, lz(0))[1] - P(0, 0, lz(1))[1]), adim = satirPx >= 12 ? 1 : 4;
    for (let l = 0; l < L; l++) {
      const a = P(-yarim, 0, lz(l)), b = P(yarim, 0, lz(l)), gw = (l + 1) / L;
      if (flowA > 0) { x.globalAlpha = flowA * (0.12 + 0.3 * gw); x.strokeStyle = gw > 0.3 ? PAL.inc : PAL.line; x.lineWidth = 0.5 + gw * 0.6; x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke(); }
      if (l % adim === 0) { x.globalAlpha = 1; x.fillStyle = PAL.t2; x.textAlign = 'right'; x.font = '10px ' + MONO; x.fillText('K' + l, a[0] - 6, a[1] + 4); }
    }
    if (heatA > 0) { x.globalAlpha = heatA; x.fillStyle = PAL.t2; x.textAlign = 'center'; for (let hh = 0; hh < H; hh += H > 24 ? 4 : 2) { const p = P(hx(hh), 0, lz(0) - RZ * 0.72); x.fillText('H' + hh, p[0], p[1]); } }
    x.globalAlpha = 1;

    // taban: noktalı harita. Grafikteki head'in noktası halkasına çekilir, öbürleri tabanda kalır.
    const rmax = 0.46 * s * RZ * (kam.zoom || 1), taban = [];
    for (let l = 0; l < L; l++) for (let hh = 0; hh < H; hh++) {
      const id = N.anahtar(l, hh), ic = grafHead.has(id), v = S.g(l, hh);
      const kucul = ic ? 1 - 0.88 * eo(mix * 1.3) : 1 - 0.3 * eo(mix);
      const al = ic ? 1 - eo((mix - 0.25) / 0.5) : Math.max(heatA, 0.7);
      const p = P(hx(hh), 0, lz(l)); taban.push([p[0], p[1], id]);
      if (al <= 0) continue;
      const rx = Math.max(1, rmax * (0.3 + 0.7 * Math.sqrt(v)) * kucul), ry = rx * Math.max(0.35, Math.sin(kam.pitch));
      x.globalAlpha = al * (0.65 + 0.35 * v); x.fillStyle = rn(0.18 + 0.82 * v); /* koyu zeminde en düşük mor kaybolmasın */ x.beginPath(); x.ellipse(p[0], p[1], rx, ry, 0, 0, T2); x.fill();
      if (sec.has(id)) { x.globalAlpha = al; x.strokeStyle = PAL.sec; x.lineWidth = 2; x.beginPath(); x.ellipse(p[0], p[1], rmax + 1.5, (rmax + 1.5) * Math.max(0.35, Math.sin(kam.pitch)), 0, 0, T2); x.stroke(); }
    }
    x.globalAlpha = 1;

    // nehirler: kübik kavis, güçlü bağ daha yüksek; girdiden cevaba doğru büyür, üstünde ışıklar akar
    const kenarlar = [];
    if (flowA > 0) {
      const sirali = [...G.baglar].sort((a, b) => a.w - b.w);
      sirali.forEach((b, i) => {
        const A = kaynak(b), B = konum(dm.get(b.h)), dz = B[2] - A[2], ag = Math.min(1, b.w / maxW);
        const wob = ((i % 2 === 0 ? -1 : 1) * (20 + ((i * 37) % 50))) / 120, kalk = (18 + ag * 60) / 120;
        const c1 = [A[0] + wob * 0.4, kalk, A[2] + dz * 0.35], c2 = [B[0] - wob * 0.6, kalk, A[2] + dz * 0.7];
        const ps = [];
        for (let j = 0; j <= ADIM; j++) { const u = j / ADIM, v = 1 - u, f = (k) => v * v * v * A[k] + 3 * v * v * u * c1[k] + 3 * v * u * u * c2[k] + u * u * u * B[k]; ps.push(P(f(0), f(1), f(2))); }
        const kd = dm.get(b.k), nereden = kd.tur === 'girdi' ? 0 : (kd.l + 1) / L;
        kenarlar.push({ b, ps, buyu: c01((mix - 0.35 - 0.3 * nereden) / 0.35), live: (canli.get(b) ?? b.w) / maxW, kes: sec.has(b.h), id: i });
      });
      for (const e of kenarlar) {
        if (e.buyu <= 0) continue;
        const gen = (1.1 + 7 * Math.min(1, e.b.w / maxW)) * kpx;
        if (e.kes) { // kesim: nehir ortadan kopar, iki yarı uçlarına çekilir, ortada ✕
          x.shadowBlur = 3; x.shadowColor = PAL.kes; x.strokeStyle = PAL.kes; x.globalAlpha = flowA * (0.55 + 0.25 * Math.sin(t / 160)); x.lineWidth = gen * 0.55;
          for (const [u0, u1] of [[0, 0.34], [0.66, 1]]) { x.beginPath(); for (let j = Math.round(u0 * ADIM); j <= Math.round(u1 * ADIM); j++) { const p = e.ps[j]; j === Math.round(u0 * ADIM) ? x.moveTo(p[0], p[1]) : x.lineTo(p[0], p[1]); } x.stroke(); }
          const m = e.ps[ADIM / 2], d = 4 * kpx; x.lineWidth = 1.6; x.beginPath(); x.moveTo(m[0] - d, m[1] - d); x.lineTo(m[0] + d, m[1] + d); x.moveTo(m[0] + d, m[1] - d); x.lineTo(m[0] - d, m[1] + d); x.stroke();
          continue;
        }
        const goster = e.ps.slice(0, Math.max(1, Math.round(e.buyu * ADIM)) + 1);
        x.shadowBlur = 4; x.shadowColor = PAL.inc; x.strokeStyle = PAL.inc; x.lineWidth = gen * (0.4 + 0.6 * Math.min(1, e.live / Math.max(1e-6, e.b.w / maxW)));
        x.globalAlpha = flowA * (0.3 + 0.55 * Math.min(1, e.live * 2.2));
        x.beginPath(); goster.forEach((p, j) => (j ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]))); x.stroke();
        if (!N.azalt() && e.buyu >= 1 && e.live > 0.04) {
          const n = Math.round(1 + e.live * 5), tohum = (e.id * 0.37) % 1;
          x.globalAlpha = flowA; x.fillStyle = PAL.isik; x.shadowBlur = 5;
          for (let j = 0; j < n; j++) {
            const u = (sn * (0.18 + e.live * 0.12) + j / n + tohum) % 1, fi = u * ADIM, i0 = Math.floor(fi), a = e.ps[i0], b2 = e.ps[Math.min(ADIM, i0 + 1)], q = fi - i0;
            x.beginPath(); x.arc(a[0] + (b2[0] - a[0]) * q, a[1] + (b2[1] - a[1]) * q, Math.max(1.2, x.lineWidth * 0.45 + 0.6), 0, T2); x.fill();
          }
        }
      }
      x.globalCompositeOperation = 'source-over'; x.shadowBlur = 0;
    }

    // düğümler: head halkası (içinde ısı rengi), token kareleri, cevap halkası
    const etiket = [], nok = [], sicak = new Set(S.sicak.slice(0, 4).map((q) => q.k));
    for (const n of G.dugumler) {
      if (n.tur !== 'head') continue;
      if (ringA <= 0) break;
      const p = P(...konum(n)), r = 6.5 * kpx, isik = sec.has(n.id);
      x.globalAlpha = ringA; x.fillStyle = PAL.bg; x.strokeStyle = isik ? PAL.sec : PAL.inc; x.lineWidth = isik ? 2.6 : 2.2;
      x.shadowBlur = 3; x.shadowColor = isik ? PAL.sec : PAL.inc;
      x.beginPath(); x.arc(p[0], p[1], isik ? r + 1 : r, 0, T2); x.fill(); x.stroke(); x.shadowBlur = 0;
      x.fillStyle = rn(S.g(n.l, n.h)); x.beginPath(); x.arc(p[0], p[1], r * 0.45, 0, T2); x.fill();
      if (mix >= 0.5) nok.push([p[0], p[1], n.id]);
      if (isik || sicak.has(n.id)) etiket.push({ ad: n.ad || N.adK(n.id), p, a: ringA, tur: 'head', isik });
    }
    if (nodeA > 0) {
      x.globalAlpha = nodeA;
      for (let i = 0; i < T; i++) { const p = P(tokX(i), 0, zTok), d = 3 * kpx; x.fillStyle = PAL.t1; x.fillRect(p[0] - d, p[1] - d, 2 * d, 2 * d); etiket.push({ ad: N.tok(toks[i]).trim() || '␣', p, a: nodeA, tur: 'token' }); }
      const p = P(0, 0, zCev);
      x.fillStyle = PAL.bg; x.strokeStyle = PAL.t1; x.lineWidth = 2.5; x.shadowBlur = 6; x.shadowColor = PAL.inc;
      x.beginPath(); x.arc(p[0], p[1], 10 * kpx, 0, T2); x.fill(); x.stroke(); x.shadowBlur = 0;
      x.fillStyle = PAL.inc; x.beginPath(); x.arc(p[0], p[1], (4.5 + 0.8 * Math.sin(t / 300)) * kpx, 0, T2); x.fill();
      const tah = S.olc([...sec]).tahmin; etiket.push({ ad: N.tok(tah).trim(), p, a: nodeA, tur: 'cikti' });
    }
    // etiketler: çakışan atlanır; seçili ve cevap önce
    const yer = [], rank = (e) => (e.isik ? 0 : e.tur === 'cikti' ? 1 : e.tur === 'token' ? 3 : 2);
    etiket.sort((a, b) => rank(a) - rank(b));
    for (const e of etiket) {
      const ort = e.tur !== 'head';
      x.font = (e.isik || e.tur === 'cikti' ? 600 : 500) + ' ' + (e.tur === 'cikti' ? 13 : 11) + 'px ' + MONO;
      const tx = ort ? e.p[0] : e.p[0] + 10 * kpx, ty = ort ? (e.tur === 'cikti' ? e.p[1] + 26 * kpx : e.p[1] - 10) : e.p[1] + 4, tw = x.measureText(e.ad).width;
      const kutu = [ort ? tx - tw / 2 - 2 : tx - 2, ty - 11, tw + 4, 14];
      if (yer.some((q) => kutu[0] < q[0] + q[2] && q[0] < kutu[0] + kutu[2] && kutu[1] < q[1] + q[3] && q[1] < kutu[1] + kutu[3]) && !e.isik && e.tur !== 'cikti') continue;
      yer.push(kutu); x.globalAlpha = e.a; x.textAlign = ort ? 'center' : 'left'; x.fillStyle = ort || e.isik ? PAL.t1 : PAL.t2; x.fillText(e.ad, tx, ty);
    }
    x.globalAlpha = 1; x.textAlign = 'left'; x.font = '11px ' + MONO; x.fillStyle = PAL.t2;
    x.fillText(mix > 0.02 ? 'Bağlar 3B · ' + G.baglar.length + ' nehir · ' + Math.round(mix * 100) + '% eğik' : 'Isı haritası (tepeden, noktalı) · aşağı sürükle', 10, h - 10);
    return (px, py) => {
      for (const q of nok) if (Math.hypot(q[0] - px, q[1] - py) < 10) return q[2];
      if (mix >= 0.5) return null;
      let en = null, ed = Math.max(6, rmax * 1.6); for (const q of taban) { const d = Math.hypot(q[0] - px, q[1] - py); if (d < ed) { ed = d; en = q[2]; } } return en;
    };
  }
  const T2 = Math.PI * 2;

  // Açılış: tepeden noktalı harita, kamera Neşter'deki gibi yumuşakça eğilir, nehirler girdiden cevaba büyür. Çift tık yeniden oynatır.
  const durum = new WeakMap();
  K.ekle({ id: 'sec-02', sinif: 'secim', kiyas: 'uc-00', ad: 'Neon bağlar', uc: true, anim: true, yuk: 460, kam: { yaw: -0.18, pitch: 1.5, zoom: 0.92 },
    tarif: 'Sevdiğin 3B bağlar, Neşter’in sahnesiyle: tepeden bakınca taban noktalı harita; eğildikçe grafikteki head’lerin noktaları halkalarına çekilir, nehirler girdiden cevaba doğru büyür ve üstlerinde ışıklar akar. Kesilen nehir ortadan kopar. Hafif neon parıltı.',
    ipucu: 'çift tıkla: açılışı yeniden oynat · sürükle: döndür (yukarı: harita, aşağı: nehirler) · tıkla: seç',
    ciz(x, w, h, ctx, kam, t) {
      let d = durum.get(x.canvas);
      if (!d) { d = { t0: null }; durum.set(x.canvas, d); x.canvas.addEventListener('dblclick', () => { d.t0 = null; }); }
      const simdi = performance.now();
      if (d.t0 === null) { d.t0 = simdi; kam.pitch = 1.5; }
      const p = N.azalt() ? 1 : Math.min(1, (simdi - d.t0 - 500) / 2600);
      if (p < 1) kam.pitch = 1.5 + (0.95 - 1.5) * eo(Math.max(0, p));
      return neonAkis(x, w, h, ctx, kam, t || 0);
    } });
})();
