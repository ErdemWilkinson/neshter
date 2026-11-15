/* Sınıf 1: ısı haritaları. Her varyant ciz(x, w, h, ctx) → bul(px, py) döndürür (anahtar ya da null). */
(function () {
  const K = window.Katalog, N = window.Nester;
  const T = Math.PI * 2;
  function cerceve(x, X, Y, W, H, renk, kalin) { x.lineWidth = kalin || 1.6; x.strokeStyle = renk; x.strokeRect(X + 0.8, Y + 0.8, W - 1.6, H - 1.6); }
  function eksen(x, ctx, w, h, sol, ust, S) {
    x.fillStyle = ctx.css('--soluk'); x.textAlign = 'right';
    for (const l of [0, Math.floor(S.L / 2), S.L - 1]) x.fillText('K' + l, sol - 4, ust + (l + 0.75) * ((h - ust) / S.L));
  }

  K.ekle({ id: 'isi-01', sinif: 'isi', ad: 'Klasik ızgara', tarif: 'Keskin kareler, viridis rampası. Uygulamadaki 2B haritanın sade hâli.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), g = ctx.izgara(w, h, 30, 4), r = ctx.rampa('viridis');
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) { x.fillStyle = r(S.g(l, hh)); x.fillRect(30 + hh * g.cw + 0.5, 4 + l * g.ch + 0.5, g.cw - 1, g.ch - 1); }
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) cerceve(x, 30 + hh * g.cw, 4 + l * g.ch, g.cw, g.ch, '#ff3b30', 2); }
      eksen(x, ctx, w, h, 30, 4, S);
      return g.hucre;
    } });

  K.ekle({ id: 'isi-02', sinif: 'isi', ad: 'Nokta matrisi', tarif: 'Her head bir nokta; büyüklük güç. Renk yok, yalnız mürekkep. Sıcaklar kendiliğinden göze batar.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), g = ctx.izgara(w, h, 30, 4), r = Math.min(g.cw, g.ch) / 2 - 0.5, mur = ctx.css('--murekkep');
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const v = S.g(l, hh), cx = 30 + (hh + 0.5) * g.cw, cy = 4 + (l + 0.5) * g.ch;
        x.fillStyle = mur; x.globalAlpha = 0.25 + 0.75 * v; x.beginPath(); x.arc(cx, cy, Math.max(0.8, r * Math.sqrt(v)), 0, T); x.fill();
        if (ctx.secili().has(N.anahtar(l, hh))) { x.globalAlpha = 1; x.strokeStyle = ctx.css('--vurgu'); x.lineWidth = 2; x.beginPath(); x.arc(cx, cy, r + 1, 0, T); x.stroke(); }
      }
      x.globalAlpha = 1; eksen(x, ctx, w, h, 30, 4, S);
      return g.hucre;
    } });

  K.ekle({ id: 'isi-03', sinif: 'isi', ad: 'Sırt çizgileri', tarif: 'Her katman bir dağ silueti; tepeler sıcak head’ler. Üst üste binen sırtlar katmanların birbirine benzerliğini gösterir.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), sol = 34, sag = 8, adim = (h - 30) / S.L, yuk = adim * 3.2, cw = (w - sol - sag) / (S.H - 1);
      const zem = ctx.css('--kart'), mur = ctx.css('--murekkep'), vur = ctx.css('--vurgu');
      for (let l = 0; l < S.L; l++) {
        const taban = 26 + (l + 1) * adim;
        x.beginPath(); x.moveTo(sol, taban);
        for (let hh = 0; hh < S.H; hh++) {
          const px = sol + hh * cw, py = taban - S.g(l, hh) * yuk;
          if (hh === 0) x.lineTo(px, py); else { const ox = sol + (hh - 0.5) * cw; x.bezierCurveTo(ox, taban - S.g(l, hh - 1) * yuk, ox, py, px, py); }
        }
        x.lineTo(sol + (S.H - 1) * cw, taban); x.closePath();
        x.fillStyle = zem; x.fill(); x.strokeStyle = mur; x.lineWidth = 1.1; x.stroke();
        for (let hh = 0; hh < S.H; hh++) if (ctx.secili().has(N.anahtar(l, hh))) { x.fillStyle = vur; x.beginPath(); x.arc(sol + hh * cw, taban - S.g(l, hh) * yuk, 3.5, 0, T); x.fill(); }
        if (l % 4 === 0) { x.fillStyle = ctx.css('--soluk'); x.textAlign = 'right'; x.fillText('K' + l, sol - 6, taban); }
      }
      return (px, py) => {
        const hh = Math.round((px - sol) / cw); if (hh < 0 || hh >= S.H) return null;
        let en = null, ed = 12;
        for (let l = 0; l < S.L; l++) { const ty = 26 + (l + 1) * adim - S.g(l, hh) * yuk, dd = Math.abs(ty - py); if (dd < ed) { ed = dd; en = N.anahtar(l, hh); } }
        return en;
      };
    } });

  K.ekle({ id: 'isi-04', sinif: 'isi', ad: 'Petek', tarif: 'Altıgen hücreler, kor rampası. Sıralar yarım kaydırılmış; komşuluk dört değil altı yönde okunur.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), r = Math.min((w - 20) / (S.H * Math.sqrt(3) + 1), (h - 12) / (S.L * 1.5 + 0.5)), rn = ctx.rampa('kor');
      const hw = Math.sqrt(3) * r, ox = (w - S.H * hw - hw / 2) / 2 + hw / 2, oy = r + 4, merkez = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const cx = ox + hh * hw + (l % 2) * hw / 2, cy = oy + l * r * 1.5;
        x.beginPath(); for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + i * Math.PI / 3; x.lineTo(cx + Math.cos(a) * (r - 0.6), cy + Math.sin(a) * (r - 0.6)); } x.closePath();
        x.fillStyle = rn(S.g(l, hh)); x.fill();
        if (ctx.secili().has(N.anahtar(l, hh))) { x.strokeStyle = ctx.css('--ikinci'); x.lineWidth = 2.4; x.stroke(); }
        merkez.push([cx, cy, N.anahtar(l, hh)]);
      }
      return (px, py) => { let en = null, ed = r; for (const [cx, cy, k] of merkez) { const dd = Math.hypot(cx - px, cy - py); if (dd < ed) { ed = dd; en = k; } } return en; };
    } });

  K.ekle({ id: 'isi-05', sinif: 'isi', ad: 'Disk', tarif: 'Halkalar katman (içte K0), dilimler head. Modelin tamamı tek bakışta; derinlik dışa doğru akar.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 18, r0 = R * 0.22, dr = (R - r0) / S.L, da = T / S.H, b = -Math.PI / 2, rn = ctx.rampa('magma');
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        x.beginPath(); x.arc(cx, cy, r0 + (l + 1) * dr - 0.4, b + hh * da + 0.01, b + (hh + 1) * da - 0.01); x.arc(cx, cy, r0 + l * dr + 0.4, b + (hh + 1) * da - 0.01, b + hh * da + 0.01, true); x.closePath();
        x.fillStyle = rn(S.g(l, hh)); x.fill();
        if (ctx.secili().has(N.anahtar(l, hh))) { x.strokeStyle = '#3ad1ff'; x.lineWidth = 2; x.stroke(); }
      }
      x.fillStyle = ctx.css('--soluk'); x.textAlign = 'center';
      for (let hh = 0; hh < S.H; hh += 2) { const a = b + (hh + 0.5) * da; x.fillText('H' + hh, cx + Math.cos(a) * (R + 10), cy + Math.sin(a) * (R + 10) + 4); }
      return (px, py) => {
        const dx = px - cx, dy = py - cy, rr = Math.hypot(dx, dy); if (rr < r0 || rr > R) return null;
        let a = Math.atan2(dy, dx) - b; a = ((a % T) + T) % T;
        return N.anahtar(Math.min(S.L - 1, Math.floor((rr - r0) / dr)), Math.min(S.H - 1, Math.floor(a / da)));
      };
    } });

  K.ekle({ id: 'isi-06', sinif: 'isi', ad: 'Glif matrisi', tarif: 'Renk yerine harf: ▁▂▃▄▅▆▇█. Metin gibi kopyalanabilir, terminale ve rapora olduğu gibi geçer.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), G = '▁▂▃▄▅▆▇█', g = ctx.izgara(w, h, 30, 4), boy = Math.min(g.cw * 1.25, g.ch * 1.15);
      x.font = boy + 'px ' + ctx.css('--m'); x.textAlign = 'center'; x.textBaseline = 'middle';
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const v = S.g(l, hh), sec = ctx.secili().has(N.anahtar(l, hh));
        if (sec) { x.fillStyle = ctx.css('--vurgu'); x.fillRect(30 + hh * g.cw, 4 + l * g.ch, g.cw, g.ch); }
        x.fillStyle = sec ? ctx.css('--kart') : ctx.css('--murekkep'); x.globalAlpha = sec ? 1 : 0.35 + 0.65 * v;
        x.fillText(G[Math.min(7, Math.floor(v * 8))], 30 + (hh + 0.5) * g.cw, 4 + (l + 0.55) * g.ch);
      }
      x.globalAlpha = 1; x.font = '11px ' + ctx.css('--m'); x.textBaseline = 'alphabetic'; eksen(x, ctx, w, h, 30, 4, S);
      return g.hucre;
    } });

  K.ekle({ id: 'isi-07', sinif: 'isi', ad: 'Topografya', tarif: 'Değerler yumuşatılıp yükselti bantlarına bölünür; eş yükselti çizgileri sıcak bölgelerin sınırını çizer.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), sol = 30, ust = 4, W = w - sol, H = h - ust, adim = 3, rn = ctx.rampa('yosun'), BANT = 7;
      const deger = (fx, fy) => { // çift doğrusal ara değer
        const gx = Math.max(0, Math.min(S.H - 1.001, fx * S.H - 0.5)), gy = Math.max(0, Math.min(S.L - 1.001, fy * S.L - 0.5));
        const i = Math.floor(gx), j = Math.floor(gy), u = gx - i, v = gy - j;
        return S.g(j, i) * (1 - u) * (1 - v) + S.g(j, i + 1) * u * (1 - v) + S.g(j + 1, i) * (1 - u) * v + S.g(j + 1, i + 1) * u * v;
      };
      const nx = Math.ceil(W / adim), ny = Math.ceil(H / adim), bant = [];
      for (let j = 0; j < ny; j++) { bant.push([]); for (let i = 0; i < nx; i++) bant[j].push(Math.min(BANT - 1, Math.floor(deger((i + 0.5) / nx, (j + 0.5) / ny) * BANT))); }
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { x.fillStyle = rn(bant[j][i] / (BANT - 1)); x.fillRect(sol + i * adim, ust + j * adim, adim + 0.5, adim + 0.5); }
      x.fillStyle = ctx.css('--murekkep'); x.globalAlpha = 0.7;
      for (let j = 1; j < ny; j++) for (let i = 1; i < nx; i++) if (bant[j][i] !== bant[j][i - 1] || bant[j][i] !== bant[j - 1][i]) x.fillRect(sol + i * adim, ust + j * adim, 1, 1);
      x.globalAlpha = 1;
      const g = ctx.izgara(w, h, sol, ust);
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) { x.strokeStyle = ctx.css('--vurgu'); x.lineWidth = 2; x.beginPath(); x.arc(sol + (hh + 0.5) * g.cw, ust + (l + 0.5) * g.ch, Math.min(g.cw, g.ch) * 0.7, 0, T); x.stroke(); } }
      eksen(x, ctx, w, h, sol, ust, S);
      return g.hucre;
    } });

  K.ekle({ id: 'isi-08', sinif: 'isi', ad: 'Ağaç haritası', tarif: 'Her katman bir şerit, yüksekliği katmanın toplam gücü; içinde head’ler alanlarıyla. Ağır katmanlar yer kaplar.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), rn = ctx.rampa('toprak'), top = [], kutu = [];
      let hepsi = 0; for (let l = 0; l < S.L; l++) { let s = 0; for (let hh = 0; hh < S.H; hh++) s += S.g(l, hh); top.push(s); hepsi += s; }
      let y = 2;
      for (let l = 0; l < S.L; l++) {
        const sh = (h - 4) * top[l] / hepsi; let xx = 30;
        for (let hh = 0; hh < S.H; hh++) {
          const cw = (w - 32) * S.g(l, hh) / top[l];
          x.fillStyle = rn(S.g(l, hh)); x.fillRect(xx + 0.4, y + 0.4, Math.max(0, cw - 0.8), Math.max(0, sh - 0.8));
          if (ctx.secili().has(N.anahtar(l, hh))) cerceve(x, xx, y, cw, sh, ctx.css('--ikinci'), 2.2);
          kutu.push([xx, y, cw, sh, N.anahtar(l, hh)]); xx += cw;
        }
        if (sh > 9) { x.fillStyle = ctx.css('--soluk'); x.textAlign = 'right'; x.fillText('K' + l, 26, y + sh / 2 + 4); }
        y += sh;
      }
      return (px, py) => { const q = kutu.find(([a, b, c, d]) => px >= a && px < a + c && py >= b && py < b + d); return q ? q[4] : null; };
    } });

  K.ekle({ id: 'isi-09', sinif: 'isi', ad: 'Kenar toplamlı', tarif: 'Ortada harita; sağda katman toplamları, altta head toplamları. Hangi katman ve hangi head numarası ağır, kenardan okunur.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), sag = 54, alt = 44, g = ctx.izgara(w, h, 30, 4, sag, alt), rn = ctx.rampa('buz'), ikinci = ctx.css('--ikinci');
      const lt = [], ht = new Array(S.H).fill(0);
      for (let l = 0; l < S.L; l++) { let s = 0; for (let hh = 0; hh < S.H; hh++) { const v = S.g(l, hh); s += v; ht[hh] += v; x.fillStyle = rn(1 - v); x.fillRect(30 + hh * g.cw + 0.5, 4 + l * g.ch + 0.5, g.cw - 1, g.ch - 1); } lt.push(s); }
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) cerceve(x, 30 + hh * g.cw, 4 + l * g.ch, g.cw, g.ch, ctx.css('--vurgu'), 2); }
      const ml = Math.max(...lt), mh = Math.max(...ht), sx = w - sag + 6, sy = h - alt + 6;
      x.fillStyle = ikinci;
      lt.forEach((s, l) => x.fillRect(sx, 4 + l * g.ch + 1, (sag - 10) * s / ml, g.ch - 2));
      ht.forEach((s, hh) => x.fillRect(30 + hh * g.cw + 1, sy, g.cw - 2, (alt - 12) * s / mh));
      eksen(x, ctx, w, h - alt, 30, 4, S);
      return (px, py) => g.hucre(px, py);
    } });

  K.ekle({ id: 'isi-10', sinif: 'isi', ad: 'Sapma', tarif: 'Her head katmanının ortalamasından ne kadar sapıyor: mavi ortalamanın altı, kırmızı üstü. Katmanın yıldızları ayrışır.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), g = ctx.izgara(w, h, 30, 4), mavi = N.rampa(['#2b59c3', '#f4f4f4']), kirmizi = N.rampa(['#f4f4f4', '#d1495b']);
      for (let l = 0; l < S.L; l++) {
        let o = 0; for (let hh = 0; hh < S.H; hh++) o += S.g(l, hh); o /= S.H;
        for (let hh = 0; hh < S.H; hh++) { const s = S.g(l, hh) - o; x.fillStyle = s < 0 ? mavi(1 + s * 4) : kirmizi(Math.min(1, s * 1.6)); x.fillRect(30 + hh * g.cw + 0.5, 4 + l * g.ch + 0.5, g.cw - 1, g.ch - 1); }
      }
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) cerceve(x, 30 + hh * g.cw, 4 + l * g.ch, g.cw, g.ch, '#111', 2); }
      eksen(x, ctx, w, h, 30, 4, S);
      return g.hucre;
    } });

  K.ekle({ id: 'isi-11', sinif: 'isi', ad: 'Sarmal', tarif: 'Head’ler katman sırasıyla içten dışa bir sarmal boyunca dizilir. Derin katmanların sıcakları dış turlarda parlar.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), n = S.L * S.H, cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 10, tur = 7, rn = ctx.rampa('magma'), nok = [];
      for (let i = 0; i < n; i++) {
        const t = i / n, a = t * tur * T, r = 10 + t * (R - 10), l = Math.floor(i / S.H), hh = i % S.H, v = S.g(l, hh);
        const px = cx + Math.cos(a) * r, py = cy + Math.sin(a) * r, rr = 1.2 + 4.4 * v;
        x.fillStyle = rn(0.15 + 0.85 * v); x.beginPath(); x.arc(px, py, rr, 0, T); x.fill();
        if (ctx.secili().has(N.anahtar(l, hh))) { x.strokeStyle = '#3ad1ff'; x.lineWidth = 2; x.beginPath(); x.arc(px, py, rr + 3, 0, T); x.stroke(); }
        nok.push([px, py, N.anahtar(l, hh)]);
      }
      return (px, py) => { let en = null, ed = 7; for (const [a, b, k] of nok) { const dd = Math.hypot(a - px, b - py); if (dd < ed) { ed = dd; en = k; } } return en; };
    } });

  K.ekle({ id: 'isi-12', sinif: 'isi', ad: 'Sıralı merdiven', tarif: 'Her katmanın head’leri büyükten küçüğe dizilir. Konum kaybolur, dağılım kalır: tek yıldızlı katman ile düz katman hemen ayrılır.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), g = ctx.izgara(w, h, 30, 4, 54), rn = ctx.rampa('buz'), harita = [];
      for (let l = 0; l < S.L; l++) {
        const sira = [...Array(S.H).keys()].sort((a, b) => S.g(l, b) - S.g(l, a)); harita.push(sira);
        sira.forEach((hh, i) => { x.fillStyle = rn(1 - S.g(l, hh)); x.fillRect(30 + i * g.cw + 0.5, 4 + l * g.ch + 0.5, g.cw - 1, g.ch - 1); if (ctx.secili().has(N.anahtar(l, hh))) cerceve(x, 30 + i * g.cw, 4 + l * g.ch, g.cw, g.ch, ctx.css('--vurgu'), 2); });
        if (g.ch > 9) { x.fillStyle = ctx.css('--soluk'); x.textAlign = 'left'; x.fillText('H' + sira[0], w - 50, 4 + (l + 0.8) * g.ch); }
      }
      eksen(x, ctx, w, h, 30, 4, S);
      return (px, py) => { const i = Math.floor((px - 30) / g.cw), l = Math.floor((py - 4) / g.ch); return l >= 0 && l < S.L && i >= 0 && i < S.H ? N.anahtar(l, harita[l][i]) : null; };
    } });
})();
