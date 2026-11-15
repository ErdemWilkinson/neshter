/* Sınıf 4: ölçüm göstergeleri, Sınıf 5: lens ve dikkat. Hepsi ortak seçimin kapatılmasını gösterir. */
(function () {
  const K = window.Katalog, N = window.Nester, T = Math.PI * 2;
  const olc = (ctx) => ctx.S().olc([...ctx.secili()]);
  const yazi = (x, ctx, t, X, Y, boy, renk, hiza, kalin) => { x.font = (kalin || 400) + ' ' + boy + 'px ' + ctx.css('--f'); x.fillStyle = renk || ctx.css('--murekkep'); x.textAlign = hiza || 'left'; x.fillText(t, X, Y); };
  const hedefP = (row, tok) => (row.find((q) => q.tok === tok) || { p: 0 }).p;

  // ── Ölçüm göstergeleri
  K.ekle({ id: 'olc-01', sinif: 'olcum', ad: 'Eğim', tarif: 'İki dikey eksen: önce ve sonra. Hedefin çizgisi düşer, rakibinki yükselir; kesiştikleri an tahmin değişir.', yuk: 240, ipucu: 'seçimi değiştir: üstteki çubuktan ya da başka bir karttan',
    ciz(x, w, h, ctx) {
      const r = olc(ctx), S = ctx.S(), x1 = w * 0.28, x2 = w * 0.72, y = (p) => h - 26 - p * (h - 50);
      x.strokeStyle = ctx.css('--cizgi'); x.lineWidth = 1; [x1, x2].forEach((X) => { x.beginPath(); x.moveTo(X, 18); x.lineTo(X, h - 22); x.stroke(); });
      yazi(x, ctx, 'önce', x1, h - 6, 11, ctx.css('--soluk'), 'center'); yazi(x, ctx, 'sonra', x2, h - 6, 11, ctx.css('--soluk'), 'center');
      [[S.istem.hedef, r.once, r.sonra, ctx.css('--ikinci')], [S.istem.rakip, r.rakipOnce, r.rakipSonra, ctx.css('--vurgu')]].forEach(([t, a, b, c]) => {
        x.strokeStyle = c; x.lineWidth = 3; x.beginPath(); x.moveTo(x1, y(a)); x.lineTo(x2, y(b)); x.stroke();
        [[x1, a, 'right', -8], [x2, b, 'left', 8]].forEach(([X, p, hz, dx]) => { x.fillStyle = c; x.beginPath(); x.arc(X, y(p), 4.5, 0, T); x.fill(); yazi(x, ctx, N.tok(t).trim() + ' ' + N.yuzde(p), X + dx, y(p) + 4, 12, c, hz, 600); });
      });
    } });

  K.ekle({ id: 'olc-02', sinif: 'olcum', ad: 'Gösterge', tarif: 'Yarım daire bir saat: soluk ibre kesimden önceki olasılık, koyu ibre sonrası. Aradaki yay kaybı gösterir.', yuk: 240,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), cx = w / 2, cy = h - 34, R = Math.min(w / 2 - 20, h - 60), a = (p) => Math.PI + p * Math.PI;
      x.lineCap = 'round'; x.lineWidth = 14; x.strokeStyle = ctx.css('--cizgi'); x.beginPath(); x.arc(cx, cy, R, Math.PI, T); x.stroke();
      x.strokeStyle = ctx.css('--vurgu'); x.beginPath(); x.arc(cx, cy, R, a(r.sonra), a(r.once)); x.stroke();
      x.strokeStyle = ctx.css('--ikinci'); x.beginPath(); x.arc(cx, cy, R, Math.PI, a(r.sonra)); x.stroke();
      [[r.once, ctx.css('--soluk'), 2], [r.sonra, ctx.css('--murekkep'), 3.5]].forEach(([p, c, lw]) => { x.strokeStyle = c; x.lineWidth = lw; x.beginPath(); x.moveTo(cx, cy); x.lineTo(cx + Math.cos(a(p)) * (R - 18), cy + Math.sin(a(p)) * (R - 18)); x.stroke(); });
      x.lineCap = 'butt';
      yazi(x, ctx, N.yuzde(r.sonra), cx, cy + 26, 22, null, 'center', 800);
      yazi(x, ctx, '%0', cx - R, cy + 18, 10, ctx.css('--soluk'), 'center'); yazi(x, ctx, '%100', cx + R, cy + 18, 10, ctx.css('--soluk'), 'center');
    } });

  K.ekle({ id: 'olc-03', sinif: 'olcum', ad: 'Şelale', tarif: 'Her kapatılan head’in payı ayrı bir basamak. Hangi head’in düşüşün çoğunu taşıdığı tek bakışta görülür.', yuk: 240,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), k = [...ctx.secili()], adim = [{ ad: 'önce', p: S.once, bas: 0 }];
      let onceki = S.once; k.forEach((q, i) => { const p = S.olc(k.slice(0, i + 1)).sonra; adim.push({ ad: N.adK(q), dus: onceki - p, ust: onceki, p }); onceki = p; });
      adim.push({ ad: 'sonra', p: onceki, bas: 0 });
      const n = adim.length, bw = (w - 30) / n, y = (p) => h - 30 - p * (h - 48);
      adim.forEach((a, i) => {
        const X = 15 + i * bw + bw * 0.15, W = bw * 0.7;
        if (a.dus == null) { x.fillStyle = i ? ctx.css('--vurgu') : ctx.css('--ikinci'); x.fillRect(X, y(a.p), W, y(0) - y(a.p)); }
        else { x.fillStyle = ctx.css('--soluk'); x.globalAlpha = 0.7; x.fillRect(X, y(a.ust), W, y(a.p) - y(a.ust)); x.globalAlpha = 1; yazi(x, ctx, '−' + Math.round(a.dus * 100), X + W / 2, y(a.p) + 14, 10, ctx.css('--soluk'), 'center'); }
        yazi(x, ctx, a.ad, X + W / 2, h - 12, 10, ctx.css('--soluk'), 'center');
        if (a.dus == null) yazi(x, ctx, N.yuzde(a.p), X + W / 2, y(a.p) - 5, 11, null, 'center', 600);
      });
      if (!k.length) yazi(x, ctx, 'Seçim yok; bir head seç.', w / 2, 24, 12, ctx.css('--soluk'), 'center');
    } });

  K.ekle({ id: 'olc-04', sinif: 'olcum', ad: 'Terazi', tarif: 'Hedef ile rakip bir terazinin iki kefesinde. Kesim hedefin kefesini hafifletir; terazi rakibe yattığında tahmin değişmiştir.', yuk: 240,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), S = ctx.S(), cx = w / 2, cy = 70, L = Math.min(w * 0.36, 150), aci = Math.max(-0.45, Math.min(0.45, (r.rakipSonra - r.sonra) * 0.9));
      x.strokeStyle = ctx.css('--murekkep'); x.lineWidth = 3; x.beginPath(); x.moveTo(cx, cy); x.lineTo(cx, h - 26); x.moveTo(cx - 40, h - 26); x.lineTo(cx + 40, h - 26); x.stroke();
      const uc = (s) => [cx + Math.cos(aci) * L * s, cy + Math.sin(aci) * L * s];
      const [ax, ay] = uc(-1), [bx, by] = uc(1);
      x.beginPath(); x.moveTo(ax, ay); x.lineTo(bx, by); x.stroke();
      [[ax, ay, S.istem.hedef, r.sonra, ctx.css('--ikinci')], [bx, by, S.istem.rakip, r.rakipSonra, ctx.css('--vurgu')]].forEach(([X, Y, t, p, c]) => {
        x.lineWidth = 1; x.beginPath(); x.moveTo(X, Y); x.lineTo(X - 26, Y + 40); x.moveTo(X, Y); x.lineTo(X + 26, Y + 40); x.stroke();
        x.fillStyle = c; x.beginPath(); x.ellipse(X, Y + 42, 32, 7, 0, 0, T); x.fill();
        yazi(x, ctx, N.tok(t).trim(), X, Y + 66, 12, c, 'center', 700); yazi(x, ctx, N.yuzde(p), X, Y + 80, 11, ctx.css('--soluk'), 'center');
      });
      x.fillStyle = ctx.css('--murekkep'); x.beginPath(); x.arc(cx, cy, 5, 0, T); x.fill();
    } });

  K.ekle({ id: 'olc-05', sinif: 'olcum', ad: 'Manşet', tarif: 'Sayılar yazının kendisi: eski olasılık üstü çizili, yenisi büyük. Tahmin değiştiyse yeni sözcük manşete çıkar.', yuk: 240,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), S = ctx.S(), b = Math.min(64, w / 5.2);
      yazi(x, ctx, N.yuzde(r.once), 14, 26 + b * 0.5, b * 0.5, ctx.css('--soluk'), 'left', 700);
      const ow = x.measureText(N.yuzde(r.once)).width; x.strokeStyle = ctx.css('--vurgu'); x.lineWidth = 3; x.beginPath(); x.moveTo(12, 14 + b * 0.32); x.lineTo(16 + ow, 14 + b * 0.32); x.stroke();
      yazi(x, ctx, N.yuzde(r.sonra), 14, 34 + b * 1.4, b * 1.25, null, 'left', 800);
      yazi(x, ctx, r.degisti ? 'Model artık “' + N.tok(r.tahmin).trim() + '” diyor.' : 'Model hâlâ “' + N.tok(r.tahmin).trim() + '” diyor.', 14, h - 40, 17, r.degisti ? ctx.css('--vurgu') : ctx.css('--murekkep'), 'left', 700);
      yazi(x, ctx, (ctx.secili().size ? [...ctx.secili()].map(N.adK).join(' + ') + ' kapalı' : 'hiçbir şey kapalı değil') + ' · KL ' + N.sayi(r.kl, 3), 14, h - 16, 11, ctx.css('--soluk'));
    } });

  K.ekle({ id: 'olc-06', sinif: 'olcum', ad: 'Yüz nokta', tarif: 'Yüz noktanın kaçı dolu: olasılığın birim grafiği. Kırmızı halkalar kesimle kaybolan kısım; yanında eski olasılık üstü çizili, yenisi büyük, neyin kapalı olduğu ve yeni tahmin.', yuk: 260,
    ciz(x, w, h, ctx) { K.yuzNokta(x, w, h, ctx); } });
  // Yüz nokta + manşet: sol 10×10 nokta, sağda açıklamalı sütun. Seçtiklerin bölümü de bunu kullanır.
  K.yuzNokta = function (x, w, h, ctx) {
    const r = olc(ctx), S = ctx.S(), n = Math.round(r.sonra * 100), o = Math.round(r.once * 100), sag = Math.min(170, Math.max(130, w * 0.45));
    const s = Math.min((w - sag - 20) / 10, (h - 16) / 10), r0 = s * 0.36, renk = { dolu: ctx.css('--ikinci'), giden: '#ff3b30', yok: ctx.css('--cizgi') };
    for (let i = 0; i < 100; i++) {
      const X = 10 + (i % 10) * s + s / 2, Y = 8 + Math.floor(i / 10) * s + s / 2;
      x.beginPath(); x.arc(X, Y, r0, 0, T);
      if (i < Math.min(n, o)) { x.fillStyle = renk.dolu; x.fill(); } else if (i < o) { x.strokeStyle = renk.giden; x.lineWidth = 1.6; x.stroke(); } else if (i < n) { x.fillStyle = renk.dolu; x.fill(); } else { x.fillStyle = renk.yok; x.fill(); }
    }
    const X = 20 + 10 * s, en = w - X - 6;
    const kisa = (t, px) => { x.font = px; if (x.measureText(t).width <= en) return t; while (t.length > 1 && x.measureText(t + '…').width > en) t = t.slice(0, -1); return t + '…'; };
    // manşet: eski üstü çizili, yeni büyük
    yazi(x, ctx, N.yuzde(r.once), X, 24, 16, ctx.css('--soluk'), 'left', 700);
    const ow = x.measureText(N.yuzde(r.once)).width; x.strokeStyle = renk.giden; x.lineWidth = 2.5; x.beginPath(); x.moveTo(X - 2, 18); x.lineTo(X + ow + 2, 18); x.stroke();
    yazi(x, ctx, N.yuzde(r.sonra), X, 58, 32, null, 'left', 800);
    yazi(x, ctx, kisa('“' + S.istem.hedef.trim() + '” olasılığı', '400 11px ' + ctx.css('--f')), X, 74, 11, ctx.css('--soluk'));
    // açıklama
    const sat = (Y, tur, metin, kalin) => {
      x.beginPath(); x.arc(X + 5, Y - 4, 4.5, 0, T);
      if (tur === 'giden') { x.strokeStyle = renk.giden; x.lineWidth = 1.6; x.stroke(); } else { x.fillStyle = renk[tur]; x.fill(); }
      yazi(x, ctx, kisa(metin, (kalin || 400) + ' 11px ' + ctx.css('--f')), X + 15, Y, 11, tur === 'giden' ? renk.giden : null, 'left', kalin || 400);
    };
    sat(100, 'dolu', 'kalan: ' + Math.min(n, o) + (n > o ? ' (+' + (n - o) + ' arttı)' : ''));
    sat(118, 'giden', 'kesimle giden: ' + Math.max(0, o - n), 600);
    sat(136, 'yok', 'zaten yoktu: ' + (100 - Math.max(n, o)));
    const sec = [...ctx.secili()].map(N.adK);
    yazi(x, ctx, 'Kapalı', X, 164, 10, ctx.css('--soluk'), 'left', 600);
    yazi(x, ctx, kisa(sec.length ? sec.slice(0, 3).join(' + ') + (sec.length > 3 ? ' +' + (sec.length - 3) : '') : 'hiçbiri', '600 11px ' + ctx.css('--m')), X, 179, 11, renk.giden, 'left', 600);
    yazi(x, ctx, 'Tahmin', X, 204, 10, ctx.css('--soluk'), 'left', 600);
    yazi(x, ctx, kisa(N.tok(r.tahmin).trim() + (r.degisti ? '  · değişti' : '  · aynı'), '700 12px ' + ctx.css('--f')), X, 220, 12, r.degisti ? renk.giden : null, 'left', 700);
  };

  // ── Lens ve dikkat
  K.ekle({ id: 'lns-01', sinif: 'lens', ad: 'Lens merdiveni', tarif: 'Her katmanda modelin üç tahmini yan yana bir çubukta, payları kadar geniş; yanında oranları. Doğru sözcüğün hangi katmanda tepeye çıktığı ve rakibin ne kadar yakın olduğu okunur.', yuk: 360,
    ciz(x, w, h, ctx) { K.lensMerdiven(x, w, h, ctx); } });
  // Lens merdiveni: 1., 2., 3. tahmin yığılmış çubuk + oranlar. Hedef turkuaz, rakip turuncu, diğerleri gri.
  K.lensMerdiven = function (x, w, h, ctx) {
    const S = ctx.S(), ln = S.lens([...ctx.secili()]), rh = (h - 22) / S.L, bw = Math.max(70, Math.min(w * 0.36, w - 230)), yx = 40 + bw + 8;
    const renk = (tok, i) => (tok === S.istem.hedef ? ctx.css('--ikinci') : tok === S.istem.rakip ? ctx.css('--vurgu') : i === 0 ? ctx.css('--soluk') : ctx.css('--cizgi'));
    const fs = Math.max(9, Math.min(11, rh * 0.8));
    ln.forEach((row, l) => {
      const Y = 4 + l * rh;
      yazi(x, ctx, 'K' + l, 30, Y + rh * 0.78, fs, ctx.css('--soluk'), 'right');
      x.fillStyle = ctx.css('--cizgi'); x.globalAlpha = 0.35; x.fillRect(36, Y + 1, bw, rh - 2); x.globalAlpha = 1;
      let bx = 36;
      row.forEach((t, i) => { x.fillStyle = renk(t.tok, i); x.globalAlpha = i === 0 ? 1 : 0.75; x.fillRect(bx, Y + 1, Math.max(0, bw * t.p - 1), rh - 2); bx += bw * t.p; });
      x.globalAlpha = 1;
      let tx = yx;
      row.forEach((t, i) => {
        const s = N.tok(t.tok).trim() + ' ' + N.yuzde(t.p), ilk = i === 0;
        x.font = (ilk ? '700 ' : '400 ') + fs + 'px ' + ctx.css('--m');
        const sw = x.measureText(s).width;
        if (tx + sw > w - 4) return;
        const c = t.tok === S.istem.hedef ? ctx.css('--ikinci') : t.tok === S.istem.rakip ? ctx.css('--vurgu') : ilk ? ctx.css('--murekkep') : ctx.css('--soluk');
        x.fillStyle = c; x.textAlign = 'left'; x.fillText(s, tx, Y + rh * 0.78); tx += sw + 12;
      });
    });
    const Y = h - 6; x.font = '10px ' + ctx.css('--f');
    [[ctx.css('--ikinci'), N.tok(S.istem.hedef).trim() + ' (doğru)'], [ctx.css('--vurgu'), N.tok(S.istem.rakip).trim() + ' (rakip)'], [ctx.css('--soluk'), 'diğer']].reduce((X, [c, t]) => { x.fillStyle = c; x.fillRect(X, Y - 8, 9, 9); yazi(x, ctx, t, X + 13, Y, 10, ctx.css('--soluk')); return X + 22 + x.measureText(t).width; }, 36);
  };

  K.ekle({ id: 'lns-02', sinif: 'lens', ad: 'Lens eğrisi', tarif: 'Hedef ve rakibin katman boyunca olasılığı. Kesikli çizgiler kesimden önce; seçim değişince eğri nerede kırıldığını gösterir.', yuk: 260,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), a = S.lens([]), b = S.lens([...ctx.secili()]), X = (l) => 30 + l / (S.L - 1) * (w - 44), Y = (p) => h - 24 - p * (h - 40);
      x.strokeStyle = ctx.css('--cizgi'); x.lineWidth = 1; [0, 0.5, 1].forEach((p) => { x.beginPath(); x.moveTo(30, Y(p)); x.lineTo(w - 14, Y(p)); x.stroke(); yazi(x, ctx, N.yuzde(p), 26, Y(p) + 4, 10, ctx.css('--soluk'), 'right'); });
      const cizgi = (rows, tok, renk, kes) => { x.strokeStyle = renk; x.lineWidth = kes ? 1.5 : 2.5; x.setLineDash(kes ? [4, 4] : []); x.beginPath(); rows.forEach((row, l) => { const p = hedefP(row, tok); l ? x.lineTo(X(l), Y(p)) : x.moveTo(X(l), Y(p)); }); x.stroke(); x.setLineDash([]); };
      cizgi(a, S.istem.hedef, ctx.css('--ikinci'), true); cizgi(a, S.istem.rakip, ctx.css('--vurgu'), true);
      cizgi(b, S.istem.hedef, ctx.css('--ikinci')); cizgi(b, S.istem.rakip, ctx.css('--vurgu'));
      yazi(x, ctx, N.tok(S.istem.hedef).trim(), X(S.L - 1), Y(hedefP(b[S.L - 1], S.istem.hedef)) - 6, 11, ctx.css('--ikinci'), 'right', 600);
      yazi(x, ctx, N.tok(S.istem.rakip).trim(), X(S.L - 1), Y(hedefP(b[S.L - 1], S.istem.rakip)) - 6, 11, ctx.css('--vurgu'), 'right', 600);
      yazi(x, ctx, 'K0', 30, h - 8, 10, ctx.css('--soluk')); yazi(x, ctx, 'K' + (S.L - 1), w - 14, h - 8, 10, ctx.css('--soluk'), 'right');
    } });

  K.ekle({ id: 'lns-03', sinif: 'lens', ad: 'Lens şeridi', tarif: 'Sütunlar katman, satırlar hedef ve rakip; üst yarı kesimden önce, alt yarı sonra. Fark, iki şerit arasındaki renk kaybıdır.', yuk: 200,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), a = S.lens([]), b = S.lens([...ctx.secili()]), rn = ctx.rampa('magma'), cw = (w - 90) / S.L, rh = (h - 30) / 4;
      [['önce · hedef', a, S.istem.hedef], ['önce · rakip', a, S.istem.rakip], ['sonra · hedef', b, S.istem.hedef], ['sonra · rakip', b, S.istem.rakip]].forEach(([ad, rows, tok], i) => {
        const Y = 4 + i * rh + (i > 1 ? 10 : 0);
        yazi(x, ctx, ad, 84, Y + rh * 0.65, 10, ctx.css('--soluk'), 'right');
        rows.forEach((row, l) => { x.fillStyle = rn(Math.min(1, hedefP(row, tok) * 1.4)); x.fillRect(90 + l * cw + 0.5, Y + 0.5, cw - 1, rh - 1); });
      });
    } });

  function dikkatHead(ctx) { const S = ctx.S(); return [...ctx.secili()][0] || S.sicak[0].k; }
  K.ekle({ id: 'dkt-01', sinif: 'lens', ad: 'Dikkat matrisi', tarif: 'Seçili ilk head’in dikkat deseni: satır sorgu, sütun anahtar belirteci. Köşegene yakın desen önceki belirtece bakar.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), k = dikkatHead(ctx), { l, h: hh } = N.ac(k), at = S.dikkat(l, hh), pad = 70, s = Math.min((w - pad - 10) / S.T, (h - 40) / S.T), rn = ctx.rampa('buz');
      for (let i = 0; i < S.T; i++) {
        yazi(x, ctx, N.tok(S.istem.tokenlar[i]).slice(0, 8), pad - 6, 8 + i * s + s * 0.62, 11, ctx.css('--soluk'), 'right');
        for (let j = 0; j <= i; j++) { x.fillStyle = rn(1 - at.m[i][j]); x.fillRect(pad + j * s + 1, 8 + i * s + 1, s - 2, s - 2); }
      }
      yazi(x, ctx, N.adK(k) + ' · ' + at.tur, pad, h - 10, 12, null, 'left', 600);
    } });

  K.ekle({ id: 'dkt-02', sinif: 'lens', ad: 'Dikkat yayları', tarif: 'Belirteçler bir çizgide; son belirtecin hangilerine baktığı yaylarla. Kalın yay güçlü dikkat. Soluk yaylar öbür sorgular.', yuk: 240,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), k = dikkatHead(ctx), { l, h: hh } = N.ac(k), at = S.dikkat(l, hh), X = (i) => 24 + i * (w - 48) / Math.max(1, S.T - 1), Y = h - 40;
      for (let i = 0; i < S.T; i++) for (let j = 0; j <= i; j++) {
        if (i === j) continue;
        const son = i === S.T - 1, wgt = at.m[i][j];
        x.strokeStyle = son ? ctx.css('--vurgu') : ctx.css('--soluk'); x.globalAlpha = son ? 0.25 + 0.75 * wgt : 0.12 * wgt * 3; x.lineWidth = son ? 1 + 10 * wgt : 1;
        x.beginPath(); x.moveTo(X(i), Y); x.quadraticCurveTo((X(i) + X(j)) / 2, Y - Math.abs(X(i) - X(j)) * 0.55, X(j), Y); x.stroke();
      }
      x.globalAlpha = 1;
      for (let i = 0; i < S.T; i++) { x.fillStyle = ctx.css('--murekkep'); x.beginPath(); x.arc(X(i), Y, 3, 0, T); x.fill(); yazi(x, ctx, N.tok(S.istem.tokenlar[i]), X(i), Y + 18, 12, i === S.T - 1 ? ctx.css('--vurgu') : null, 'center', i === S.T - 1 ? 700 : 400); }
      yazi(x, ctx, N.adK(k) + ' · ' + at.tur, 12, h - 6, 11, ctx.css('--soluk'));
    } });

  K.ekle({ id: 'dkt-03', sinif: 'lens', ad: 'Desen albümü', tarif: 'Altı sıcak head’in desenleri yan yana küçük resimler. Tıkla, o head seçime girer; benzer desenler gruplanır.', yuk: 280,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), headler = [...new Set([...ctx.secili(), ...S.sicak.map((q) => q.k)])].slice(0, 6), sut = w > 420 ? 3 : 2, sat = Math.ceil(headler.length / sut);
      const kw = w / sut, kh = (h - 4) / sat, rn = ctx.rampa('yosun'), kutu = [];
      headler.forEach((k, i) => {
        const { l, h: hh } = N.ac(k), at = S.dikkat(l, hh), X = (i % sut) * kw + 8, Y = Math.floor(i / sut) * kh + 4, s = Math.min((kw - 16) / S.T, (kh - 26) / S.T);
        for (let a = 0; a < S.T; a++) for (let b = 0; b <= a; b++) { x.fillStyle = rn(at.m[a][b]); x.fillRect(X + b * s, Y + a * s, s - 1, s - 1); }
        if (ctx.secili().has(k)) { x.strokeStyle = ctx.css('--vurgu'); x.lineWidth = 2; x.strokeRect(X - 3, Y - 3, S.T * s + 5, S.T * s + 5); }
        yazi(x, ctx, N.adK(k) + ' · ' + at.tur, X, Y + S.T * s + 14, 10.5, ctx.css('--soluk'));
        kutu.push([X, Y, kw - 8, kh - 4, k]);
      });
      return (px, py) => { const q = kutu.find(([a, b, c, d]) => px >= a && px < a + c && py >= b && py < b + d); return q ? q[4] : null; };
    } });
})();
