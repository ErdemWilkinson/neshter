/* Sınıf 2: 3B sistemler. ciz(x, w, h, ctx, kam, t): kam sürüklemeyle değişir; P = ctx.P(kam, w, h, ölçek) dünya → ekran. */
(function () {
  const K = window.Katalog, N = window.Nester, T = Math.PI * 2;
  function cokgen(x, n, fill, stroke, lw) { x.beginPath(); n.forEach((p, i) => (i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]))); x.closePath(); if (fill) { x.fillStyle = fill; x.fill(); } if (stroke) { x.strokeStyle = stroke; x.lineWidth = lw || 1; x.stroke(); } }
  function golge(renk, f) { // rgb(...) → koyulaştır
    const m = renk.match(/\d+/g); if (!m) return renk;
    return 'rgb(' + m.slice(0, 3).map((v) => Math.round(v * f)).join(',') + ')';
  }
  function enYakin(liste, px, py, yaricap) {
    let en = null, ed = yaricap, ez = -Infinity;
    for (const q of liste) { const dd = Math.hypot(q[0] - px, q[1] - py); if (dd < ed || (dd < yaricap && Math.abs(dd - ed) < 2 && q[2] > ez)) { ed = dd; en = q[3]; ez = q[2]; } }
    return en;
  }
  function akis(G, kesik) {
    const ham = (kes) => { const a = new Map([['girdi', 1]]); for (const d of G.dugumler) { if (d.tur === 'girdi') continue; let s = 0; for (const b of G.baglar) if (b.h === d.id && !kes.has(b.id)) s += b.w * (a.get(b.k) || 0); a.set(d.id, s); } return a; };
    const tam = ham(new Set()), simdi = ham(kesik), oran = new Map();
    for (const [id, v] of simdi) oran.set(id, tam.get(id) ? v / tam.get(id) : 0);
    return oran;
  }

  K.ekle({ id: 'uc-01', sinif: 'uc', ad: 'Sütun ormanı', tarif: 'Her head bir sütun, yüksekliği gücü. Sıcaklar orman içinde kule gibi yükselir; arkadakileri görmek için döndür.', yuk: 360, uc: true,
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), rz = 0.7, s = Math.min(w * 0.6 / S.H, h * 1.0 / (S.L * rz)), P = ctx.P(kam, w, h, s, 30), rn = ctx.rampa('viridis'), Hm = S.H * 0.45;
      const bar = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const X = hh - (S.H - 1) / 2, Z = (l - (S.L - 1) / 2) * rz, v = S.g(l, hh), Y = 0.05 + v * Hm, a = 0.36, b = 0.3;
        bar.push({ X, Z, Y, v, a, b, k: N.anahtar(l, hh), z: P(X, Y / 2, Z)[2] });
      }
      bar.sort((p, q) => p.z - q.z);
      const ust = [];
      for (const q of bar) {
        const r = rn(q.v), k = [[-q.a, -q.b], [q.a, -q.b], [q.a, q.b], [-q.a, q.b]];
        const alt = k.map(([dx, dz]) => P(q.X + dx, 0, q.Z + dz)), tep = k.map(([dx, dz]) => P(q.X + dx, q.Y, q.Z + dz));
        const yuz = [0, 1, 2, 3].map((i) => ({ i, z: (alt[i][2] + alt[(i + 1) % 4][2]) / 2 })).sort((p, q2) => p.z - q2.z);
        for (const { i } of yuz) cokgen(x, [alt[i], alt[(i + 1) % 4], tep[(i + 1) % 4], tep[i]], golge(r, i % 2 ? 0.62 : 0.78));
        const sec = ctx.secili().has(q.k);
        cokgen(x, tep, r, sec ? '#ff3b30' : null, 2);
        const m = P(q.X, q.Y, q.Z); ust.push([m[0], m[1], m[2], q.k]);
      }
      return (px, py) => enYakin(ust, px, py, 9);
    } });

  K.ekle({ id: 'uc-02', sinif: 'uc', ad: 'Tel kafes arazi', tarif: 'Güç bir arazinin yüksekliği. Çizgiler katman ve head boyunca ilerler; tepeler sıcak, ovalar sessiz.', yuk: 360, uc: true, kam: { yaw: -0.7, pitch: 0.55 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), rz = 0.75, s = Math.min(w * 0.62 / S.H, h * 1.05 / (S.L * rz)), P = ctx.P(kam, w, h, s, 26), Hm = S.H * 0.4, rn = ctx.rampa('kor');
      const V = (l, hh) => P(hh - (S.H - 1) / 2, S.g(l, hh) * Hm, (l - (S.L - 1) / 2) * rz);
      const nok = [];
      x.lineWidth = 1.2;
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const a = V(l, hh); nok.push([a[0], a[1], a[2], N.anahtar(l, hh)]);
        if (hh + 1 < S.H) { const b = V(l, hh + 1); x.strokeStyle = rn(0.2 + 0.8 * (S.g(l, hh) + S.g(l, hh + 1)) / 2); x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke(); }
        if (l + 1 < S.L) { const b = V(l + 1, hh); x.strokeStyle = rn(0.2 + 0.8 * (S.g(l, hh) + S.g(l + 1, hh)) / 2); x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke(); }
      }
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) { const a = V(l, hh); x.fillStyle = ctx.css('--ikinci'); x.beginPath(); x.arc(a[0], a[1], 5, 0, T); x.fill(); x.fillStyle = ctx.css('--murekkep'); x.textAlign = 'center'; x.fillText(N.adK(k), a[0], a[1] - 9); } }
      return (px, py) => enYakin(nok, px, py, 8);
    } });

  K.ekle({ id: 'uc-03', sinif: 'uc', ad: 'Katman plakaları', tarif: 'Her katman ayrı, yarı saydam bir plaka; plakalar aralıklı üst üste. Sıcak hücreler parlar, soğuklar neredeyse görünmez.', yuk: 380, uc: true, kam: { yaw: -0.6, pitch: 0.42 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), ara = 0.55, s = Math.min(w / (S.H * 1.25), h / (S.L * ara * 1.25)), P = ctx.P(kam, w, h, s), rn = ctx.rampa('buz'), mur = ctx.css('--murekkep');
      const hucre = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const X = hh - (S.H - 1) / 2, Y = ((S.L - 1) / 2 - l) * ara, d = 0.46;
        const k = [P(X - d, Y, -1.4), P(X + d, Y, -1.4), P(X + d, Y, 1.4), P(X - d, Y, 1.4)];
        hucre.push({ k, z: (k[0][2] + k[2][2]) / 2, v: S.g(l, hh), a: N.anahtar(l, hh), m: P(X, Y, 0) });
      }
      hucre.sort((p, q) => p.z - q.z);
      for (const q of hucre) {
        x.globalAlpha = 0.08 + 0.86 * q.v * q.v; cokgen(x, q.k, rn(1 - q.v));
        if (ctx.secili().has(q.a)) { x.globalAlpha = 1; cokgen(x, q.k, null, '#ff6b00', 2); }
      }
      x.globalAlpha = 0.5; x.fillStyle = mur; x.textAlign = 'right';
      for (let l = 0; l < S.L; l += 4) { const p = P(-(S.H - 1) / 2 - 0.9, ((S.L - 1) / 2 - l) * ara, 0); x.fillText('K' + l, p[0], p[1] + 4); }
      x.globalAlpha = 1;
      return (px, py) => enYakin(hucre.map((q) => [q.m[0], q.m[1], q.m[2], q.a]), px, py, 7);
    } });

  K.ekle({ id: 'uc-04', sinif: 'uc', ad: 'Yükselen bağlar', tarif: 'Uygulamadaki 3B’nin özü: zemin ısı haritası, en sıcak head’ler kendi hücrelerinin üstünde yükselir, bağlar aralarında yay çizer.', yuk: 380, uc: true, kam: { yaw: -0.25, pitch: 0.9 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), G = N.bagGrafi(S), rz = 0.62, s = Math.min(w * 0.62 / S.H, h * 1.0 / ((S.L + 4) * rz)), P = ctx.P(kam, w, h, s, 40), rn = ctx.rampaTema();
      const hx = (hh) => hh - (S.H - 1) / 2, lz = (l) => (l - (S.L - 1) / 2) * rz, Yt = S.L * rz * 0.55;
      const zemin = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) { const k = [P(hx(hh) - 0.47, 0, lz(l) - 0.29), P(hx(hh) + 0.47, 0, lz(l) - 0.29), P(hx(hh) + 0.47, 0, lz(l) + 0.29), P(hx(hh) - 0.47, 0, lz(l) + 0.29)]; zemin.push({ k, z: k[0][2], v: S.g(l, hh) }); }
      zemin.sort((p, q) => p.z - q.z); x.globalAlpha = 0.7; zemin.forEach((q) => cokgen(x, q.k, rn(q.v))); x.globalAlpha = 1;
      const kon = new Map(G.dugumler.map((n) => { const yuk = n.tur === 'girdi' ? 0.15 : n.tur === 'cikti' ? 1 : 0.25 + 0.75 * n.l / S.L; return [n.id, { z: P(hx(n.h), 0, lz(n.l)), p: P(hx(n.h), yuk * Yt, lz(n.l)), n }]; }));
      const oran = akis(G, new Set(G.baglar.filter((b) => ctx.secili().has(b.h)).map((b) => b.id)));
      for (const b of G.baglar) {
        const A = kon.get(b.k).p, B = kon.get(b.h).p, kes = ctx.secili().has(b.h);
        x.strokeStyle = kes ? ctx.css('--vurgu') : ctx.css('--ikinci'); x.globalAlpha = kes ? 0.9 : 0.6; x.lineWidth = kes ? 1.5 : 1 + 6 * b.w * (oran.get(b.k) || 0); x.setLineDash(kes ? [4, 4] : []);
        x.beginPath(); x.moveTo(A[0], A[1]); x.quadraticCurveTo((A[0] + B[0]) / 2, Math.min(A[1], B[1]) - 26, B[0], B[1]); x.stroke();
      }
      x.setLineDash([]); x.globalAlpha = 1;
      const nok = [];
      for (const { z, p, n } of kon.values()) {
        x.strokeStyle = ctx.css('--soluk'); x.lineWidth = 1; x.beginPath(); x.moveTo(z[0], z[1]); x.lineTo(p[0], p[1]); x.stroke();
        const sec = ctx.secili().has(n.id); x.fillStyle = sec ? ctx.css('--vurgu') : n.tur === 'cikti' ? ctx.css('--ikinci') : ctx.css('--murekkep');
        x.beginPath(); x.arc(p[0], p[1], n.tur === 'head' ? 6 : 7, 0, T); x.fill();
        x.fillStyle = ctx.css('--murekkep'); x.textAlign = 'center'; x.fillText(n.tur === 'cikti' ? n.ad + ' ' + N.yuzde(oran.get('cikti') || 0) : n.ad, p[0], p[1] - 11);
        if (n.tur === 'head') nok.push([p[0], p[1], p[2], n.id]);
      }
      return (px, py) => enYakin(nok, px, py, 10);
    } });

  K.ekle({ id: 'uc-05', sinif: 'uc', ad: 'Küre', tarif: 'Head’ler bir kürenin yüzeyinde: enlem katman (K0 kuzey kutbu), boylam head. Arka yüz soluk; döndürünce öne gelir.', yuk: 360, uc: true, kam: { yaw: 0.3, pitch: 0.35 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), R = 1, P = ctx.P(kam, w, h, Math.min(w, h) * 0.4), rn = ctx.rampa('magma'), mur = ctx.css('--murekkep');
      const pos = (l, hh) => { const lat = (0.5 - l / (S.L - 1)) * Math.PI * 0.86, lon = hh / S.H * T; return [Math.cos(lat) * Math.cos(lon) * R, Math.sin(lat) * R, Math.cos(lat) * Math.sin(lon) * R]; };
      x.strokeStyle = mur; x.globalAlpha = 0.12; x.lineWidth = 1;
      for (let l = 0; l < S.L; l += 3) { x.beginPath(); for (let i = 0; i <= 48; i++) { const lat = (0.5 - l / (S.L - 1)) * Math.PI * 0.86, lon = i / 48 * T, p = P(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)); i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]); } x.stroke(); }
      x.globalAlpha = 1;
      const nok = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) { const p = P(...pos(l, hh)); nok.push([p[0], p[1], p[2], N.anahtar(l, hh), S.g(l, hh)]); }
      nok.sort((a, b) => a[2] - b[2]);
      for (const [px, py, z, k, v] of nok) {
        const on = z > 0; x.globalAlpha = on ? 1 : 0.22;
        x.fillStyle = rn(0.15 + 0.85 * v); x.beginPath(); x.arc(px, py, 1.5 + 5 * v, 0, T); x.fill();
        if (ctx.secili().has(k)) { x.strokeStyle = '#3ad1ff'; x.lineWidth = 2; x.beginPath(); x.arc(px, py, 4 + 5 * v, 0, T); x.stroke(); }
      }
      x.globalAlpha = 1;
      return (px, py) => enYakin(nok.filter((q) => q[2] > -0.2), px, py, 8);
    } });

  K.ekle({ id: 'uc-06', sinif: 'uc', ad: 'Tüp', tarif: 'Model soldan sağa bir tüp: eksen boyunca katmanlar, çevresinde head’ler. Bir head’in katmanlar boyu izi bir çizgi olarak okunur.', yuk: 330, uc: true, kam: { yaw: 0.25, pitch: 0.3 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), dx = 0.32, R = 1.1, s = Math.min(w / (S.L * dx * 1.35), h / (R * 3.2)), P = ctx.P(kam, w, h, s), rn = ctx.rampa('viridis'), da = T / S.H;
      const q = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const X0 = (l - S.L / 2) * dx, X1 = X0 + dx * 0.92, a0 = hh * da, a1 = a0 + da * 0.92;
        const k = [P(X0, Math.cos(a0) * R, Math.sin(a0) * R), P(X1, Math.cos(a0) * R, Math.sin(a0) * R), P(X1, Math.cos(a1) * R, Math.sin(a1) * R), P(X0, Math.cos(a1) * R, Math.sin(a1) * R)];
        const m = P(X0 + dx / 2, Math.cos(a0 + da / 2) * R, Math.sin(a0 + da / 2) * R);
        q.push({ k, m, z: m[2], v: S.g(l, hh), a: N.anahtar(l, hh) });
      }
      q.sort((a, b) => a.z - b.z);
      for (const c of q) { x.globalAlpha = c.z < 0 ? 0.35 : 1; cokgen(x, c.k, rn(c.v), ctx.secili().has(c.a) ? '#ff3b30' : null, 2); }
      x.globalAlpha = 1; x.fillStyle = ctx.css('--soluk'); x.textAlign = 'center';
      for (const l of [0, S.L - 1]) { const p = P((l - S.L / 2) * dx + dx / 2, -R - 0.45, 0); x.fillText('K' + l, p[0], p[1]); }
      return (px, py) => enYakin(q.filter((c) => c.z >= 0).map((c) => [c.m[0], c.m[1], c.m[2], c.a]), px, py, 9);
    } });

  K.ekle({ id: 'uc-07', sinif: 'uc', ad: 'Parçacık tüneli', tarif: 'Bilgi girdiden çıktıya akan parçacıklar. Bir head’e tıkla, kapanır: içinden geçen akış durur, çıktıdaki parçacıklar seyrelir.', yuk: 360, uc: true, anim: true, kam: { yaw: 0.5, pitch: 0.28 },
      ipucu: 'sürükle: döndür · bir düğüme tıkla: kapat',
    ciz(x, w, h, ctx, kam, t) {
      const S = ctx.S(), G = N.bagGrafi(S), dx = 0.22, R = 1, s = Math.min(w / ((S.L + 4) * dx * 1.2), h / 3), P = ctx.P(kam, w, h, s);
      const pos = (n) => { const X = (n.l - S.L / 2) * dx; if (n.tur !== 'head') return [X, 0, 0]; const a = n.h / S.H * T; return [X, Math.cos(a) * R, Math.sin(a) * R]; };
      const kesik = new Set(G.baglar.filter((b) => ctx.secili().has(b.h)).map((b) => b.id)), oran = akis(G, kesik);
      x.strokeStyle = ctx.css('--soluk'); x.globalAlpha = 0.18;
      for (let i = 0; i <= 6; i++) { const X = (-1.6 + i / 6 * (S.L + 2.2) - S.L / 2) * dx; x.beginPath(); for (let j = 0; j <= 32; j++) { const a = j / 32 * T, p = P(X, Math.cos(a) * R, Math.sin(a) * R); j ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]); } x.stroke(); }
      x.globalAlpha = 1;
      const D = new Map(G.dugumler.map((n) => [n.id, n]));
      for (const b of G.baglar) {
        const A = pos(D.get(b.k)), B = pos(D.get(b.h)), kes = kesik.has(b.id), akan = kes ? 0 : (oran.get(b.k) || 0);
        const a = P(...A), bb = P(...B);
        x.strokeStyle = kes ? ctx.css('--vurgu') : ctx.css('--cizgi'); x.lineWidth = 1; x.globalAlpha = 0.8; x.setLineDash(kes ? [3, 4] : []);
        x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(bb[0], bb[1]); x.stroke(); x.setLineDash([]);
        const say = Math.round(2 + 9 * b.w * akan);
        x.fillStyle = ctx.css('--ikinci');
        for (let i = 0; i < say; i++) {
          const f = ((t / 2400) * (0.6 + b.w) + i / say + (N.hash(b.id) % 100) / 100) % 1;
          const p = P(A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f);
          x.globalAlpha = 0.5 + 0.5 * akan; x.beginPath(); x.arc(p[0], p[1], 2, 0, T); x.fill();
        }
      }
      x.globalAlpha = 1;
      const nok = [];
      for (const n of G.dugumler) {
        const p = P(...pos(n)), sec = ctx.secili().has(n.id);
        x.fillStyle = sec ? ctx.css('--vurgu') : n.tur === 'cikti' ? ctx.css('--ikinci') : ctx.css('--murekkep');
        x.beginPath(); x.arc(p[0], p[1], n.tur === 'cikti' ? 4 + 8 * (oran.get('cikti') || 0) : 5, 0, T); x.fill();
        x.fillStyle = ctx.css('--soluk'); x.textAlign = 'center'; x.fillText(n.tur === 'cikti' ? n.ad + ' ' + N.yuzde(oran.get('cikti') || 0) : n.ad, p[0], p[1] - 10);
        if (n.tur === 'head') nok.push([p[0], p[1], p[2], n.id]);
      }
      return (px, py) => enYakin(nok, px, py, 10);
    } });

  K.ekle({ id: 'uc-08', sinif: 'uc', ad: 'Sarmal merdiven', tarif: 'Katmanlar bir döner merdivenin basamakları; her basamak içten dışa head’ler. Merdiveni dolaşarak katmanları sırayla gezersin.', yuk: 380, uc: true, kam: { yaw: -0.4, pitch: 0.5 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), adim = 0.16, r0 = 0.35, dr = 0.12, P = ctx.P(kam, w, h, Math.min(w, h) * 0.24), rn = ctx.rampa('toprak');
      const blok = [];
      for (let l = 0; l < S.L; l++) {
        const a = l * (T / 12), Y = ((S.L - 1) / 2 - l) * adim, da = T / 15;
        for (let hh = 0; hh < S.H; hh++) {
          const ri = r0 + hh * dr, ro = ri + dr * 0.9;
          const k = [[ri, a], [ro, a], [ro, a + da], [ri, a + da]].map(([r, aa]) => P(Math.cos(aa) * r, Y, Math.sin(aa) * r));
          const m = P(Math.cos(a + da / 2) * (ri + dr / 2), Y, Math.sin(a + da / 2) * (ri + dr / 2));
          blok.push({ k, m, z: m[2], v: S.g(l, hh), a: N.anahtar(l, hh) });
        }
      }
      blok.sort((p, q) => p.z - q.z);
      for (const b of blok) cokgen(x, b.k, rn(b.v), ctx.secili().has(b.a) ? '#2b59c3' : 'rgba(0,0,0,0.12)', ctx.secili().has(b.a) ? 2.2 : 0.6);
      x.strokeStyle = ctx.css('--soluk'); x.lineWidth = 2; const u = P(0, (S.L - 1) / 2 * adim + 0.2, 0), a2 = P(0, -(S.L - 1) / 2 * adim - 0.2, 0); x.beginPath(); x.moveTo(u[0], u[1]); x.lineTo(a2[0], a2[1]); x.stroke();
      return (px, py) => enYakin(blok.map((b) => [b.m[0], b.m[1], b.m[2], b.a]), px, py, 7);
    } });
})();
