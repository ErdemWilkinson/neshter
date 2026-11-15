/* İkinci dalga: her sınıfa yeni varyantlar. */
(function () {
  const K = window.Katalog, N = window.Nester, el = N.el, T = Math.PI * 2;
  const yazi = (x, ctx, t, X, Y, boy, renk, hiza, kalin) => { x.font = (kalin || 400) + ' ' + boy + 'px ' + ctx.css('--f'); x.fillStyle = renk || ctx.css('--murekkep'); x.textAlign = hiza || 'left'; x.fillText(t, X, Y); };
  const yakin = (liste, px, py, r) => { let en = null, ed = r; for (const q of liste) { const d = Math.hypot(q[0] - px, q[1] - py); if (d < ed) { ed = d; en = q[2]; } } return en; };
  const kutuBul = (kutu) => (px, py) => { const q = kutu.find(([a, b, c, d]) => px >= a && px < a + c && py >= b && py < b + d); return q ? q[4] : null; };
  function cokgen(x, n, fill, stroke, lw) { x.beginPath(); n.forEach((p, i) => (i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]))); x.closePath(); if (fill) { x.fillStyle = fill; x.fill(); } if (stroke) { x.strokeStyle = stroke; x.lineWidth = lw || 1; x.stroke(); } }
  const hp = (row, tok) => (row.find((q) => q.tok === tok) || { p: 0 }).p;

  // ═══ Isı haritaları
  K.ekle({ id: 'isi-13', sinif: 'isi', ad: 'Kıvılcım satırları', tarif: 'Her katman bir satır küçük çubuk; çubuk boyu güç. Barkod gibi okunur, renk gerekmez; dar ekranda bile ayrışır.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), rh = (h - 6) / S.L, cw = (w - 36) / S.H, kutu = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const v = S.g(l, hh), X = 32 + hh * cw, Y = 3 + l * rh, bh = Math.max(1, (rh - 2) * v), k = N.anahtar(l, hh);
        x.fillStyle = ctx.secili().has(k) ? ctx.css('--vurgu') : ctx.css('--murekkep'); x.globalAlpha = ctx.secili().has(k) ? 1 : 0.8;
        x.fillRect(X + cw * 0.2, Y + rh - 1 - bh, cw * 0.6, bh); kutu.push([X, Y, cw, rh, k]);
      }
      x.globalAlpha = 1; for (let l = 0; l < S.L; l += 4) yazi(x, ctx, 'K' + l, 28, 3 + (l + 0.85) * rh, 10, ctx.css('--soluk'), 'right');
      return kutuBul(kutu);
    } });

  K.ekle({ id: 'isi-14', sinif: 'isi', ad: 'Boyutlu kareler', tarif: 'Kare büyüklüğü güç, renk derinlik (sığ katman sarı, derin katman mor). İki bilgi tek hücrede.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), g = ctx.izgara(w, h, 30, 4), rn = ctx.rampa(['#f9c74f', '#f3722c', '#9d4edd', '#3c096c']);
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const v = S.g(l, hh), m = Math.min(g.cw, g.ch) * (0.15 + 0.85 * Math.sqrt(v)), cx = 30 + (hh + 0.5) * g.cw, cy = 4 + (l + 0.5) * g.ch;
        x.fillStyle = rn(l / (S.L - 1)); x.fillRect(cx - m / 2, cy - m / 2, m, m);
        if (ctx.secili().has(N.anahtar(l, hh))) { x.strokeStyle = ctx.css('--murekkep'); x.lineWidth = 2; x.strokeRect(cx - g.cw / 2 + 1, cy - g.ch / 2 + 1, g.cw - 2, g.ch - 2); }
      }
      return g.hucre;
    } });

  K.ekle({ id: 'isi-15', sinif: 'isi', ad: 'Head izleri', tarif: 'Paralel koordinatlar: her head bir çizgi, soldan sağa katmanlar. Bir head’in derinlik boyunca nerede parladığı izinden okunur.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), X = (l) => 30 + l / (S.L - 1) * (w - 44), Y = (v) => h - 22 - v * (h - 36), son = [];
      for (let hh = 0; hh < S.H; hh++) {
        const vurgu = [...ctx.secili()].some((k) => N.ac(k).h === hh);
        x.strokeStyle = vurgu ? ctx.css('--vurgu') : ctx.css('--soluk'); x.globalAlpha = vurgu ? 1 : 0.35; x.lineWidth = vurgu ? 2.2 : 1;
        x.beginPath(); for (let l = 0; l < S.L; l++) { const p = [X(l), Y(S.g(l, hh))]; l ? x.lineTo(...p) : x.moveTo(...p); son.push([p[0], p[1], N.anahtar(l, hh)]); } x.stroke();
      }
      x.globalAlpha = 1;
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L) { x.fillStyle = ctx.css('--vurgu'); x.beginPath(); x.arc(X(l), Y(S.g(l, hh)), 4, 0, T); x.fill(); } }
      yazi(x, ctx, 'K0', 30, h - 6, 10, ctx.css('--soluk')); yazi(x, ctx, 'K' + (S.L - 1), w - 14, h - 6, 10, ctx.css('--soluk'), 'right');
      return (px, py) => yakin(son, px, py, 8);
    } });

  K.ekle({ id: 'isi-16', sinif: 'isi', ad: 'Kutu grafikleri', tarif: 'Her katmanın head güçlerinin dağılımı: kutu ortadaki yarı, bıyıklar uçlar; dışarıda kalan noktalar sıcak head’ler. Noktaya tıkla.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), cw = (w - 40) / S.L, Y = (v) => h - 22 - v * (h - 34), nok = [];
      x.strokeStyle = ctx.css('--cizgi'); x.beginPath(); x.moveTo(34, Y(0)); x.lineTo(w - 6, Y(0)); x.stroke();
      for (let l = 0; l < S.L; l++) {
        const v = [...Array(S.H).keys()].map((hh) => [S.g(l, hh), hh]).sort((a, b) => a[0] - b[0]), q = (f) => v[Math.min(S.H - 1, Math.floor(f * (S.H - 1)))][0];
        const q1 = q(0.25), md = q(0.5), q3 = q(0.75), iqr = q3 - q1, X = 36 + (l + 0.5) * cw, bw = cw * 0.55;
        x.strokeStyle = ctx.css('--murekkep'); x.lineWidth = 1;
        x.beginPath(); x.moveTo(X, Y(v[0][0])); x.lineTo(X, Y(Math.min(q3 + 1.5 * iqr, v[S.H - 1][0]))); x.stroke();
        x.fillStyle = ctx.css('--kart'); x.fillRect(X - bw / 2, Y(q3), bw, Y(q1) - Y(q3)); x.strokeRect(X - bw / 2, Y(q3), bw, Y(q1) - Y(q3));
        x.strokeStyle = ctx.css('--ikinci'); x.lineWidth = 2; x.beginPath(); x.moveTo(X - bw / 2, Y(md)); x.lineTo(X + bw / 2, Y(md)); x.stroke();
        for (const [val, hh] of v) if (val > q3 + 1.5 * iqr) { const k = N.anahtar(l, hh); x.fillStyle = ctx.secili().has(k) ? ctx.css('--vurgu') : ctx.css('--murekkep'); x.beginPath(); x.arc(X, Y(val), ctx.secili().has(k) ? 4.5 : 3, 0, T); x.fill(); nok.push([X, Y(val), k]); }
        if (l % 4 === 0) yazi(x, ctx, 'K' + l, X, h - 6, 10, ctx.css('--soluk'), 'center');
      }
      return (px, py) => yakin(nok, px, py, 8);
    } });

  K.ekle({ id: 'isi-17', sinif: 'isi', ad: 'Gül kadranları', tarif: 'Her katman küçük bir gül: yapraklar head, yaprak boyu güç. Yirmi dört kadran yan yana; tek yapraklı katman bir bakışta seçilir.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), sut = Math.max(4, Math.round(Math.sqrt(S.L * w / h))), sat = Math.ceil(S.L / sut), kw = w / sut, kh = h / sat, R = Math.min(kw, kh) / 2 - 8, rn = ctx.rampa('viridis'), nok = [];
      for (let l = 0; l < S.L; l++) {
        const cx = (l % sut + 0.5) * kw, cy = (Math.floor(l / sut) + 0.5) * kh - 4, da = T / S.H;
        for (let hh = 0; hh < S.H; hh++) {
          const v = S.g(l, hh), a0 = -Math.PI / 2 + hh * da, r = R * (0.12 + 0.88 * v), k = N.anahtar(l, hh);
          x.beginPath(); x.moveTo(cx, cy); x.arc(cx, cy, r, a0, a0 + da * 0.9); x.closePath(); x.fillStyle = rn(v); x.fill();
          if (ctx.secili().has(k)) { x.strokeStyle = '#ff3b30'; x.lineWidth = 1.6; x.stroke(); }
          nok.push([cx + Math.cos(a0 + da / 2) * r * 0.7, cy + Math.sin(a0 + da / 2) * r * 0.7, k]);
        }
        yazi(x, ctx, 'K' + l, cx, cy + R + 11, 9.5, ctx.css('--soluk'), 'center');
      }
      return (px, py) => yakin(nok, px, py, 7);
    } });

  K.ekle({ id: 'isi-18', sinif: 'isi', ad: 'Gölgeli kabartma', tarif: 'Değerler bir kabartma haritası gibi, sol üstten ışıklandırılmış. Tepeler gölge düşürür; düz bölgeler sakin kalır.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), sol = 30, W = w - sol, H = h - 4, adim = 3, nx = Math.ceil(W / adim), ny = Math.ceil(H / adim);
      const f = (fx, fy) => { const gx = Math.max(0, Math.min(S.H - 1.001, fx * S.H - 0.5)), gy = Math.max(0, Math.min(S.L - 1.001, fy * S.L - 0.5)), i = Math.floor(gx), j = Math.floor(gy), u = gx - i, v = gy - j; return S.g(j, i) * (1 - u) * (1 - v) + S.g(j, i + 1) * u * (1 - v) + S.g(j + 1, i) * (1 - u) * v + S.g(j + 1, i + 1) * u * v; };
      const taban = ctx.rampa('toprak');
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const z = f(i / nx, j / ny), dx = f((i + 1) / nx, j / ny) - z, dy = f(i / nx, (j + 1) / ny) - z, isik = Math.max(0, Math.min(1, 0.6 - (dx + dy) * 18));
        const c = taban(z).match(/\d+/g).map(Number);
        x.fillStyle = 'rgb(' + c.map((q) => Math.round(q * (0.55 + 0.6 * isik))).map((q) => Math.min(255, q)).join(',') + ')';
        x.fillRect(sol + i * adim, 4 + j * adim, adim + 0.5, adim + 0.5);
      }
      const g = ctx.izgara(w, h, sol, 4);
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) { x.strokeStyle = '#2b59c3'; x.lineWidth = 2; x.strokeRect(sol + hh * g.cw + 1, 4 + l * g.ch + 1, g.cw - 2, g.ch - 2); } }
      return g.hucre;
    } });

  // ═══ 3B
  K.ekle({ id: 'uc-09', sinif: 'uc', ad: 'Saçılım kutusu', tarif: 'Bir cam kutunun içinde noktalar: genişlik head, derinlik katman, yükseklik güç. Her noktadan zemine bir ip iner.', yuk: 360, uc: true, kam: { yaw: -0.8, pitch: 0.45 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), rz = 0.6, s = Math.min(w * 0.55 / S.H, h * 0.9 / (S.L * rz)), P = ctx.P(kam, w, h, s, 20), Hm = S.H * 0.55, rn = ctx.rampa('magma');
      const X0 = -S.H / 2, X1 = S.H / 2, Z0 = -S.L * rz / 2, Z1 = S.L * rz / 2;
      const kose = [[X0, 0, Z0], [X1, 0, Z0], [X1, 0, Z1], [X0, 0, Z1], [X0, Hm, Z0], [X1, Hm, Z0], [X1, Hm, Z1], [X0, Hm, Z1]].map((p) => P(...p));
      x.strokeStyle = ctx.css('--cizgi'); x.lineWidth = 1;
      [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => { x.beginPath(); x.moveTo(kose[a][0], kose[a][1]); x.lineTo(kose[b][0], kose[b][1]); x.stroke(); });
      const nok = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) { const X = hh - (S.H - 1) / 2, Z = (l - (S.L - 1) / 2) * rz, v = S.g(l, hh); nok.push({ p: P(X, v * Hm, Z), z0: P(X, 0, Z), v, k: N.anahtar(l, hh) }); }
      nok.sort((a, b) => a.p[2] - b.p[2]);
      for (const q of nok) {
        if (q.v > 0.3) { x.strokeStyle = ctx.css('--soluk'); x.globalAlpha = 0.4; x.beginPath(); x.moveTo(q.z0[0], q.z0[1]); x.lineTo(q.p[0], q.p[1]); x.stroke(); }
        x.globalAlpha = 1; x.fillStyle = rn(0.2 + 0.8 * q.v); x.beginPath(); x.arc(q.p[0], q.p[1], 1.5 + 4 * q.v, 0, T); x.fill();
        if (ctx.secili().has(q.k)) { x.strokeStyle = '#3ad1ff'; x.lineWidth = 2; x.beginPath(); x.arc(q.p[0], q.p[1], 7, 0, T); x.stroke(); }
      }
      return (px, py) => yakin(nok.map((q) => [q.p[0], q.p[1], q.k]), px, py, 8);
    } });

  K.ekle({ id: 'uc-10', sinif: 'uc', ad: 'Halka kule', tarif: 'Her katman bir halka, halkalar üst üste bir kule. Halkadaki boncukların büyüklüğü güç; sıcak head’ler kulenin üzerinde parlar.', yuk: 380, uc: true, kam: { yaw: 0, pitch: 0.35 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), P = ctx.P(kam, w, h, Math.min(w, h) * 0.3), ara = 2.4 / S.L, rn = ctx.rampa('buz'), nok = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) { const a = hh / S.H * T, Y = (S.L / 2 - l) * ara; nok.push({ p: P(Math.cos(a), Y, Math.sin(a)), v: S.g(l, hh), k: N.anahtar(l, hh) }); }
      nok.sort((a, b) => a.p[2] - b.p[2]);
      for (const q of nok) {
        x.globalAlpha = q.p[2] < 0 ? 0.3 : 1; x.fillStyle = rn(1 - q.v); x.beginPath(); x.arc(q.p[0], q.p[1], 1.2 + 6 * q.v * q.v, 0, T); x.fill();
        if (ctx.secili().has(q.k)) { x.globalAlpha = 1; x.strokeStyle = '#ff6b00'; x.lineWidth = 2; x.beginPath(); x.arc(q.p[0], q.p[1], 8, 0, T); x.stroke(); }
      }
      x.globalAlpha = 1;
      return (px, py) => yakin(nok.filter((q) => q.p[2] >= -0.2).map((q) => [q.p[0], q.p[1], q.k]), px, py, 8);
    } });

  K.ekle({ id: 'uc-11', sinif: 'uc', ad: 'Işıklı yüzey', tarif: 'Arazi dolu ve ışıklandırılmış: her yüzün eğimi gölgesini belirler. Tel kafesin aksine hacim duygusu verir.', yuk: 360, uc: true, kam: { yaw: -0.5, pitch: 0.6 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), rz = 0.75, s = Math.min(w * 0.6 / S.H, h * 1.0 / (S.L * rz)), P = ctx.P(kam, w, h, s, 26), Hm = S.H * 0.38, rn = ctx.rampa('yosun');
      const V = (l, hh) => [hh - (S.H - 1) / 2, S.g(l, hh) * Hm, (l - (S.L - 1) / 2) * rz], yuz = [];
      for (let l = 0; l < S.L - 1; l++) for (let hh = 0; hh < S.H - 1; hh++) {
        const a = V(l, hh), b = V(l, hh + 1), c = V(l + 1, hh + 1), d = V(l + 1, hh);
        const nx = -(b[1] - a[1]), nz = -(d[1] - a[1]) / rz, isik = Math.max(0.25, Math.min(1, 0.75 + (nx * 0.5 - nz * 0.4) * 0.6));
        const k = [a, b, c, d].map((p) => P(...p)), ort = (S.g(l, hh) + S.g(l, hh + 1) + S.g(l + 1, hh) + S.g(l + 1, hh + 1)) / 4;
        const cc = rn(ort).match(/\d+/g).map(Number);
        yuz.push({ k, z: (k[0][2] + k[2][2]) / 2, renk: 'rgb(' + cc.map((q) => Math.min(255, Math.round(q * isik))).join(',') + ')' });
      }
      yuz.sort((p, q) => p.z - q.z); yuz.forEach((q) => cokgen(x, q.k, q.renk, q.renk, 0.6));
      const nok = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) { const p = P(...V(l, hh)); nok.push([p[0], p[1], N.anahtar(l, hh)]); }
      for (const k of ctx.secili()) { const { l, h: hh } = N.ac(k); if (l < S.L && hh < S.H) { const p = P(...V(l, hh)); x.fillStyle = '#ff3b30'; x.beginPath(); x.arc(p[0], p[1], 5, 0, T); x.fill(); } }
      return (px, py) => yakin(nok, px, py, 7);
    } });

  K.ekle({ id: 'uc-12', sinif: 'uc', ad: 'Simit', tarif: 'Katmanlar büyük çember boyunca, head’ler tüpün çevresinde. Son katman ilk katmanın yanına döner; model bir döngü gibi görünür.', yuk: 360, uc: true, kam: { yaw: 0.2, pitch: 0.75 },
    ciz(x, w, h, ctx, kam) {
      const S = ctx.S(), R = 1, r = 0.38, P = ctx.P(kam, w, h, Math.min(w, h) * 0.32), rn = ctx.rampa('kor'), q = [];
      const pos = (u, v) => [(R + r * Math.cos(v)) * Math.cos(u), r * Math.sin(v), (R + r * Math.cos(v)) * Math.sin(u)];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const u0 = l / S.L * T, u1 = (l + 0.92) / S.L * T, v0 = hh / S.H * T, v1 = (hh + 0.9) / S.H * T;
        const k = [pos(u0, v0), pos(u1, v0), pos(u1, v1), pos(u0, v1)].map((p) => P(...p)), m = P(...pos((u0 + u1) / 2, (v0 + v1) / 2));
        q.push({ k, m, z: m[2], v: S.g(l, hh), a: N.anahtar(l, hh) });
      }
      q.sort((a, b) => a.z - b.z);
      for (const c of q) cokgen(x, c.k, rn(c.v), ctx.secili().has(c.a) ? '#3ad1ff' : null, 2);
      return (px, py) => yakin(q.filter((c) => c.z > -0.3).map((c) => [c.m[0], c.m[1], c.a]), px, py, 8);
    } });

  K.ekle({ id: 'uc-13', sinif: 'uc', ad: 'Gezegen dizisi', tarif: 'Her katman bir gezegen, head’leri çevresinde dönen uydular; uydunun büyüklüğü gücü. Sıcak uydular yörüngede parlar.', yuk: 340, uc: true, anim: true, kam: { yaw: 0, pitch: 0.5 },
    ciz(x, w, h, ctx, kam, t) {
      const S = ctx.S(), P = ctx.P(kam, w, h, w / (S.L * 1.25)), rn = ctx.rampa('magma'), nok = [], don = (t || 0) / 4000, cisim = [];
      for (let l = 0; l < S.L; l++) {
        const X = (l - (S.L - 1) / 2) * 1.2;
        cisim.push({ p: P(X, 0, 0), gez: true, l });
        for (let hh = 0; hh < S.H; hh++) { const a = hh / S.H * T + don * (1 + l * 0.05), rr = 0.45; cisim.push({ p: P(X + Math.cos(a) * rr * 0.4, Math.sin(a) * rr, Math.cos(a) * rr), v: S.g(l, hh), k: N.anahtar(l, hh) }); }
      }
      cisim.sort((a, b) => a.p[2] - b.p[2]);
      for (const c of cisim) {
        if (c.gez) { x.fillStyle = ctx.css('--soluk'); x.globalAlpha = 0.6; x.beginPath(); x.arc(c.p[0], c.p[1], 3, 0, T); x.fill(); x.globalAlpha = 1; continue; }
        x.fillStyle = rn(0.2 + 0.8 * c.v); x.beginPath(); x.arc(c.p[0], c.p[1], 0.8 + 4 * c.v, 0, T); x.fill();
        if (ctx.secili().has(c.k)) { x.strokeStyle = '#3ad1ff'; x.lineWidth = 1.8; x.beginPath(); x.arc(c.p[0], c.p[1], 6, 0, T); x.stroke(); }
        nok.push([c.p[0], c.p[1], c.k]);
      }
      return (px, py) => yakin(nok, px, py, 7);
    } });

  // ═══ İskeletler
  const YUK = 330;
  const cer = (kap, stil) => { const f = el('div', { class: 'mini' }); f.style.cssText = 'height:' + YUK + 'px;position:relative;' + (stil || ''); kap.append(f); return f; };
  const pan = (ad, stil) => { const p = el('div', { class: 'p' }); if (ad) p.append(el('b', { text: ad })); if (stil) p.style.cssText = stil; return p; };
  const hepsi = (fs) => () => fs.forEach((f) => f());

  K.ekle({ id: 'isk-11', sinif: 'iskelet', ad: 'Masaüstü pencereler', tarif: 'Her araç ayrı bir pencere; başlığından tutup istediğin yere sürüklersin, tıkladığın öne gelir. Kendi düzenini kurarsın.',
    dom(kap, ctx) {
      const f = cer(kap, 'overflow:hidden;background:repeating-linear-gradient(45deg,transparent 0 10px,color-mix(in srgb,var(--murekkep) 3%,transparent) 10px 11px)');
      let z = 1; const cizler = [];
      [['Isı', 8, 8, 58, 62], ['Sıcak noktalar', 62, 6, 36, 48], ['Ölçüm', 46, 60, 40, 30], ['Lens', 4, 72, 38, 24]].forEach(([ad, l, t, w, h], i) => {
        const p = pan('', 'position:absolute;left:' + l + '%;top:' + t + '%;width:' + w + '%;height:' + h + '%;background:var(--kart);box-shadow:0 6px 18px rgba(0,0,0,.18);padding:0;z-index:' + (++z));
        const bas = el('div', { text: ad, style: 'font:600 10.5px var(--f);padding:4px 8px;background:color-mix(in srgb,var(--murekkep) 8%,var(--kart));cursor:move;user-select:none;touch-action:none' });
        const ic = el('div', { style: 'padding:4px 6px;overflow:hidden' }); p.append(bas, ic); f.append(p);
        let s = null;
        bas.addEventListener('pointerdown', (e) => { s = { x: e.clientX, y: e.clientY, l: p.offsetLeft, t: p.offsetTop }; bas.setPointerCapture(e.pointerId); p.style.zIndex = String(++z); });
        bas.addEventListener('pointermove', (e) => { if (!s) return; p.style.left = Math.max(0, Math.min(f.clientWidth - 40, s.l + e.clientX - s.x)) + 'px'; p.style.top = Math.max(0, Math.min(f.clientHeight - 20, s.t + e.clientY - s.y)) + 'px'; });
        bas.addEventListener('pointerup', () => { s = null; }); p.addEventListener('pointerdown', () => { p.style.zIndex = String(++z); });
        cizler.push(i === 0 ? ctx.mini.isi(ic, 160) : i === 1 ? ctx.mini.sicak(ic, 5) : i === 2 ? ctx.mini.olcum(ic) : ctx.mini.metin(ic, () => { const S = ctx.S(), ln = S.lens([...ctx.secili()]); return 'son katman: ' + N.tok(ln[S.L - 1][0].tok); }));
      });
      return hepsi(cizler);
    } });

  K.ekle({ id: 'isk-12', sinif: 'iskelet', ad: 'Sohbet', tarif: 'Neşter’le konuşur gibi: “en sıcak head’ler?”, “K19·H10’u kapat”, “ölç”. Cevaplar içinde küçük çizimler gelir.',
    dom(kap, ctx) {
      const f = cer(kap, 'display:grid;grid-template-rows:1fr auto;gap:6px');
      const akis = el('div', { style: 'overflow-y:auto;display:grid;gap:6px;align-content:start;padding-right:4px' });
      const g = el('input', { placeholder: 'Sor: sıcak · kapat K19·H10 · ölç · harita', 'aria-label': 'Mesaj' });
      f.append(akis, g);
      const balon = (ben, ...c) => { const b = el('div', { style: 'max-width:88%;padding:6px 9px;border-radius:10px;font-size:12px;' + (ben ? 'justify-self:end;background:var(--ikinci);color:var(--kart)' : 'justify-self:start;background:color-mix(in srgb,var(--murekkep) 6%,var(--kart));border:1px solid var(--cizgi)') }, ...c); akis.append(b); akis.scrollTop = akis.scrollHeight; return b; };
      function cevap(q) {
        const S = ctx.S(), m = q.toLocaleLowerCase('tr'), hk = q.match(/k?(\d+)\s*[·.]\s*h?(\d+)/i);
        if (hk && +hk[1] < S.L && +hk[2] < S.H) { ctx.degis(N.anahtar(+hk[1], +hk[2])); return balon(false, N.ad(+hk[1], +hk[2]) + (ctx.secili().has(N.anahtar(+hk[1], +hk[2])) ? ' seçime girdi.' : ' seçimden çıktı.')); }
        if (m.includes('sıcak') || m.includes('sicak')) return balon(false, 'En sıcak üç: ' + S.sicak.slice(0, 3).map((s) => N.ad(s.l, s.h) + ' (' + N.sayi(s.v) + ')').join(', '));
        if (m.includes('ölç') || m.includes('olc')) { const r = S.olc([...ctx.secili()]); return balon(false, ctx.secili().size + ' head kapalıyken “' + S.istem.hedef.trim() + '” ' + N.yuzde(r.once) + ' → ' + N.yuzde(r.sonra) + (r.degisti ? '; tahmin ' + N.tok(r.tahmin) + ' oldu.' : '.')); }
        if (m.includes('harita') || m.includes('ısı')) { const b = balon(false, 'İşte harita:'); b.style.width = '80%'; ctx.mini.isi(b, 110)(); return b; }
        return balon(false, 'Bunu anlamadım. “sıcak”, “kapat K19·H10”, “ölç” ya da “harita” diyebilirsin.');
      }
      g.addEventListener('keydown', (e) => { if (e.key !== 'Enter' || !g.value.trim()) return; balon(true, g.value); cevap(g.value); g.value = ''; });
      balon(true, 'en sıcak head’ler?'); cevap('sıcak'); balon(true, 'ölç'); cevap('ölç');
      return () => {};
    } });

  K.ekle({ id: 'isk-13', sinif: 'iskelet', ad: 'Pano', tarif: 'Head’ler kartlar hâlinde sütunlarda ilerler: Aday → Seçili → Sepette. Karta tıkla, bir sonraki sütuna geçsin. İş akışı göz önünde.',
    dom(kap, ctx) {
      const f = cer(kap, 'display:grid;grid-template-columns:repeat(3,1fr);gap:6px');
      const sutun = ['Aday', 'Seçili', 'Sepette'].map((a) => { const p = pan(a, 'overflow-y:auto'); f.append(p); return p; });
      const sepet = new Set();
      return function ciz() {
        const S = ctx.S();
        const kart = (s, yer) => el('div', { style: 'border:1px solid var(--cizgi);border-radius:5px;padding:4px 6px;background:var(--kart);cursor:pointer;font:11px var(--m);display:flex;justify-content:space-between', onclick: () => {
          if (yer === 0) ctx.degis(s.k); else if (yer === 1) { sepet.add(s.k); ciz(); } else { sepet.delete(s.k); ciz(); }
        } }, el('span', { text: N.ad(s.l, s.h) }), el('span', { text: yer === 2 ? '×' : '→', style: 'color:var(--soluk)' }));
        const yer = (s) => (sepet.has(s.k) ? 2 : ctx.secili().has(s.k) ? 1 : 0);
        sutun.forEach((p, i) => { p.replaceChildren(p.firstChild, ...S.sicak.filter((s) => yer(s) === i).map((s) => kart(s, i))); });
        if (sepet.size) sutun[2].append(el('div', { class: 'kucuk', text: 'sepet ölçümü: ' + N.yuzde(S.olc([...sepet]).sonra) }));
      };
    } });

  K.ekle({ id: 'isk-14', sinif: 'iskelet', ad: 'Kaydırmalı sayfalar', tarif: 'Her adım tam genişlikte bir sayfa; yana kaydırarak geçilir, alttaki noktalar nerede olduğunu söyler. Telefona en yakın omurga.',
    dom(kap, ctx) {
      const f = cer(kap, 'display:grid;grid-template-rows:1fr auto;gap:6px');
      const kay = el('div', { style: 'display:grid;grid-auto-flow:column;grid-auto-columns:100%;overflow-x:auto;scroll-snap-type:x mandatory;gap:8px' });
      const nok = el('div', { style: 'display:flex;gap:6px;justify-content:center' });
      f.append(kay, nok);
      const ad = ['1 · Isı', '2 · Sıcak', '3 · Ölçüm', '4 · Kaydet'], cizler = [];
      ad.forEach((a, i) => { const p = pan(a, 'scroll-snap-align:start'); kay.append(p); cizler.push(i === 0 ? ctx.mini.isi(p, YUK - 70) : i === 1 ? ctx.mini.sicak(p, 10) : i === 2 ? ctx.mini.olcum(p) : ctx.mini.metin(p, () => ctx.S().model.id + '-kesik · ' + ctx.secili().size + ' kesim · safetensors')); });
      const noktalar = ad.map((a, i) => el('span', { title: a, style: 'width:8px;height:8px;border-radius:50%;cursor:pointer;background:var(--cizgi)', onclick: () => kay.scrollTo({ left: i * kay.clientWidth, behavior: 'smooth' }) }));
      nok.append(...noktalar);
      const isaretle = () => { const i = Math.round(kay.scrollLeft / Math.max(1, kay.clientWidth)); noktalar.forEach((n, j) => { n.style.background = j === i ? 'var(--vurgu)' : 'var(--cizgi)'; }); };
      kay.addEventListener('scroll', isaretle); isaretle();
      return hepsi(cizler);
    } });

  K.ekle({ id: 'isk-15', sinif: 'iskelet', ad: 'Hesap tablosu', tarif: 'Katmanlar satır, head’ler sütun, hücrelerde sayılar. Üstte formül çubuğu gibi bir seçim satırı; Excel’e alışkın göz için.',
    dom(kap, ctx) {
      const f = cer(kap, 'display:grid;grid-template-rows:auto 1fr;gap:4px');
      const formul = el('div', { style: 'display:flex;gap:6px;align-items:center;border:1px solid var(--cizgi);border-radius:4px;padding:3px 6px;font:11px var(--m)' });
      const kaydir = el('div', { style: 'overflow:auto;border:1px solid var(--cizgi);border-radius:4px' });
      f.append(formul, kaydir);
      return () => {
        const S = ctx.S(), r = S.olc([...ctx.secili()]), rn = ctx.rampaTema();
        formul.replaceChildren(el('b', { text: 'fx', style: 'color:var(--soluk);font-style:italic' }), el('span', { text: '=ÖLÇ(' + ([...ctx.secili()].map(N.adK).join('; ') || '') + ')  → ' + N.yuzde(r.sonra) }));
        const tb = el('table', { style: 'border-collapse:collapse;font:10px var(--m)' });
        tb.append(el('tr', null, el('th', { text: '', style: 'position:sticky;left:0;top:0;background:var(--kart);z-index:2' }), ...[...Array(S.H).keys()].map((hh) => el('th', { text: 'H' + hh, style: 'position:sticky;top:0;background:color-mix(in srgb,var(--murekkep) 6%,var(--kart));padding:2px 4px;font-weight:600' }))));
        for (let l = 0; l < S.L; l++) tb.append(el('tr', null, el('th', { text: 'K' + l, style: 'position:sticky;left:0;background:color-mix(in srgb,var(--murekkep) 6%,var(--kart));padding:2px 4px;font-weight:600' }),
          ...[...Array(S.H).keys()].map((hh) => { const k = N.anahtar(l, hh), v = S.g(l, hh); return el('td', { text: N.sayi(v), onclick: () => ctx.degis(k), style: 'padding:2px 4px;border:1px solid var(--cizgi);cursor:pointer;text-align:right;background:' + (ctx.secili().has(k) ? 'var(--vurgu);color:var(--kart)' : v > 0.5 ? rn(v * 0.6) : 'transparent') }); })));
        const sl = kaydir.scrollLeft, st = kaydir.scrollTop; kaydir.replaceChildren(tb); kaydir.scrollLeft = sl; kaydir.scrollTop = st;
      };
    } });

  // ═══ Ölçüm
  const olc = (ctx) => ctx.S().olc([...ctx.secili()]);
  K.ekle({ id: 'olc-07', sinif: 'olcum', ad: 'Termometre', tarif: 'Olasılık bir termometrede; soluk sütun eski seviye, dolu sütun yenisi. Ölçek yanında, rakibin seviyesi ince bir çizgi.', yuk: 240,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), cx = w / 2, ust = 16, alt = h - 40, Y = (p) => alt - p * (alt - ust), gen = 22;
      x.fillStyle = ctx.css('--cizgi'); x.fillRect(cx - gen / 2, ust, gen, alt - ust); x.beginPath(); x.arc(cx, alt + 12, 18, 0, T); x.fill();
      x.fillStyle = ctx.css('--soluk'); x.globalAlpha = 0.35; x.fillRect(cx - gen / 2 + 4, Y(r.once), gen - 8, alt - Y(r.once)); x.globalAlpha = 1;
      x.fillStyle = ctx.css('--vurgu'); x.fillRect(cx - gen / 2 + 4, Y(r.sonra), gen - 8, alt - Y(r.sonra) + 6); x.beginPath(); x.arc(cx, alt + 12, 13, 0, T); x.fill();
      [0, 0.25, 0.5, 0.75, 1].forEach((p) => { x.fillStyle = ctx.css('--soluk'); x.fillRect(cx + gen / 2 + 2, Y(p), 6, 1); yazi(x, ctx, N.yuzde(p), cx + gen / 2 + 12, Y(p) + 4, 10, ctx.css('--soluk')); });
      x.strokeStyle = ctx.css('--ikinci'); x.lineWidth = 2; x.beginPath(); x.moveTo(cx - gen, Y(r.rakipSonra)); x.lineTo(cx + gen / 2, Y(r.rakipSonra)); x.stroke();
      yazi(x, ctx, 'rakip ' + N.yuzde(r.rakipSonra), cx - gen - 4, Y(r.rakipSonra) + 4, 10, ctx.css('--ikinci'), 'right');
      yazi(x, ctx, N.yuzde(r.sonra), cx - gen - 4, Y(r.sonra) - 6, 18, null, 'right', 800);
    } });

  K.ekle({ id: 'olc-08', sinif: 'olcum', ad: 'İç içe halkalar', tarif: 'Dış halka kesimden önce, iç halka sonra; dilimler hedef, rakip ve geri kalan. Oranların nasıl el değiştirdiği tek şekilde.', yuk: 240,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), S = ctx.S(), cy = h / 2, R = Math.max(40, Math.min(cy - 10, (w - 140) / 2 - 6)), cx = R + 8;
      const halka = (dil, ro, ri) => { let a = -Math.PI / 2; dil.forEach(([p, c]) => { x.beginPath(); x.arc(cx, cy, ro, a, a + p * T); x.arc(cx, cy, ri, a + p * T, a, true); x.closePath(); x.fillStyle = c; x.fill(); a += p * T; }); };
      const renk = [ctx.css('--ikinci'), ctx.css('--vurgu'), ctx.css('--cizgi')];
      halka([[r.once, renk[0]], [r.rakipOnce, renk[1]], [1 - r.once - r.rakipOnce, renk[2]]], R, R * 0.78);
      halka([[r.sonra, renk[0]], [r.rakipSonra, renk[1]], [Math.max(0, 1 - r.sonra - r.rakipSonra), renk[2]]], R * 0.7, R * 0.45);
      const X = cx + R + 18;
      [['dış: önce', ''], [N.tok(S.istem.hedef).trim(), N.yuzde(r.once) + ' → ' + N.yuzde(r.sonra), renk[0]], [N.tok(S.istem.rakip).trim(), N.yuzde(r.rakipOnce) + ' → ' + N.yuzde(r.rakipSonra), renk[1]], ['iç: sonra', '']].forEach(([a, b, c], i) => {
        if (c) { x.fillStyle = c; x.fillRect(X, 40 + i * 30 - 9, 10, 10); }
        yazi(x, ctx, a, X + (c ? 16 : 0), 40 + i * 30, 12, c ? null : ctx.css('--soluk'), 'left', c ? 700 : 400); if (b) yazi(x, ctx, b, X + 16, 40 + i * 30 + 14, 11, ctx.css('--soluk'));
      });
    } });

  K.ekle({ id: 'olc-09', sinif: 'olcum', ad: 'Akış bandı', tarif: 'Önceki olasılık kütlesi soldan girer, kesimden sonra hedefe, rakibe ve diğerlerine nasıl dağıldığı bantlarla sağa akar.', yuk: 240,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), S = ctx.S(), x0 = 80, x1 = w - 90, H = h - 30;
      const sol = [['hedef', r.once, ctx.css('--ikinci')], ['rakip', r.rakipOnce, ctx.css('--vurgu')], ['diğer', 1 - r.once - r.rakipOnce, ctx.css('--soluk')]];
      const sag = [['hedef', r.sonra, ctx.css('--ikinci')], ['rakip', r.rakipSonra, ctx.css('--vurgu')], ['diğer', Math.max(0, 1 - r.sonra - r.rakipSonra), ctx.css('--soluk')]];
      const kalan = r.sonra, kayan = r.once - r.sonra, akim = [[0, 0, kalan], [0, 1, kayan * 0.55], [0, 2, kayan * 0.45], [1, 1, r.rakipOnce], [2, 2, 1 - r.once - r.rakipOnce]];
      const ys = (lst) => { let y = 12; return lst.map(([, p]) => { const a = y; y += p * H + 4; return a; }); };
      const ly = ys(sol), ry = ys(sag), lof = [0, 0, 0], rof = [0, 0, 0];
      for (const [a, b, p] of akim) {
        if (p <= 0.001) continue;
        const ya = ly[a] + lof[a], yb = ry[b] + rof[b], t = p * H; lof[a] += t; rof[b] += t;
        x.globalAlpha = 0.35; x.fillStyle = sag[b][2];
        x.beginPath(); x.moveTo(x0, ya); x.bezierCurveTo((x0 + x1) / 2, ya, (x0 + x1) / 2, yb, x1, yb); x.lineTo(x1, yb + t); x.bezierCurveTo((x0 + x1) / 2, yb + t, (x0 + x1) / 2, ya + t, x0, ya + t); x.closePath(); x.fill();
      }
      x.globalAlpha = 1;
      sol.forEach(([a, p, c], i) => { x.fillStyle = c; x.fillRect(x0 - 8, ly[i], 8, p * H); yazi(x, ctx, a + ' ' + N.yuzde(p), x0 - 12, ly[i] + p * H / 2 + 4, 11, null, 'right'); });
      sag.forEach(([a, p, c], i) => { x.fillStyle = c; x.fillRect(x1, ry[i], 8, p * H); yazi(x, ctx, a + ' ' + N.yuzde(p), x1 + 12, ry[i] + p * H / 2 + 4, 11); });
      yazi(x, ctx, 'önce', x0 - 4, h - 4, 10, ctx.css('--soluk'), 'right'); yazi(x, ctx, 'sonra', x1 + 4, h - 4, 10, ctx.css('--soluk'));
    } });

  K.ekle({ id: 'olc-10', sinif: 'olcum', ad: 'Mermi grafiği', tarif: 'Yatay bir çubukta: gri bantlar rastgele kesimlerin KL aralığı, koyu çubuk bu kesimin KL’si. Bantı aşıyorsa kesim anlamlı.', yuk: 200,
    ciz(x, w, h, ctx) {
      const r = olc(ctx), lo = 0.012 + 0.01 * ctx.secili().size, hi = lo * 5.5, enb = Math.max(1.2, r.kl * 1.25), X = (v) => 20 + v / enb * (w - 40), Y = 70;
      x.fillStyle = ctx.css('--cizgi'); x.fillRect(X(0), Y, X(enb) - X(0), 36);
      x.fillStyle = ctx.css('--soluk'); x.globalAlpha = 0.45; x.fillRect(X(lo), Y, X(hi) - X(lo), 36); x.globalAlpha = 1;
      x.fillStyle = r.kl > hi ? ctx.css('--vurgu') : ctx.css('--murekkep'); x.fillRect(X(0), Y + 12, X(r.kl) - X(0), 12);
      [0, enb / 2, enb].forEach((v) => yazi(x, ctx, N.sayi(v, 2), X(v), Y + 52, 10, ctx.css('--soluk'), 'center'));
      yazi(x, ctx, 'KL ' + N.sayi(r.kl, 3), 20, 30, 20, null, 'left', 800);
      yazi(x, ctx, ctx.secili().size ? (r.kl > hi ? 'rastgele kesim bandını aşıyor: anlamlı' : 'rastgele kesimlerden ayırt edilemiyor') : 'seçim yok', 20, 50, 12, r.kl > hi ? ctx.css('--vurgu') : ctx.css('--soluk'));
      yazi(x, ctx, 'gri bant: rastgele kesimler ' + N.sayi(lo, 3) + '–' + N.sayi(hi, 3), 20, h - 14, 11, ctx.css('--soluk'));
    } });

  K.ekle({ id: 'olc-11', sinif: 'olcum', ad: 'Pano kutucukları', tarif: 'Dört küçük kutu: önce, sonra, KL ve yeni tahmin. Panoda bir bakışta okunacak kadar sade; değişen kutunun kenarı renklenir.',
    dom(kap, ctx) {
      const f = el('div', { class: 'mini', style: 'display:grid;grid-template-columns:repeat(2,1fr);gap:8px;min-height:240px;align-content:start' }); kap.append(f);
      return () => {
        const r = olc(ctx), S = ctx.S();
        const kutu = (ad, deger, alt, vurgu) => el('div', { class: 'p', style: 'padding:10px 12px;' + (vurgu ? 'border-color:var(--vurgu);box-shadow:inset 3px 0 0 var(--vurgu)' : '') }, el('b', { text: ad }), el('div', { class: 'buyuk', text: deger }), el('div', { class: 'kucuk', text: alt }));
        f.replaceChildren(kutu('Önce', N.yuzde(r.once), '“' + S.istem.hedef.trim() + '” olasılığı'), kutu('Sonra', N.yuzde(r.sonra), ctx.secili().size + ' head kapalı', r.sonra < r.once - 0.05),
          kutu('KL', N.sayi(r.kl, 2), 'çıktı değişimi', r.kl > 0.3), kutu('Tahmin', N.tok(r.tahmin).trim(), r.degisti ? 'değişti' : 'aynı', r.degisti));
      };
    } });

  // ═══ Lens ve dikkat
  K.ekle({ id: 'lns-04', sinif: 'lens', ad: 'Olasılık dağları', tarif: 'Katman boyunca hedef, rakip ve en iyi diğer tahminin olasılıkları üst üste yığılmış alanlar. Hedefin dağı nerede yükseliyor, görünür.', yuk: 260,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), ln = S.lens([...ctx.secili()]), X = (l) => 20 + l / (S.L - 1) * (w - 40), Y = (p) => h - 22 - p * (h - 40);
      const seri = [[(r) => hp(r, S.istem.hedef), ctx.css('--ikinci'), N.tok(S.istem.hedef).trim()], [(r) => hp(r, S.istem.rakip), ctx.css('--vurgu'), N.tok(S.istem.rakip).trim()], [(r) => r.filter((q) => q.tok !== S.istem.hedef && q.tok !== S.istem.rakip).reduce((a, q) => a + q.p, 0), ctx.css('--soluk'), 'diğer']];
      const taban = new Array(S.L).fill(0);
      seri.forEach(([f, c, ad]) => {
        const ust = ln.map((r, l) => taban[l] + f(r));
        x.beginPath(); ust.forEach((v, l) => (l ? x.lineTo(X(l), Y(v)) : x.moveTo(X(l), Y(v)))); for (let l = S.L - 1; l >= 0; l--) x.lineTo(X(l), Y(taban[l])); x.closePath();
        x.fillStyle = c; x.globalAlpha = 0.6; x.fill(); x.globalAlpha = 1;
        const m = ust.indexOf(Math.max(...ust.map((v, l) => v - taban[l]).map((d, l) => d))) ;
        yazi(x, ctx, ad, X(S.L - 1) - 4, Y((ust[S.L - 1] + taban[S.L - 1]) / 2) + 4, 11, null, 'right', 600);
        ust.forEach((v, l) => { taban[l] = v; }); void m;
      });
      yazi(x, ctx, 'K0', 20, h - 6, 10, ctx.css('--soluk')); yazi(x, ctx, 'K' + (S.L - 1), w - 20, h - 6, 10, ctx.css('--soluk'), 'right');
    } });

  K.ekle({ id: 'lns-05', sinif: 'lens', ad: 'Yazı sütunları', tarif: 'Her katmanın üç tahmini, olasılığı kadar büyük harflerle. Okuyarak ilerlenir: model önce “ve”, sonra “İstanbul”, en sonda “Ankara” der.', yuk: 260,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), ln = S.lens([...ctx.secili()]), adim = Math.max(1, Math.ceil(S.L / 8)), kat = ln.map((r, l) => [r, l]).filter(([, l]) => l % adim === 0 || l === S.L - 1), cw = w / kat.length;
      kat.forEach(([r, l], i) => {
        const cx = (i + 0.5) * cw; yazi(x, ctx, 'K' + l, cx, h - 6, 10, ctx.css('--soluk'), 'center');
        let y = 20; r.forEach((t) => { const boy = 9 + 26 * t.p; y += boy; yazi(x, ctx, N.tok(t.tok).trim() || '␣', cx, y, boy, t.tok === S.istem.hedef ? ctx.css('--ikinci') : t.tok === S.istem.rakip ? ctx.css('--vurgu') : ctx.css('--soluk'), 'center', 700); y += 6; });
      });
    } });

  K.ekle({ id: 'dkt-04', sinif: 'lens', ad: 'Dikkat akoru', tarif: 'Belirteçler bir çemberin çevresinde; aralarındaki bantlar dikkat. Kalın bant, bir belirtecin öbürüne çok baktığını söyler.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), k = [...ctx.secili()][0] || S.sicak[0].k, { l, h: hh } = N.ac(k), at = S.dikkat(l, hh), cx = w / 2, cy = h / 2 - 6, R = Math.min(w, h) / 2 - 40;
      const a = (i) => -Math.PI / 2 + i / S.T * T, P = (i) => [cx + Math.cos(a(i)) * R, cy + Math.sin(a(i)) * R];
      for (let i = 0; i < S.T; i++) for (let j = 0; j < i; j++) {
        const wgt = at.m[i][j]; if (wgt < 0.04) continue;
        const [x1, y1] = P(i), [x2, y2] = P(j);
        x.strokeStyle = i === S.T - 1 ? ctx.css('--vurgu') : ctx.css('--ikinci'); x.globalAlpha = 0.15 + 0.8 * wgt; x.lineWidth = 1 + 9 * wgt;
        x.beginPath(); x.moveTo(x1, y1); x.quadraticCurveTo(cx, cy, x2, y2); x.stroke();
      }
      x.globalAlpha = 1;
      for (let i = 0; i < S.T; i++) { const [px, py] = P(i), [lx, ly] = [cx + Math.cos(a(i)) * (R + 18), cy + Math.sin(a(i)) * (R + 18)]; x.fillStyle = ctx.css('--murekkep'); x.beginPath(); x.arc(px, py, 4, 0, T); x.fill(); yazi(x, ctx, N.tok(S.istem.tokenlar[i]), lx, ly + 4, 11, null, 'center'); }
      yazi(x, ctx, N.adK(k) + ' · ' + at.tur, 10, h - 6, 11, ctx.css('--soluk'));
    } });

  K.ekle({ id: 'dkt-05', sinif: 'lens', ad: 'Dikkat köprüsü', tarif: 'Solda okuyan, sağda okunan belirteçler; aralarında dikkat kadar kalın bantlar. Hangi sözcüğün kime yaslandığı akış gibi okunur.', yuk: 280,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), k = [...ctx.secili()][0] || S.sicak[0].k, { l, h: hh } = N.ac(k), at = S.dikkat(l, hh), x0 = 90, x1 = w - 90, rh = (h - 30) / S.T;
      const ofs = new Array(S.T).fill(0), girdi = new Array(S.T).fill(0);
      for (let i = 0; i < S.T; i++) for (let j = 0; j <= i; j++) girdi[j] += at.m[i][j];
      const top = girdi.reduce((a, b) => a + b, 0), sagY = []; let y = 10; girdi.forEach((g) => { sagY.push(y); y += g / top * (h - 30) + 2; });
      for (let i = 0; i < S.T; i++) {
        let ly = 10 + i * rh;
        for (let j = 0; j <= i; j++) {
          const t = at.m[i][j] * (rh - 2), u = at.m[i][j] / top * (h - 30); if (t < 0.3) { ly += t; continue; }
          const ry = sagY[j] + ofs[j]; ofs[j] += u;
          x.fillStyle = i === S.T - 1 ? ctx.css('--vurgu') : ctx.css('--ikinci'); x.globalAlpha = i === S.T - 1 ? 0.6 : 0.25;
          x.beginPath(); x.moveTo(x0, ly); x.bezierCurveTo((x0 + x1) / 2, ly, (x0 + x1) / 2, ry, x1, ry); x.lineTo(x1, ry + u); x.bezierCurveTo((x0 + x1) / 2, ry + u, (x0 + x1) / 2, ly + t, x0, ly + t); x.closePath(); x.fill();
          ly += t;
        }
      }
      x.globalAlpha = 1;
      for (let i = 0; i < S.T; i++) { yazi(x, ctx, N.tok(S.istem.tokenlar[i]), x0 - 6, 10 + (i + 0.6) * rh, 11, i === S.T - 1 ? ctx.css('--vurgu') : null, 'right'); yazi(x, ctx, N.tok(S.istem.tokenlar[i]), x1 + 6, sagY[i] + girdi[i] / top * (h - 30) / 2 + 4, 11); }
      yazi(x, ctx, 'okuyan', x0 - 6, h - 4, 10, ctx.css('--soluk'), 'right'); yazi(x, ctx, 'okunan', x1 + 6, h - 4, 10, ctx.css('--soluk'));
    } });

  K.ekle({ id: 'dkt-06', sinif: 'lens', ad: 'Son belirtecin bakışı', tarif: 'Satırlar en sıcak on head, sütunlar belirteçler: her head son belirteçteyken nereye bakıyor. Head’ler arası kıyas tek tabloda. Satıra tıkla.', yuk: 300,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), hs = S.sicak.slice(0, 10), pad = 64, cw = (w - pad - 8) / S.T, rh = (h - 30) / hs.length, rn = ctx.rampa('viridis'), kutu = [];
      hs.forEach((s, i) => {
        const at = S.dikkat(s.l, s.h).m[S.T - 1], Y = 4 + i * rh, sec = ctx.secili().has(s.k);
        yazi(x, ctx, N.ad(s.l, s.h), pad - 6, Y + rh * 0.65, 10.5, sec ? ctx.css('--vurgu') : ctx.css('--soluk'), 'right', sec ? 700 : 400);
        for (let j = 0; j < S.T; j++) { x.fillStyle = rn(at[j]); x.fillRect(pad + j * cw + 0.5, Y + 0.5, cw - 1, rh - 1); }
        kutu.push([0, Y, w, rh, s.k]);
      });
      for (let j = 0; j < S.T; j++) yazi(x, ctx, N.tok(S.istem.tokenlar[j]).slice(0, 6), pad + (j + 0.5) * cw, h - 8, 10, ctx.css('--soluk'), 'center');
      return kutuBul(kutu);
    } });
})();
