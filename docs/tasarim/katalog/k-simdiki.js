/* "Şu anki Neşter": uygulamadaki bugünkü çizimlerin aynısı (tokens.css, drawHeatmap.ts, FlowScene, Explore, ControlStatsCard, LensPanel, AttentionPanel).
   Renkler uygulamanın sabit koyu paleti; kıyaslamak için her sınıfın başına konur. */
(function () {
  const K = window.Katalog, N = window.Nester, el = N.el, T = Math.PI * 2;
  const APP = { s0: '#0b0e13', s1: '#121722', border: '#222a38', ctrl: '#5f6b7f', t1: '#e6eaf2', t2: '#a7b0c0', inc: '#2ee6c5', dec: '#9fc3ff', basket: '#ff7a45', badge: '#ffd27a', ring: '#9fc3ff' };
  const MONO = '"JetBrains Mono", ui-monospace, monospace', HEAD = '"Space Grotesk", system-ui, sans-serif';
  const karis = N.rampa([APP.s1, APP.inc]);
  const acik = (rgb) => { const m = rgb.match(/\d+/g).map(Number); return 'rgb(' + m.slice(0, 3).map((v) => Math.round(v + (255 - v) * 0.4)).join(',') + ')'; };

  // drawHeatmap.ts: hücre başına fillRect; seçili açılır + odak halkası 3px; sepet turuncu 2px.
  function appIsi(x, S, secili, sepet, X0, Y0, W, H, olcek) {
    const cw = W / S.H, ch = H / S.L, enb = [];
    for (let l = 0; l < S.L; l++) { let m = 0; for (let h = 0; h < S.H; h++) m = Math.max(m, S.g(l, h)); enb.push(m || 1); }
    const deger = (l, h) => (olcek === 'tum' ? S.g(l, h) : S.g(l, h) / enb[l]);
    const cizgi = [];
    for (let l = 0; l < S.L; l++) for (let h = 0; h < S.H; h++) {
      const k = N.anahtar(l, h), c = karis(deger(l, h)), r = [X0 + h * cw, Y0 + l * ch, cw, ch];
      x.fillStyle = secili.has(k) ? acik(c) : c; x.fillRect(r[0], r[1], Math.ceil(r[2]), Math.ceil(r[3]));
      if (secili.has(k) || sepet.has(k)) cizgi.push([k, r]);
    }
    for (const [k, r] of cizgi) {
      const i = (inset) => Math.min(inset, r[2] / 4, r[3] / 4);
      if (secili.has(k)) { const a = i(1.5); x.strokeStyle = APP.ring; x.lineWidth = 3; x.strokeRect(r[0] + a, r[1] + a, r[2] - 2 * a, r[3] - 2 * a); }
      if (sepet.has(k)) { const a = i(3); x.strokeStyle = APP.basket; x.lineWidth = 2; x.strokeRect(r[0] + a, r[1] + a, r[2] - 2 * a, r[3] - 2 * a); }
    }
    return (px, py) => { const h = Math.floor((px - X0) / cw), l = Math.floor((py - Y0) / ch); return l >= 0 && l < S.L && h >= 0 && h < S.H ? N.anahtar(l, h) : null; };
  }
  K.appIsi = appIsi;

  K.ekle({ id: 'isi-00', sinif: 'isi', simdiki: true, ad: '2B ısı haritası', tarif: 'Uygulamadaki harita: koyu panel → turkuaz, katman içinde 0–1. Seçili hücre açılır ve mavi halka alır; sepettekiler turuncu.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S();
      x.fillStyle = APP.s0; x.fillRect(0, 0, w, h);
      x.font = '12px ' + HEAD; x.fillStyle = APP.t2; x.textAlign = 'left';
      x.fillText('Head çıktı gücü, katman içinde 0–1', 30, 14);
      const tw = x.measureText('Head çıktı gücü, katman içinde 0–1').width;
      x.fillStyle = APP.badge; x.font = '600 10px ' + HEAD; x.fillText('tahmini', 38 + tw, 14);
      const g = x.createLinearGradient(w - 110, 0, w - 30, 0); g.addColorStop(0, APP.s1); g.addColorStop(1, APP.inc); x.fillStyle = g; x.fillRect(w - 110, 6, 80, 8);
      x.font = '10px ' + MONO; x.fillStyle = APP.t2; x.fillText('düşük', w - 150, 14); x.fillText('yüksek', w - 26, 14);
      x.textAlign = 'right'; for (const l of [0, Math.floor(S.L / 2), S.L - 1]) x.fillText('K' + l, 26, 24 + (l + 0.75) * ((h - 30) / S.L));
      return appIsi(x, S, ctx.secili(), new Set(), 30, 22, w - 36, h - 28, 'katman');
    } });

  // FlowScene kopyası. opt.neon: parlayan bağlar + akan ışıklar; opt.buyu (0..1): bağlar girdiden cevaba doğru büyür.
  function appBag(x, w, h, ctx, kam, t, opt) {
      opt = opt || {}; t = t || 0;
      const S = ctx.S(), G = N.bagGrafi(S), rz = 0.62, buyu = opt.buyu == null ? 1 : opt.buyu;
      x.fillStyle = APP.s0; x.fillRect(0, 0, w, h);
      const karisim = Math.max(0, Math.min(1, (Math.PI / 2 - kam.pitch) / (Math.PI / 2 - 0.95)));
      const s = Math.min(w * 0.88 / S.H, h * 0.86 / (S.L * rz)) * (0.75 + 0.25 * Math.sin(kam.pitch)), P = ctx.P(kam, w, h, s, 10);
      const hx = (hh) => hh - (S.H - 1) / 2, lz = (l) => (l - (S.L - 1) / 2) * rz, Yt = S.L * rz * 0.5;
      const zem = [];
      for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
        const k = [P(hx(hh) - 0.5, 0, lz(l) - rz / 2), P(hx(hh) + 0.5, 0, lz(l) - rz / 2), P(hx(hh) + 0.5, 0, lz(l) + rz / 2), P(hx(hh) - 0.5, 0, lz(l) + rz / 2)];
        zem.push({ k, z: k[0][2], c: karis(S.g(l, hh)), a: N.anahtar(l, hh) });
      }
      zem.sort((p, q) => p.z - q.z);
      for (const q of zem) {
        x.globalAlpha = 1 - 0.5 * karisim; x.beginPath(); q.k.forEach((p, i) => (i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]))); x.closePath();
        x.fillStyle = ctx.secili().has(q.a) ? acik(q.c) : q.c; x.fill();
        if (ctx.secili().has(q.a)) { x.globalAlpha = 1; x.strokeStyle = APP.ring; x.lineWidth = 2; x.stroke(); }
      }
      x.globalAlpha = 1;
      const nok = [];
      if (karisim > 0.02) {
        const kon = new Map(G.dugumler.map((n) => { const y = n.tur === 'girdi' ? 0.1 : n.tur === 'cikti' ? 1 : 0.25 + 0.75 * n.l / S.L; return [n.id, P(hx(n.h), y * Yt * karisim, lz(n.l))]; }));
        const dm = new Map(G.dugumler.map((n) => [n.id, n]));
        const nereden = (id) => { const n = dm.get(id); return n.tur === 'girdi' ? 0 : n.tur === 'cikti' ? 1 : (n.l + 1) / S.L; };
        const egri = (A, B) => { const cx = (A[0] + B[0]) / 2, cy = Math.min(A[1], B[1]) - 24 * karisim; return (u) => [(1 - u) * (1 - u) * A[0] + 2 * (1 - u) * u * cx + u * u * B[0], (1 - u) * (1 - u) * A[1] + 2 * (1 - u) * u * cy + u * u * B[1]]; };
        if (opt.neon) { x.globalCompositeOperation = 'lighter'; x.shadowBlur = 12; }
        for (const b of G.baglar) {
          const A = kon.get(b.k), B = kon.get(b.h), kes = ctx.secili().has(b.h), g = Math.max(0, Math.min(1, (buyu * 1.5 - nereden(b.k)) / 0.5));
          if (g <= 0) continue;
          const f = egri(A, B), titre = kes && opt.neon ? 0.65 + 0.35 * Math.sin(t / 110) : 1;
          x.globalAlpha = karisim * (kes ? 0.9 : opt.neon ? 0.7 : 0.55) * titre; x.strokeStyle = kes ? APP.basket : APP.inc; x.shadowColor = kes ? APP.basket : APP.inc;
          x.lineWidth = kes ? 1.5 : 1 + 5 * b.w; x.setLineDash(kes ? [4, 4] : []); if (kes && opt.neon) x.lineDashOffset = -t / 40;
          x.beginPath(); x.moveTo(A[0], A[1]); for (let i = 1; i <= 18; i++) { const p = f(g * i / 18); x.lineTo(p[0], p[1]); } x.stroke();
          if (opt.neon && !kes && g >= 1) for (const j of [0, 0.5]) {
            const p = f((t / 2200 + j + b.w * 0.37) % 1);
            x.globalAlpha = karisim * 0.95; x.fillStyle = '#e9fffb'; x.beginPath(); x.arc(p[0], p[1], 1.4 + 2.2 * b.w, 0, T); x.fill();
          }
        }
        x.setLineDash([]); x.lineDashOffset = 0; x.globalCompositeOperation = 'source-over'; x.shadowBlur = 0;
        for (const n of G.dugumler) {
          const p = kon.get(n.id), sec = ctx.secili().has(n.id);
          if (buyu * 1.5 < nereden(n.id)) continue;
          x.globalAlpha = karisim; x.fillStyle = sec ? APP.ring : n.tur === 'cikti' ? APP.inc : APP.t1;
          if (opt.neon) { x.shadowColor = sec ? APP.ring : APP.inc; x.shadowBlur = n.tur === 'cikti' || sec ? 18 : 8; }
          x.beginPath(); x.arc(p[0], p[1], n.tur === 'cikti' && opt.neon ? 6 + Math.sin(t / 300) : 5, 0, T); x.fill();
          x.shadowBlur = 0;
          x.font = '11px ' + MONO; x.fillStyle = APP.t2; x.textAlign = 'center'; x.fillText(n.ad, p[0], p[1] - 9);
          if (n.tur === 'head') nok.push([p[0], p[1], n.id]);
        }
        x.globalAlpha = 1;
      }
      x.font = '11px ' + HEAD; x.fillStyle = APP.t2; x.textAlign = 'left';
      x.fillText(karisim > 0.02 ? 'Bağlar 3B · ' + G.baglar.length + ' bağ · ' + Math.round(karisim * 100) + '% eğik' : 'Isı haritası (tepeden) · aşağı sürükle', 10, h - 10);
      const zemBul = (px, py) => { let en = null, ed = 1e9; for (const q of zem) { const cx = (q.k[0][0] + q.k[2][0]) / 2, cy = (q.k[0][1] + q.k[2][1]) / 2, d = Math.hypot(cx - px, cy - py); if (d < ed) { ed = d; en = q.a; } } return ed < Math.max(6, s * 0.5) ? en : null; };
      return (px, py) => { for (const q of nok) if (Math.hypot(q[0] - px, q[1] - py) < 9) return q[2]; return karisim < 0.5 ? zemBul(px, py) : null; };
  }
  K.appBag = appBag;
  K.ekle({ id: 'uc-00', sinif: 'uc', simdiki: true, ad: '3B bağlar', tarif: 'Uygulamadaki FlowScene: tepeden bakınca ısı haritası; eğdikçe tahmini bağ grafiği her head kendi hücresinin üstünde yükselir. Aşağı sürükle, eğilsin.', yuk: 380, uc: true, kam: { yaw: -0.18, pitch: 0.95, zoom: 0.86 },
    ciz: (x, w, h, ctx, kam, t) => appBag(x, w, h, ctx, kam, t) });

  // Explore ekranı (Keşif): üst şerit + istem + görünüm anahtarı + token şeridi + harita; sağda Seçim ve Sepet; altta lens/dikkat.
  K.ekle({ id: 'isk-00', sinif: 'iskelet', simdiki: true, ad: 'Keşif ekranı', tarif: 'Bugünkü Neşter: üstte adımlar ve sepet, istem satırı, görünüm anahtarı ve token şeridi; solda harita, sağda Seçim ve Sepet, altta lens ve dikkat.',
    dom(kap, ctx) {
      const f = el('div', { class: 'mini', style: 'height:430px;display:grid;grid-template-rows:auto auto auto 1fr auto;gap:6px;font-family:' + HEAD });
      const buton = (t, ana) => el('span', { text: t, style: 'font:600 11px ' + HEAD + ';padding:4px 9px;border-radius:8px;white-space:nowrap;' + (ana ? 'background:' + APP.inc + ';color:' + APP.s0 : 'border:1px solid ' + APP.ctrl + ';color:' + APP.t1) });
      const ust = el('div', { style: 'display:flex;align-items:center;gap:8px;border-bottom:1px solid ' + APP.border + ';padding-bottom:6px;box-shadow:0 6px 10px -10px #000' },
        el('b', { text: 'Neşter', style: 'font:700 14px ' + HEAD }),
        el('span', { style: 'display:flex;gap:6px;font-size:10.5px;color:' + APP.t2 + ';overflow:hidden;white-space:nowrap' }, ...['1 Model', '2 Keşif', '3 Kavram', '4 Karşılaştır', '5 Kaydet'].map((a, i) => el('span', { text: a, style: i === 1 ? 'color:' + APP.t1 + ';border-bottom:2px solid ' + APP.inc : '' }))),
        el('span', { style: 'margin-left:auto;font:600 11px ' + HEAD + ';color:' + APP.basket + ';border:1px solid ' + APP.basket + ';border-radius:999px;padding:2px 9px', text: 'Sepet' }));
      const istem = el('div', { style: 'display:flex;gap:6px' }, el('span', { style: 'flex:1;border:1px solid ' + APP.ctrl + ';border-radius:8px;padding:4px 8px;font:12px ' + HEAD + ';color:' + APP.t1 + ';white-space:nowrap;overflow:hidden' }), buton('Çalıştır', true));
      let gorunum = '2b';
      const sw = el('div', { style: 'display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:11px;color:' + APP.t2 });
      const serit = el('div', { style: 'display:flex;gap:3px;flex-wrap:wrap;font:11px ' + MONO });
      const govde = el('div', { style: 'display:grid;grid-template-columns:minmax(0,1fr) 9.5rem;gap:8px;min-height:0' });
      const ana = el('div', { style: 'position:relative;min-height:0' });
      const yan = el('div', { style: 'display:grid;gap:6px;align-content:start;min-width:0' });
      const secim = el('div', { style: 'background:rgba(18,23,34,.7);backdrop-filter:blur(12px);border:1px solid ' + APP.border + ';border-radius:8px;padding:6px;display:grid;gap:5px' });
      const sepet = el('div', { style: 'border:1px solid ' + APP.border + ';border-radius:8px;padding:6px;display:grid;gap:3px;font-size:11px' });
      yan.append(secim, sepet); govde.append(ana, yan);
      const alt = el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:11px' }, ...['Katman katman tahmin', 'Dikkat deseni'].map((t) => el('div', { style: 'border:1px solid ' + APP.border + ';border-radius:8px;padding:5px 8px;display:flex;justify-content:space-between;color:' + APP.t1 }, el('b', { text: t, style: 'font:600 11px ' + HEAD }), el('span', { text: 'Göster', style: 'color:' + APP.dec }))));
      f.append(ust, istem, el('div', { style: 'display:grid;gap:4px' }, sw, serit), govde, alt); kap.append(f);
      const c = el('canvas', { style: 'position:absolute;inset:0;width:100%;height:100%;cursor:pointer' }); let bul = () => null;
      c.addEventListener('click', (e) => { const r = c.getBoundingClientRect(), k = bul(e.clientX - r.left, e.clientY - r.top); if (k) ctx.degis(k); });
      return function ciz() {
        const S = ctx.S();
        istem.firstChild.textContent = S.istem.metin;
        sw.replaceChildren('Görünüm', el('span', { style: 'display:inline-flex;border:1px solid ' + APP.ctrl + ';border-radius:8px;overflow:hidden' },
          ...[['hot', 'Sıcak noktalar'], ['2b', '2B ısı haritası'], ['list', 'Liste'], ['3d', '3B']].map(([id, a]) => el('span', { text: a, onclick: () => { gorunum = id; ciz(); }, style: 'padding:2px 7px;cursor:pointer;' + (id === gorunum ? 'background:' + APP.t1 + ';color:' + APP.s0 : '') }))),
          'Ölçek', el('span', { text: 'Katman içi', style: 'color:' + APP.t1 }));
        serit.replaceChildren(...S.istem.tokenlar.map((t, i) => el('span', { text: N.tok(t), style: 'padding:1px 5px;border-radius:4px;border:1px solid ' + (i === S.T - 1 ? APP.inc : APP.border) + ';color:' + (i === S.T - 1 ? APP.inc : APP.t2) })));
        ana.replaceChildren();
        if (gorunum === '2b' || gorunum === '3d') {
          ana.append(c);
          requestAnimationFrame(() => { const dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight; if (!w) return; c.width = w * dpr; c.height = h * dpr; const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.fillStyle = APP.s0; x.fillRect(0, 0, w, h); bul = appIsi(x, S, ctx.secili(), new Set(), 0, 0, w, h, 'katman'); });
          if (gorunum === '3d') ana.append(el('span', { text: '3B için “3B bağlar” kartına bak', style: 'position:absolute;left:6px;bottom:6px;font-size:10px;color:' + APP.t2 }));
        } else {
          ana.append(el('ul', { class: 'liste', style: 'height:100%;overflow:auto' }, ...S.sicak.map((q) => el('li', { class: ctx.secili().has(q.k) ? 'on' : '', onclick: () => ctx.degis(q.k) }, el('span', { text: N.ad(q.l, q.h) }), el('span', { text: N.sayi(q.v) })))));
        }
        secim.replaceChildren(el('b', { text: 'Seçim', style: 'font:700 12px ' + HEAD }),
          ctx.secili().size ? el('div', { style: 'display:flex;flex-wrap:wrap;gap:3px' }, ...[...ctx.secili()].map((k) => el('span', { style: 'font:10.5px ' + MONO + ';border:1px solid ' + APP.border + ';border-radius:999px;padding:1px 6px;cursor:pointer', text: N.adK(k) + ' ×', onclick: () => ctx.degis(k) }))) : el('span', { text: 'Haritadan ya da listeden head seçin.', style: 'font-size:10.5px;color:' + APP.t2 }),
          buton('Gerçek etkiyi ölç', true));
        sepet.replaceChildren(el('b', { text: 'Sepet', style: 'font:700 12px ' + HEAD + ';color:' + APP.basket }), el('span', { text: 'Henüz kesim yok.', style: 'color:' + APP.t2 }));
      };
    } });

  // ControlStatsCard: "Gerçek etki" + ölçülen rozeti; KL ± SE, istem sayısı, rastgele kesim bandı, p değerli hüküm.
  K.ekle({ id: 'olc-00', sinif: 'olcum', simdiki: true, ad: 'Gerçek etki kartı', tarif: 'Uygulamadaki ölçüm kartı: önce/sonra olasılığı değil çıktı değişimi (KL) gösterir, rastgele kesimlerle kıyaslayıp anlamlı mı söyler.',
    dom(kap, ctx) {
      const f = el('div', { class: 'mini', style: 'min-height:240px;display:grid;gap:8px;align-content:start;font-family:' + HEAD + ';border:1px solid ' + APP.border + ';border-radius:8px;padding:12px;background:' + APP.s1 });
      kap.append(f);
      return () => {
        const S = ctx.S(), r = S.olc([...ctx.secili()]), se = r.kl * 0.06 + 0.003, lo = 0.012 + 0.01 * ctx.secili().size, hi = lo * 5.5, anlamli = r.kl > hi;
        const n = Math.max(20, 200 - Math.round(r.kl * 40)), p = anlamli ? Math.max(0.001, 0.02 / (1 + r.kl * 10)) : 0.31;
        const fmt = (v) => N.sayi(v, 3);
        f.replaceChildren(
          el('div', { style: 'display:flex;gap:8px;align-items:center' }, el('h4', { text: 'Gerçek etki', style: 'margin:0;font:700 15px ' + HEAD + ';color:' + APP.t1 }), el('span', { text: 'ölçülen', style: 'font:600 10px ' + HEAD + ';color:' + APP.s0 + ';background:' + APP.inc + ';border-radius:999px;padding:1px 7px' })),
          ctx.secili().size ? el('p', { text: anlamli ? 'Bu kesim çıktıyı anlamlı biçimde değiştiriyor.' : 'Bu kesim çıktıyı anlamlı biçimde değiştirmiyor görünüyor: etkisi rastgele seçilen head’lerden ayırt edilemedi.', style: 'margin:0;font-size:13px;color:' + APP.t1 })
            : el('p', { text: 'Kesince bu head’lerin katkısı kapanır; model dosyası değişmez.', style: 'margin:0;font-size:13px;color:' + APP.t2 }),
          el('p', { text: 'Çıktı değişimi (KL) ' + fmt(r.kl) + ' ± ' + fmt(se), style: 'margin:0;font:12px ' + MONO + ';color:' + APP.t1 }),
          el('p', { text: '64 istem', style: 'margin:0;font:12px ' + MONO + ';color:' + APP.t2 }),
          el('p', { text: 'rastgele kesimlerde yaklaşık ' + fmt(lo) + '–' + fmt(hi), style: 'margin:0;font:12px ' + MONO + ';color:' + APP.t2 }),
          ctx.secili().size ? el('p', { text: anlamli ? 'Seçilen head’ler, aynı katmanlardan rastgele seçilen head’lerden daha etkili (p = ' + fmt(p) + ', ' + n + ' rastgele deneme).' : 'p = ' + N.sayi(p, 2) + ' > 0,05: rastgele seçimden ayırt edilemedi (' + n + ' rastgele deneme).', style: 'margin:0;font-size:12px;color:' + APP.t2 }) : null);
      };
    } });

  // LensPanel: Nokta | en olası 3 tahmin | "cevap" olasılığı + çubuk; cevabın ilk tepeye çıktığı satır vurgulu.
  K.ekle({ id: 'lns-00', sinif: 'lens', simdiki: true, ad: 'Katman katman tahmin', tarif: 'Uygulamadaki lens paneli: her katman bir tablo satırı, en olası üç tahmin yüzdeleriyle; son sütunda cevabın olasılık çubuğu.',
    dom(kap, ctx) {
      const f = el('div', { class: 'mini', style: 'height:320px;display:grid;grid-template-rows:auto auto 1fr;gap:6px;font-family:' + HEAD });
      kap.append(f);
      return () => {
        const S = ctx.S(), ln = S.lens([...ctx.secili()]), cevap = S.istem.hedef, ilk = ln.findIndex((r) => r[0].tok === cevap);
        const satir = ln.map((r, l) => el('tr', { style: l === ilk ? 'background:rgba(46,230,197,.12)' : '' },
          el('td', { text: 'K' + l + ' sonrası', style: 'padding:2px 8px 2px 0;color:' + APP.t2 + ';white-space:nowrap' }),
          el('td', { style: 'padding:2px 8px 2px 0;white-space:nowrap' }, ...r.map((t) => el('span', { style: 'margin-right:6px' }, el('code', { text: N.tok(t.tok), style: 'color:' + APP.t1 + ';font:11px ' + MONO }), ' ', el('span', { text: N.yuzde(t.p), style: 'color:' + APP.t2 + ';font:10.5px ' + MONO })))),
          el('td', { style: 'white-space:nowrap' }, el('span', { style: 'display:inline-block;width:60px;height:6px;background:' + APP.border + ';border-radius:3px;vertical-align:middle;margin-right:5px;position:relative;overflow:hidden' },
            el('i', { style: 'position:absolute;inset:0 auto 0 0;width:' + Math.round(((r.find((q) => q.tok === cevap) || { p: 0 }).p) * 100) + '%;background:' + APP.inc })),
            el('span', { text: N.yuzde((r.find((q) => q.tok === cevap) || { p: 0 }).p), style: 'font:10.5px ' + MONO + ';color:' + APP.t1 }))));
        f.replaceChildren(
          el('div', { style: 'display:flex;justify-content:space-between' }, el('b', { text: 'Katman katman tahmin', style: 'font:700 13px ' + HEAD + ';color:' + APP.t1 }), el('span', { text: 'Gizle', style: 'font-size:11px;color:' + APP.dec })),
          el('p', { style: 'margin:0;font-size:11.5px;color:' + APP.t2, text: '“' + S.istem.tokenlar[S.T - 1].trim() + '” token’ından sonra son cevap: “' + cevap.trim() + '”. ' + (ilk >= 0 ? 'İlk kez K' + ilk + ' sonrası en olası tahmin oluyor.' : '') }),
          el('div', { style: 'overflow:auto;border:1px solid ' + APP.border + ';border-radius:6px' }, el('table', { style: 'border-collapse:collapse;font-size:11px;width:100%' },
            el('thead', null, el('tr', null, ...['Nokta', 'O noktadaki en olası 3 tahmin', '“' + cevap.trim() + '” olasılığı'].map((t) => el('th', { text: t, style: 'text-align:left;padding:3px 8px 3px 0;color:' + APP.t2 + ';font-weight:600;position:sticky;top:0;background:' + APP.s1 })))),
            el('tbody', null, ...satir))));
      };
    } });

  // AttentionPanel: özet satırı (en çok bakılan 3 token) + satır okuyan / sütun okunan ızgarası.
  K.ekle({ id: 'dkt-00', sinif: 'lens', simdiki: true, ad: 'Dikkat deseni', tarif: 'Uygulamadaki dikkat paneli: önce son token’ın en çok baktığı yerler cümleyle, altında okuyan × okunan ızgarası turkuaz tonlarıyla.', yuk: 320,
    ciz(x, w, h, ctx) {
      const S = ctx.S(), k = [...ctx.secili()][0] || S.sicak[0].k, { l, h: hh } = N.ac(k), at = S.dikkat(l, hh), son = at.m[S.T - 1];
      x.fillStyle = APP.s0; x.fillRect(0, 0, w, h);
      const en = son.map((p, j) => [p, j]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      x.font = '700 13px ' + HEAD; x.fillStyle = APP.t1; x.textAlign = 'left'; x.fillText('Dikkat deseni', 10, 18);
      x.font = '11.5px ' + HEAD; x.fillStyle = APP.t2;
      x.fillText(N.ad(l, hh) + ', “' + S.istem.tokenlar[S.T - 1].trim() + '” token’ında en çok şunlara bakıyor:', 10, 36);
      x.font = '11px ' + MONO; x.fillStyle = APP.t1; x.fillText(en.map(([p, j]) => N.tok(S.istem.tokenlar[j]) + ' ' + N.yuzde(p)).join('   '), 10, 52);
      const pad = 70, s = Math.min((w - pad - 10) / S.T, (h - 92) / S.T);
      for (let i = 0; i < S.T; i++) {
        x.fillStyle = APP.t2; x.textAlign = 'right'; x.fillText(N.tok(S.istem.tokenlar[i]).slice(0, 8), pad - 6, 66 + i * s + s * 0.62);
        for (let j = 0; j < S.T; j++) {
          x.fillStyle = j <= i ? 'rgba(46,230,197,' + (0.06 + 0.94 * at.m[i][j]).toFixed(3) + ')' : APP.s1;
          x.fillRect(pad + j * s + 1, 66 + i * s + 1, s - 2, s - 2);
        }
      }
      x.textAlign = 'left'; x.fillStyle = APP.t2; x.font = '10.5px ' + HEAD; x.fillText('Satır: okuyan token · sütun: okunan token', pad, h - 6);
    } });
})();
