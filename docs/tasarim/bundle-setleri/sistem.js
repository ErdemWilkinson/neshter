/* Neşter sistemleri: uygulamadaki bütün görünümler ve ekranlar, her bundle'a takılabilen tek modül.
   Görünüm: Sıcak noktalar · 2B ısı haritası · Liste · 3B (üstten ısı → eğilince bağ grafiği)
   Ekranlar: Logit lens · Dikkat · Kavram avı (süpürme) · Karşılaştır (iki cevap + test seti) · Kaydet
   Renk ve yazı yüzü bundle'dan gelir: kök öğe --n-zemin, --n-yuzey, --n-murekkep, --n-soluk, --n-cizgi, --n-vurgu, --n-ikinci, --n-f, --n-m tanımlar. */
(function () {
  'use strict';
  const N = window.Nester, el = N.el;
  const NS = 'http://www.w3.org/2000/svg';

  const CSS = `
  .ns { display: grid; gap: 12px; min-width: 0; font-family: var(--n-f); color: var(--n-murekkep); }
  .ns-sekmeler { display: flex; flex-wrap: wrap; gap: 0 16px; border-bottom: 1px solid var(--n-cizgi); }
  .ns-sekmeler [role=tab] { appearance: none; background: none; border: 0; border-bottom: 2px solid transparent; margin-bottom: -1px; padding: 6px 0; font: inherit; font-size: 13px; color: var(--n-soluk); cursor: pointer; white-space: nowrap; }
  .ns-sekmeler [role=tab]:hover { color: var(--n-murekkep); }
  .ns-sekmeler [role=tab][aria-selected=true] { color: var(--n-murekkep); border-bottom-color: var(--n-vurgu); font-weight: 600; }
  .ns-sekmeler [role=tab]:focus-visible { outline: 2px solid var(--n-vurgu); outline-offset: 2px; }
  .ns-sekmeler .ns-ayrac { width: 1px; background: var(--n-cizgi); margin: 6px 0; }
  .ns-govde { min-width: 0; display: grid; gap: 10px; }
  .ns-not { margin: 0; font-size: 12.5px; color: var(--n-soluk); }
  .ns-m { font-family: var(--n-m); font-variant-numeric: tabular-nums; }
  .ns-arac { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; font-size: 12.5px; color: var(--n-soluk); }
  .ns-arac .ns-sec { cursor: pointer; color: var(--n-soluk); }
  .ns-arac .ns-sec.on { color: var(--n-murekkep); text-decoration: underline; text-decoration-color: var(--n-vurgu); text-decoration-thickness: 2px; text-underline-offset: 4px; }
  .ns-arac .ns-sec:focus-visible { outline: 2px solid var(--n-vurgu); outline-offset: 2px; }
  .ns-arac input[type=range] { accent-color: var(--n-vurgu); width: 9rem; }
  .ns-tuval { position: relative; min-width: 0; }
  .ns-tuval canvas { display: block; width: 100%; touch-action: none; }
  .ns-ipucu { position: absolute; pointer-events: none; background: var(--n-yuzey); border: 1px solid var(--n-cizgi); padding: 4px 8px; border-radius: 4px; font: 400 11.5px var(--n-m); white-space: nowrap; transform: translate(10px, 10px); z-index: 2; }
  .ns-satirlar { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
  .ns-satirlar li { display: grid; grid-template-columns: 4.6rem minmax(0, 1fr) 3rem; gap: 10px; align-items: center; padding: 4px 6px; border-radius: 4px; cursor: pointer; font-size: 12.5px; }
  .ns-satirlar li:hover, .ns-satirlar li:focus-visible { background: color-mix(in srgb, var(--n-murekkep) 6%, transparent); outline: none; }
  .ns-satirlar li.on { background: color-mix(in srgb, var(--n-vurgu) 15%, transparent); }
  .ns-cubuk { height: 8px; border-radius: 4px; background: color-mix(in srgb, var(--n-murekkep) 8%, transparent); position: relative; overflow: hidden; }
  .ns-cubuk i { position: absolute; inset: 0 auto 0 0; background: var(--n-ikinci); border-radius: 4px; }
  .ns-cubuk b { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--n-murekkep); opacity: 0.7; }
  .ns-tablo { overflow-x: auto; }
  .ns-tablo table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
  .ns-tablo th, .ns-tablo td { text-align: left; padding: 4px 12px 4px 0; border-bottom: 1px solid var(--n-cizgi); white-space: nowrap; font-weight: 400; }
  .ns-tablo th { color: var(--n-soluk); font-size: 11.5px; }
  .ns-tablo tbody tr { cursor: pointer; }
  .ns-tablo tbody tr:hover { background: color-mix(in srgb, var(--n-murekkep) 5%, transparent); }
  .ns-tablo tbody tr.on td:first-child { color: var(--n-vurgu); font-weight: 600; }
  .ns-iki { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 10px; }
  .ns-kutu { border: 1px solid var(--n-cizgi); border-radius: 6px; padding: 10px 12px; display: grid; gap: 6px; align-content: start; min-width: 0; background: var(--n-yuzey); }
  .ns-kutu h4 { margin: 0; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--n-soluk); font-weight: 600; }
  .ns-kutu p { margin: 0; font-size: 13.5px; line-height: 1.5; overflow-wrap: anywhere; }
  .ns-iyi { color: var(--n-ikinci); } .ns-kotu { color: var(--n-vurgu); }
  .ns-eylem { color: var(--n-vurgu); font-weight: 600; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; font-size: 13px; }
  .ns-eylem:focus-visible { outline: 2px solid var(--n-vurgu); outline-offset: 2px; }
  .ns-eylem[aria-disabled=true] { color: var(--n-soluk); cursor: default; text-decoration: none; }
  .ns-ilerleme { height: 6px; border-radius: 3px; background: color-mix(in srgb, var(--n-murekkep) 8%, transparent); overflow: hidden; }
  .ns-ilerleme i { display: block; height: 100%; background: var(--n-vurgu); width: 0; transition: width 0.15s linear; }
  .ns-katman { display: flex; align-items: flex-end; gap: 1px; height: 70px; }
  .ns-katman i { flex: 1; background: var(--n-ikinci); border-radius: 1px 1px 0 0; min-height: 1px; opacity: 0.55; }
  .ns-katman i.tepe { background: var(--n-vurgu); opacity: 1; }
  .ns-form { display: grid; gap: 8px; font-size: 13px; }
  .ns-form label { display: grid; gap: 3px; color: var(--n-soluk); font-size: 12px; }
  .ns-form input[type=text] { font: 400 13px var(--n-m); padding: 6px 8px; border: 1px solid var(--n-cizgi); border-radius: 4px; background: var(--n-zemin); color: var(--n-murekkep); min-width: 0; }
  .ns-form input[type=text]:focus { outline: 2px solid var(--n-vurgu); outline-offset: 0; }
  .ns-form fieldset { border: 0; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 4px 14px; }
  .ns-form legend { color: var(--n-soluk); font-size: 12px; padding: 0; margin-bottom: 3px; }
  .ns-form fieldset label { display: flex; gap: 5px; align-items: center; color: var(--n-murekkep); font-size: 13px; }
  .ns-form input[type=radio], .ns-form input[type=checkbox] { accent-color: var(--n-vurgu); }
  .ns-hata { color: var(--n-vurgu); font-size: 12px; }
  .ns-lens { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(3.4rem, 1fr); gap: 2px; overflow-x: auto; padding-bottom: 4px; }
  .ns-lens div { display: grid; gap: 1px; font: 400 11px var(--n-m); text-align: center; }
  .ns-lens span { padding: 3px 2px; border-radius: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ns-lens small { color: var(--n-soluk); }
  @media (prefers-reduced-motion: reduce) { .ns-ilerleme i { transition: none; } }
  `;
  function stilEkle() {
    if (document.getElementById('ns-stil')) return;
    const s = document.createElement('style'); s.id = 'ns-stil'; s.textContent = CSS; document.head.append(s);
  }

  const CEVAP = {
    "Türkiye'nin başkenti": [" Ankara'dır. Ülkenin en kalabalık şehri ise İstanbul'dur.", " İstanbul'dur. Ülkenin en kalabalık şehri de İstanbul'dur."],
    'Kedi süt içti, köpek de': [' su içti. İkisi de bahçede uyudu.', ' süt içti. İkisi de bahçede uyudu.'],
    '2 + 3 =': [' 5. Yani iki ile üçün toplamı beştir.', ' 6. Yani iki ile üçün toplamı altıdır.'],
  };
  const TEST = ["Türkiye'nin başkenti", 'Fransa’nın başkenti', 'Kedi süt içti, köpek de', '2 + 3 =', 'Güneş doğudan'];
  const KAVRAMLAR = [
    { id: 'baskent', ad: 'başkent bilgisi', tepe: 0.62 },
    { id: 'olumsuz', ad: 'olumsuzluk', tepe: 0.45 },
    { id: 'sayi', ad: 'sayı ve işlem', tepe: 0.7 },
    { id: 'nezaket', ad: 'nezaket', tepe: 0.35 },
  ];
  const YONTEM = [['attribution', 'Tahmini etki'], ['activation_diff', 'Etkinlik farkı'], ['direction', 'Davranış yönü']];

  // 3B bağ grafiği: en sıcak head'ler ısı haritasındaki kendi hücrelerinin üstünde; girdi altta, çıktı üstte.
  const grafOnbellek = new WeakMap();
  function bagGrafi(S) {
    if (grafOnbellek.has(S)) return grafOnbellek.get(S);
    const r = N.rng(N.hash('3b|' + S.model.id + S.istem.metin));
    const bas = S.sicak.slice(0, 8).slice().sort((a, b) => a.l - b.l);
    const dug = [{ id: 'girdi', tur: 'girdi', l: -1.6, h: (S.H - 1) / 2, ad: 'girdi' }]
      .concat(bas.map((s) => ({ id: s.k, tur: 'head', l: s.l, h: s.h, v: s.v, ad: N.ad(s.l, s.h) })))
      .concat([{ id: 'cikti', tur: 'cikti', l: S.L + 0.6, h: (S.H - 1) / 2, ad: N.tok(S.istem.hedef) }]);
    const bag = [];
    for (let i = 1; i < dug.length; i++) {
      const once = dug.slice(0, i).filter((d) => d.l < dug[i].l);
      const kaynak = dug[i].tur === 'cikti' ? once.slice(-3) : once.slice().sort(() => r() - 0.5).slice(0, Math.min(once.length, 1 + Math.floor(r() * 2)));
      for (const k of kaynak) bag.push({ id: k.id + '>' + dug[i].id, k: k.id, h: dug[i].id, w: 0.3 + 0.7 * r() * (dug[i].v || 0.8) });
    }
    for (const d of dug) {
      if (d.tur === 'cikti' || bag.some((b) => b.k === d.id)) continue;
      bag.push({ id: d.id + '>cikti', k: d.id, h: 'cikti', w: 0.25 + 0.4 * r() });
    }
    const g = { dugumler: dug, baglar: bag };
    grafOnbellek.set(S, g);
    return g;
  }
  function akis(G, kesik) {
    const ham = (kes) => {
      const a = new Map([['girdi', 1]]);
      for (const d of G.dugumler) {
        if (d.tur === 'girdi') continue;
        let s = 0;
        for (const b of G.baglar) if (b.h === d.id && !kes.has(b.id)) s += b.w * (a.get(b.k) || 0);
        a.set(d.id, s);
      }
      return a;
    };
    const tam = ham(new Set()), simdi = ham(kesik), oran = new Map();
    for (const [id, v] of simdi) oran.set(id, tam.get(id) ? v / tam.get(id) : 0);
    return { oran, cikti: oran.get('cikti') || 0 };
  }

  const KAMERA = { ust: { yaw: 0, pitch: Math.PI / 2, zoom: 1 }, egik: { yaw: -0.18, pitch: 0.95, zoom: 0.86 }, alcak: { yaw: -0.4, pitch: 0.42, zoom: 0.95 } };

  function sistem(kok, ctx) {
    stilEkle();
    const S = () => ctx.S();
    const secili = () => ctx.secili();
    const v = (ad) => getComputedStyle(kok).getPropertyValue(ad).trim();
    const SEKMELER = [['sicak', 'Sıcak noktalar'], ['2b', '2B ısı haritası'], ['liste', 'Liste'], ['3b', '3B'], ['|'],
      ['lens', 'Katman katman'], ['dikkat', 'Dikkat'], ['kavram', 'Kavram avı'], ['kars', 'Karşılaştır'], ['kaydet', 'Kaydet']];
    const d = { sekme: ctx.sekme || '2b', olcek: 'katman', kam: { ...KAMERA.ust }, karisim: 0, kesik: new Set(), uzerinde: null,
      kavram: null, kavramSec: 'baskent', yontem: 'attribution', supurme: 0, kayit: null, kayitAd: '', bicim: 'safetensors', ollama: false };
    let sonS = null;

    kok.classList.add('ns');
    const sekmeler = el('div', { class: 'ns-sekmeler', role: 'tablist', 'aria-label': 'Neşter görünümleri' });
    const govde = el('div', { class: 'ns-govde', role: 'tabpanel' });
    kok.replaceChildren(sekmeler, govde);

    function sekmeCiz() {
      sekmeler.replaceChildren(...SEKMELER.map(([id, ad]) => id === '|' ? el('span', { class: 'ns-ayrac', 'aria-hidden': 'true' })
        : el('button', { type: 'button', role: 'tab', 'aria-selected': String(id === d.sekme), tabindex: id === d.sekme ? '0' : '-1', 'data-id': id, text: ad,
          onclick: () => { d.sekme = id; yenile(); } })));
    }
    sekmeler.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const ids = SEKMELER.map((x) => x[0]).filter((x) => x !== '|');
      const i = (ids.indexOf(d.sekme) + (e.key === 'ArrowRight' ? 1 : ids.length - 1)) % ids.length;
      d.sekme = ids[i]; yenile(); sekmeler.querySelector('[data-id="' + ids[i] + '"]').focus();
    });

    const tikla = (o, f) => { o.tabIndex = 0; o.addEventListener('click', f); o.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); f(); } }); return o; };
    const sec = (ad, on, f) => tikla(el('span', { class: 'ns-sec' + (on ? ' on' : ''), text: ad }), f);
    const rampa = () => N.rampa([v('--n-yuzey') || '#ffffff', v('--n-ikinci') || '#3b82f6', v('--n-vurgu') || '#e0662a']);

    // ── Sıcak noktalar
    function gSicak() {
      const s = S();
      return [el('p', { class: 'ns-not', text: 'Bu istemde en güçlü yazan on head. Tıkla, seçime girsin.' }),
        el('ul', { class: 'ns-satirlar' }, ...s.sicak.map((x) => {
          const c = el('span', { class: 'ns-cubuk' }); const i = el('i'); i.style.width = (x.v * 100) + '%'; c.append(i);
          return tikla(el('li', { class: secili().has(x.k) ? 'on' : '' }, el('span', { class: 'ns-m', text: N.ad(x.l, x.h) }), c, el('span', { class: 'ns-m', text: N.sayi(x.v) })), () => ctx.degis(x.k));
        }))];
    }
    // ── 2B ısı haritası (ölçek: katman içi / tüm model)
    function gIki() {
      const s = S();
      const kap = el('div', { class: 'ns-tuval' }), c = el('canvas', { 'aria-label': '2B ısı haritası: satır katman, sütun head' }), ip = el('div', { class: 'ns-ipucu', hidden: '' });
      c.style.height = Math.max(220, Math.min(420, s.L * 15)) + 'px';
      kap.append(c, ip);
      const olcekli = d.olcek === 'katman' ? (() => {
        const enb = []; for (let l = 0; l < s.L; l++) { let m = 0; for (let h = 0; h < s.H; h++) m = Math.max(m, s.g(l, h)); enb.push(m || 1); }
        return { ...s, g: (l, h) => s.g(l, h) / enb[l] };
      })() : s;
      let hucre = () => null, uz = null;
      const ciz = () => { hucre = N.isiCiz(c, olcekli, { renk: rampa(), secili: secili(), uzerinde: uz, cizgi: v('--n-murekkep'), bosluk: 1.5 }); };
      requestAnimationFrame(ciz);
      kap._ciz = ciz;
      const yer = (e) => { const r = c.getBoundingClientRect(); return hucre(e.clientX - r.left, e.clientY - r.top); };
      c.addEventListener('pointermove', (e) => {
        const o = yer(e); ip.hidden = !o;
        if (o) { const r = kap.getBoundingClientRect(); ip.style.left = (e.clientX - r.left) + 'px'; ip.style.top = (e.clientY - r.top) + 'px'; ip.textContent = N.ad(o.l, o.h) + ' · ' + N.sayi(s.g(o.l, o.h)) + (secili().has(o.k) ? ' · seçili' : ''); }
        const k = o ? o.k : null; if (k !== uz) { uz = k; ciz(); }
      });
      c.addEventListener('pointerleave', () => { ip.hidden = true; uz = null; ciz(); });
      c.addEventListener('click', (e) => { const o = yer(e); if (o) ctx.degis(o.k); });
      return [el('div', { class: 'ns-arac' }, 'Ölçek:', sec('Katman içi', d.olcek === 'katman', () => { d.olcek = 'katman'; yenile(); }), sec('Tüm model', d.olcek === 'tum', () => { d.olcek = 'tum'; yenile(); }),
        el('span', { text: d.olcek === 'katman' ? 'Head çıktı gücü, katman içinde 0–1' : 'Head çıktı gücü, tüm modelde 0–1' })), kap,
        el('p', { class: 'ns-not', text: 'Satır katman (0 üstte), sütun head. Tıkla seç; çerçeveli olanlar seçili. Değerler tahmini.' })];
    }
    // ── Liste
    function gListe() {
      const s = S(), satir = [];
      for (let l = 0; l < s.L; l++) for (let h = 0; h < s.H; h++) satir.push({ l, h, k: N.anahtar(l, h), g: s.g(l, h) });
      satir.sort((a, b) => b.g - a.g);
      return [el('div', { class: 'ns-tablo' }, el('table', null,
        el('thead', null, el('tr', null, ...['Head', 'Katman', 'Head no', 'Güç', 'Sıra'].map((t) => el('th', { text: t })))),
        el('tbody', null, ...satir.slice(0, 40).map((x, i) => tikla(el('tr', { class: secili().has(x.k) ? 'on' : '' },
          el('td', { class: 'ns-m', text: N.ad(x.l, x.h) }), el('td', { class: 'ns-m', text: String(x.l) }), el('td', { class: 'ns-m', text: String(x.h) }),
          el('td', { class: 'ns-m', text: N.sayi(x.g) }), el('td', { class: 'ns-m', text: String(i + 1) })), () => ctx.degis(x.k)))))),
        el('p', { class: 'ns-not', text: 'İlk 40 head, güce göre. Satıra tıkla, seçime girsin.' })];
    }
    // ── 3B: üstten ısı haritası; karışım arttıkça bağ grafiği hücrelerin üstünde yükselir.
    function gUc() {
      const s = S(), G = bagGrafi(s);
      const kap = el('div', { class: 'ns-tuval' }), c = el('canvas', { tabindex: '0', 'aria-label': 'Isı haritası ve 3B bağ grafiği. Sürükleyerek döndür, tekerlekle yakınlaştır; düğüme tıkla seç, bağa tıkla kes.' }), ip = el('div', { class: 'ns-ipucu', hidden: '' });
      c.style.height = 'clamp(320px, 56vh, 520px)'; kap.append(c, ip);
      let isaret = { dug: [], bag: [] }, anim = 0;
      const COL = 1, ROW = 0.62;
      const W = (x, y, z) => [x, y, z];
      const hx = (h) => (h - (s.H - 1) / 2) * COL, lz = (l) => (l - (s.L - 1) / 2) * ROW;
      function ciz() {
        const dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
        if (!w) return;
        c.width = w * dpr; c.height = h * dpr;
        const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, w, h);
        const k = d.kam, cy = Math.cos(k.yaw), sy = Math.sin(k.yaw), cp = Math.cos(k.pitch), sp = Math.sin(k.pitch);
        const dunyaW = s.H * COL, dunyaD = (s.L + 4) * ROW;
        const olcek = Math.min(w * 0.88 / dunyaW, h * 0.86 / (dunyaD * (0.55 + 0.45 * Math.sin(k.pitch)))) * k.zoom;
        const yukseklik = d.karisim * dunyaD * 0.45;
        const P = ([X, Y, Z]) => {
          const xr = X * cy - Z * sy, zr = X * sy + Z * cy;
          return [w / 2 + xr * olcek, h / 2 + (zr * sp - Y * cp) * olcek + yukseklik * cp * olcek * 0.35, zr * cp + Y * sp];
        };
        // Zemin: ısı hücreleri
        const rn = rampa(), solgun = 1 - 0.55 * d.karisim;
        const hucreler = [];
        for (let l = 0; l < s.L; l++) for (let hh = 0; hh < s.H; hh++) {
          const x0 = hx(hh) - COL / 2 + 0.04, x1 = hx(hh) + COL / 2 - 0.04, z0 = lz(l) - ROW / 2 + 0.04, z1 = lz(l) + ROW / 2 - 0.04;
          const kos = [P(W(x0, 0, z0)), P(W(x1, 0, z0)), P(W(x1, 0, z1)), P(W(x0, 0, z1))];
          hucreler.push({ kos, l, h: hh, z: (kos[0][2] + kos[2][2]) / 2 });
        }
        hucreler.sort((a, b) => a.z - b.z);
        const sec = secili(), mur = v('--n-murekkep');
        for (const q of hucreler) {
          x.globalAlpha = solgun; x.fillStyle = rn(s.g(q.l, q.h));
          x.beginPath(); q.kos.forEach((p, i) => (i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]))); x.closePath(); x.fill();
          if (sec.has(N.anahtar(q.l, q.h))) { x.globalAlpha = 1; x.strokeStyle = mur; x.lineWidth = 1.5; x.stroke(); }
        }
        x.globalAlpha = 1;
        // Bağ grafiği
        const a = akis(G, d.kesik), konum = new Map();
        for (const n of G.dugumler) {
          const yuk = n.tur === 'girdi' ? 0.15 : n.tur === 'cikti' ? 1 : 0.2 + 0.8 * (n.l / s.L);
          konum.set(n.id, { zemin: P(W(hx(n.h), 0, lz(n.l))), p: P(W(hx(n.h), yuk * d.karisim * dunyaD * 0.5, lz(n.l))) });
        }
        isaret = { dug: [], bag: [] };
        if (d.karisim > 0.02) {
          const vurgu = v('--n-vurgu'), ikinci = v('--n-ikinci');
          const bagSira = G.baglar.map((b) => ({ b, A: konum.get(b.k).p, B: konum.get(b.h).p })).sort((p, q) => (p.A[2] + p.B[2]) - (q.A[2] + q.B[2]));
          for (const { b, A, B } of bagSira) {
            const kes = d.kesik.has(b.id), gen = kes ? 1.5 : Math.max(1.2, 9 * b.w * (a.oran.get(b.k) || 0) * d.karisim);
            const mx = (A[0] + B[0]) / 2, my = Math.min(A[1], B[1]) - 28 * d.karisim;
            x.globalAlpha = kes ? 0.8 : (d.uzerinde === b.id ? 0.95 : 0.55) * d.karisim;
            x.strokeStyle = kes ? vurgu : ikinci; x.lineWidth = gen; x.setLineDash(kes ? [4, 5] : []);
            x.beginPath(); x.moveTo(A[0], A[1]); x.quadraticCurveTo(mx, my, B[0], B[1]); x.stroke();
            const ornek = []; for (let t = 0; t <= 1.0001; t += 0.1) ornek.push([(1 - t) * (1 - t) * A[0] + 2 * (1 - t) * t * mx + t * t * B[0], (1 - t) * (1 - t) * A[1] + 2 * (1 - t) * t * my + t * t * B[1]]);
            isaret.bag.push({ b, ornek, gen });
          }
          x.setLineDash([]);
          for (const n of G.dugumler) {
            const { zemin, p } = konum.get(n.id), o = n.tur === 'girdi' ? 1 : a.oran.get(n.id) || 0;
            x.globalAlpha = 0.35 * d.karisim; x.strokeStyle = mur; x.lineWidth = 1;
            x.beginPath(); x.moveTo(zemin[0], zemin[1]); x.lineTo(p[0], p[1]); x.stroke();
            const r = n.tur === 'head' ? 5 + 3 * o : 7;
            x.globalAlpha = d.karisim; x.fillStyle = n.tur === 'head' && sec.has(n.id) ? vurgu : n.tur === 'cikti' ? ikinci : mur;
            x.beginPath(); x.arc(p[0], p[1], r, 0, Math.PI * 2); x.fill();
            if (d.uzerinde === n.id) { x.strokeStyle = vurgu; x.lineWidth = 2; x.beginPath(); x.arc(p[0], p[1], r + 4, 0, Math.PI * 2); x.stroke(); }
            x.fillStyle = mur; x.font = '11px ' + (v('--n-m') || 'monospace'); x.textAlign = 'center';
            x.fillText(n.tur === 'cikti' ? n.ad + ' · ' + N.yuzde(a.cikti) : n.ad, p[0], p[1] - r - 6);
            isaret.dug.push({ n, p, r });
          }
        }
        x.globalAlpha = 1;
        durum.textContent = d.karisim > 0.02 ? G.baglar.length + ' bağ · ' + d.kesik.size + ' kesik · çıktıya ulaşan akış ' + N.yuzde(a.cikti) : 'Üstten bakış: zemin ısı haritası. Kaydırıcıyı sağa çek, bağ grafiği yükselsin.';
      }
      function bul(e) {
        const r = c.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
        for (const q of isaret.dug) if (Math.hypot(q.p[0] - px, q.p[1] - py) < q.r + 5) return { tur: 'dug', id: q.n.id, n: q.n };
        let en = null, ed = Infinity;
        for (const q of isaret.bag) for (const o of q.ornek) { const dd = Math.hypot(o[0] - px, o[1] - py); if (dd < Math.max(7, q.gen / 2 + 4) && dd < ed) { ed = dd; en = q.b; } }
        return en ? { tur: 'bag', id: en.id, b: en } : null;
      }
      function kameraGit(hedef) {
        const bas = { ...d.kam }, t0 = performance.now(), sure = N.azalt() ? 0 : 450;
        cancelAnimationFrame(anim);
        const adim = (t) => {
          const f = sure ? Math.min(1, (t - t0) / sure) : 1, e = 1 - Math.pow(1 - f, 3);
          for (const q of ['yaw', 'pitch', 'zoom']) d.kam[q] = bas[q] + (hedef[q] - bas[q]) * e;
          ciz(); if (f < 1) anim = requestAnimationFrame(adim);
        };
        anim = requestAnimationFrame(adim);
      }
      let surukle = null;
      c.addEventListener('pointerdown', (e) => { surukle = { x: e.clientX, y: e.clientY, yaw: d.kam.yaw, pitch: d.kam.pitch, m: 0 }; c.setPointerCapture(e.pointerId); });
      c.addEventListener('pointermove', (e) => {
        if (surukle) {
          surukle.m = Math.max(surukle.m, Math.hypot(e.clientX - surukle.x, e.clientY - surukle.y));
          if (surukle.m > 3) { d.kam.yaw = surukle.yaw + (e.clientX - surukle.x) * 0.008; d.kam.pitch = Math.max(0.25, Math.min(Math.PI / 2, surukle.pitch - (e.clientY - surukle.y) * 0.006)); ciz(); }
          return;
        }
        const o = bul(e), id = o ? o.id : null;
        ip.hidden = !o;
        if (o) {
          const r = kap.getBoundingClientRect(); ip.style.left = (e.clientX - r.left) + 'px'; ip.style.top = (e.clientY - r.top) + 'px';
          ip.textContent = o.tur === 'dug' ? (o.n.tur === 'head' ? o.n.ad + ' · tıkla seç' : o.n.ad) : (d.kesik.has(o.id) ? 'kesik bağ · tıkla geri bağla' : 'bağ · ağırlık ' + N.sayi(o.b.w) + ' · tıkla kes');
        }
        c.style.cursor = o ? 'pointer' : 'grab';
        if (id !== d.uzerinde) { d.uzerinde = id; ciz(); }
      });
      const birak = (e) => {
        if (!surukle) return; const tik = surukle.m <= 3; surukle = null;
        if (!tik || e.type !== 'pointerup') return;
        const o = bul(e);
        if (!o) return;
        if (o.tur === 'dug' && o.n.tur === 'head') ctx.degis(o.id);
        else if (o.tur === 'bag') { d.kesik.has(o.id) ? d.kesik.delete(o.id) : d.kesik.add(o.id); ciz(); }
      };
      c.addEventListener('pointerup', birak); c.addEventListener('pointercancel', birak);
      c.addEventListener('pointerleave', () => { ip.hidden = true; });
      c.addEventListener('wheel', (e) => { if (!e.deltaY) return; e.preventDefault(); d.kam.zoom = Math.max(0.4, Math.min(3, d.kam.zoom * Math.exp(-e.deltaY * 0.0012))); ciz(); }, { passive: false });
      c.addEventListener('keydown', (e) => {
        const m = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key];
        if (m) { e.preventDefault(); d.kam.yaw += m[0]; d.kam.pitch = Math.max(0.25, Math.min(Math.PI / 2, d.kam.pitch + m[1])); ciz(); }
      });
      const kaydirici = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round(d.karisim * 100)), 'aria-label': 'Isı haritası ile bağ grafiği arasında' });
      kaydirici.addEventListener('input', () => {
        const once = d.karisim; d.karisim = kaydirici.value / 100;
        if (once <= 0.02 && d.karisim > 0.02 && d.kam.pitch > 1.4) kameraGit(KAMERA.egik);
        else if (d.karisim <= 0.02 && once > 0.02) kameraGit(KAMERA.ust); else ciz();
      });
      const durum = el('p', { class: 'ns-not' });
      requestAnimationFrame(ciz);
      kap._ciz = ciz;
      return [el('div', { class: 'ns-arac' }, 'Isı', kaydirici, 'Bağlar', el('span', { text: '·' }),
        sec('Üstten', false, () => kameraGit(KAMERA.ust)), sec('Eğik', false, () => kameraGit(KAMERA.egik)), sec('Alçak', false, () => kameraGit(KAMERA.alcak)),
        d.kesik.size ? sec('Kesikleri geri bağla', false, () => { d.kesik.clear(); ciz(); }) : null), kap, durum];
    }
    // ── Logit lens
    function gLens() {
      const s = S(), sec = [...secili()], a = s.lens([]), b = s.lens(sec), rn = rampa();
      const sutun = (rows, etiket) => el('div', { class: 'ns-lens', role: 'list', 'aria-label': etiket }, ...rows.map((row, l) => el('div', { role: 'listitem' },
        el('small', { text: 'K' + l }), ...row.slice(0, 2).map((t) => { const sp = el('span', { text: N.tok(t.tok), title: N.yuzde(t.p, 1) }); sp.style.background = rn(Math.min(1, t.p * 1.3)); sp.style.color = t.p > 0.45 ? v('--n-yuzey') : v('--n-murekkep'); return sp; }))));
      return [el('p', { class: 'ns-not', text: 'Her katmanda modelin o anki en iyi iki tahmini. Renk koyulaştıkça olasılık artar.' }),
        el('h4', { class: 'ns-not', text: 'Hiçbir şey kapalı değilken' }), sutun(a, 'Kesimsiz'),
        sec.length ? el('h4', { class: 'ns-not', text: 'Seçili ' + sec.length + ' head kapalıyken' }) : null, sec.length ? sutun(b, 'Kesimli') : el('p', { class: 'ns-not', text: 'Bir head seçersen kapalı hâli de burada görünür.' })];
    }
    // ── Dikkat
    function gDikkat() {
      const s = S(), k = [...secili()][0] || s.sicak[0].k, { l, h } = N.ac(k), at = s.dikkat(l, h), T = s.T;
      const c = el('canvas', { 'aria-label': N.adK(k) + ' dikkat deseni: ' + at.tur }), kap = el('div', { class: 'ns-tuval', style: 'max-width:30rem' }, c);
      const pad = 78, hc = Math.max(18, Math.min(34, 360 / T));
      c.style.height = (T * hc + pad * 0.4) + 'px';
      requestAnimationFrame(() => {
        const dpr = devicePixelRatio || 1, w = c.clientWidth, hh = c.clientHeight; c.width = w * dpr; c.height = hh * dpr;
        const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0);
        const rn = rampa(), hs = Math.min(hc, (w - pad) / T);
        x.font = '11px ' + v('--n-m'); x.textBaseline = 'middle'; x.fillStyle = v('--n-soluk'); x.textAlign = 'right';
        for (let i = 0; i < T; i++) {
          x.fillStyle = v('--n-soluk'); x.fillText(N.tok(s.istem.tokenlar[i]).slice(0, 8), pad - 6, i * hs + hs / 2);
          for (let j = 0; j <= i; j++) { x.fillStyle = rn(at.m[i][j]); x.fillRect(pad + j * hs + 1, i * hs + 1, hs - 2, hs - 2); }
        }
      });
      return [el('div', { class: 'ns-arac' }, 'Head:', ...[...new Set([k, ...secili(), ...s.sicak.slice(0, 4).map((q) => q.k)])].slice(0, 6).map((q) => sec(N.adK(q), q === k, () => { if (!secili().has(q)) ctx.degis(q); }))),
        el('p', { class: 'ns-not', text: N.adK(k) + ' · desen: ' + at.tur + '. Satır sorgu, sütun anahtar belirteci.' }), kap];
    }
    // ── Kavram avı (tahmini etki / etkinlik farkı / davranış yönü + süpürme)
    function kavramHesap(s, kv, yontem) {
      const r = N.rng(N.hash('kavram|' + s.model.id + kv.id + yontem));
      const tepe = Math.round(kv.tepe * (s.L - 1));
      const skor = Array.from({ length: s.L + 1 }, (_, l) => Math.max(0.03, Math.exp(-Math.pow((l - 1 - tepe) / (s.L * 0.15), 2)) * (0.75 + 0.25 * r())));
      const bas = [];
      for (let i = 0; i < 6; i++) { const l = Math.max(0, Math.min(s.L - 1, tepe + Math.round((r() - 0.6) * s.L * 0.3))); const h = Math.floor(r() * s.H); bas.push({ k: N.anahtar(l, h), skor: 0.9 - i * 0.11 - r() * 0.05, dusus: l <= tepe ? 0.1 + 0.4 * r() : null, kl: 0.01 + 0.08 * r() }); }
      return { kv, yontem, tepe, skor, headler: bas, dogruluk: 0.86 + 0.1 * r(), n: 26 };
    }
    function gKavram() {
      const s = S(), kv = KAVRAMLAR.find((q) => q.id === d.kavramSec);
      const ust = [el('div', { class: 'ns-arac' }, 'Kavram:', ...KAVRAMLAR.map((q) => sec(q.ad, q.id === d.kavramSec, () => { d.kavramSec = q.id; d.kavram = null; yenile(); }))),
        el('div', { class: 'ns-arac' }, 'Yöntem:', ...YONTEM.map(([id, ad]) => sec(ad, id === d.yontem, () => { d.yontem = id; d.kavram = null; yenile(); })),
          el('span', { text: d.yontem === 'attribution' ? 'GPU’da önerilen' : '' }))];
      if (!d.kavram || d.kavram.s !== s) {
        const cubuk = el('div', { class: 'ns-ilerleme' }, el('i')), yazi = el('p', { class: 'ns-not', text: 'Örnekler hazır: 32 gösteren, 32 göstermeyen istek (demo).' });
        const ara = tikla(el('span', { class: 'ns-eylem', role: 'button', text: 'Ara' }), () => {
          if (ara.getAttribute('aria-disabled') === 'true') return;
          ara.setAttribute('aria-disabled', 'true');
          const asama = d.yontem === 'direction' ? [[0.3, 'Örnekler okunuyor'], [0.8, 'Katmanlar ölçülüyor'], [0.9, 'Süpürme yapılıyor'], [1, 'Ölçüm tamamlandı']] : [[0.4, 'Örnekler okunuyor'], [0.9, 'Head’ler ölçülüyor'], [1, 'Ölçüm tamamlandı']];
          const t0 = performance.now(), sure = N.azalt() ? 300 : 1600;
          const adim = (t) => {
            const f = Math.min(1, (t - t0) / sure); cubuk.firstChild.style.width = (f * 100) + '%';
            yazi.textContent = (asama.find((q) => f <= q[0]) || asama[asama.length - 1])[1] + ' · %' + Math.round(f * 100);
            if (f < 1) requestAnimationFrame(adim); else { d.kavram = { s, sonuc: kavramHesap(s, kv, d.yontem) }; d.supurme = 0; yenile(); }
          };
          requestAnimationFrame(adim);
        });
        return [...ust, el('p', { class: 'ns-not', text: 'Bir davranışı örneklerle tarif et; Neşter onu modelin içinde arasın.' }), ara, cubuk, yazi];
      }
      const R = d.kavram.sonuc, enb = Math.max(...R.skor);
      const grafik = el('div', { class: 'ns-katman', role: 'img', 'aria-label': 'Katman skorları; tepe K' + R.tepe + ' sonrası' }, ...R.skor.map((q, i) => { const b = el('i', { class: i - 1 === R.tepe ? 'tepe' : '', title: (i ? 'K' + (i - 1) + ' sonrası' : 'gömme') + ' · ' + N.sayi(q) }); b.style.height = (q / enb * 100) + '%'; return b; }));
      const sonuc = [el('div', { class: 'ns-kutu' }, el('h4', { text: 'Katman skorları' }), grafik,
        el('p', { class: 'ns-not', text: 'Tepe: K' + R.tepe + ' sonrası · doğrulamada doğruluk %' + Math.round(R.dogruluk * 100) + ' (n=' + R.n + ')' }))];
      if (R.yontem === 'direction') {
        const kay = el('input', { type: 'range', min: '-20', max: '20', value: String(Math.round(d.supurme * 10)), 'aria-label': 'Yön gücü' });
        const p = 1 / (1 + Math.exp(-(Math.log(s.once / (1 - s.once)) - d.supurme * 1.6)));
        kay.addEventListener('input', () => { d.supurme = kay.value / 10; yenile(); });
        sonuc.push(el('div', { class: 'ns-kutu' }, el('h4', { text: 'Bulunan yön' }), el('p', { text: 'Bulunduğu nokta: K' + R.tepe + ' sonrası' }),
          el('div', { class: 'ns-arac' }, 'Yön gücü', kay, el('span', { class: 'ns-m', text: (d.supurme >= 0 ? '+' : '') + N.sayi(d.supurme, 1) })),
          el('p', { text: '“' + s.istem.hedef.trim() + '” olasılığı ' + N.yuzde(s.once) + ' → ' + N.yuzde(p) }),
          el('p', { class: 'ns-not', text: 'Sağa çektikçe yön çıkarılır, sola çektikçe eklenir.' })));
      } else {
        sonuc.push(el('div', { class: 'ns-kutu' }, el('h4', { text: 'Bulunan head’ler' }), el('div', { class: 'ns-tablo' }, el('table', null,
          el('thead', null, el('tr', null, ...['Head', 'Tahmini skor', 'Ölçülen etki'].map((t) => el('th', { text: t })))),
          el('tbody', null, ...R.headler.map((q) => tikla(el('tr', { class: secili().has(q.k) ? 'on' : '' }, el('td', { class: 'ns-m', text: N.adK(q.k) }), el('td', { class: 'ns-m', text: N.sayi(q.skor) }),
            el('td', { class: 'ns-m', text: q.dusus == null ? 'sinyali etkileyemez · KL ' + N.sayi(q.kl, 3) : 'sinyal düşüşü ' + N.yuzde(q.dusus) + ' · KL ' + N.sayi(q.kl, 3) })), () => ctx.degis(q.k)))))),
          el('p', { class: 'ns-not', text: 'Satıra tıkla: haritada seçilir ve ölçülür.' })));
      }
      return [...ust, el('div', { class: 'ns-iki' }, ...sonuc), tikla(el('span', { class: 'ns-eylem', role: 'button', text: 'Yeniden ara' }), () => { d.kavram = null; yenile(); })];
    }
    // ── Karşılaştır: iki cevap + test seti + sepetteki kesimler
    function gKars() {
      const s = S(), sec = [...secili()], r = s.olc(sec), cv = CEVAP[s.istem.metin] || [s.istem.hedef + ' …', s.istem.rakip + ' …'];
      const orj = s.istem.metin + cv[0], kes = s.istem.metin + (r.degisti ? cv[1] : cv[0]);
      const sepet = (ctx.sepet && ctx.sepet()) || [];
      const test = TEST.map((m) => { const s2 = N.oturum(s.model.id, m), q = s2.olc(sec); return { m, once: q.once, sonra: q.sonra, degisti: q.degisti }; });
      return [el('p', { class: 'ns-not', text: sec.length ? 'Karşılaştırılan kesim: ' + sec.map(N.adK).join(' + ') : 'Seçim boş; önce bir head seç. Şimdilik iki cevap aynı.' }),
        el('div', { class: 'ns-iki' },
          el('div', { class: 'ns-kutu' }, el('h4', { text: 'Orijinal' }), el('p', { text: orj }), el('p', { class: 'ns-not ns-iyi', text: 'hedef davranış var' })),
          el('div', { class: 'ns-kutu' }, el('h4', { text: 'Kesilmiş' }), el('p', { text: kes }), el('p', { class: 'ns-not ' + (r.degisti ? 'ns-kotu' : 'ns-iyi'), text: r.degisti ? 'hedef davranış yok' : 'hedef davranış var' }))),
        el('p', { class: 'ns-not', text: 'sıcaklık 0 · seed 42 (aynı soruya hep aynı cevap) · iki cevap birlikte üretildi' }),
        el('div', { class: 'ns-kutu' }, el('h4', { text: 'Test seti' }), el('div', { class: 'ns-tablo' }, el('table', null,
          el('thead', null, el('tr', null, ...['İstek', 'Orijinal', 'Kesilmiş', 'Fark'].map((t) => el('th', { text: t })))),
          el('tbody', null, ...test.map((q) => el('tr', null, el('td', { text: q.m }), el('td', { class: 'ns-m', text: N.yuzde(q.once) }),
            el('td', { class: 'ns-m' + (q.degisti ? ' ns-kotu' : ''), text: N.yuzde(q.sonra) }), el('td', { class: 'ns-m', text: Math.round((q.sonra - q.once) * 100) + ' puan' }))))))),
        sepet.length ? el('div', { class: 'ns-kutu' }, el('h4', { text: 'Sepetteki kesimler' }), ...sepet.map((q) => {
          const c = el('span', { class: 'ns-cubuk' }); const i = el('i'); i.style.width = (q.sonra * 100) + '%'; const b = el('b'); b.style.left = (q.once * 100) + '%'; c.append(i, b);
          return el('div', { style: 'display:grid;grid-template-columns:minmax(0,9rem) 1fr 3rem;gap:8px;align-items:center;font-size:12.5px' },
            el('span', { class: 'ns-m', style: 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap', text: (q.ad ? q.ad + ' ' : '') + q.headler.map(N.adK).join('+') }), c, el('span', { class: 'ns-m', text: N.yuzde(q.sonra) }));
        })) : null];
    }
    // ── Kaydet
    function adKontrol(a) {
      if (!a.trim()) return 'Bir ad girin.';
      if (/[^A-Za-z0-9._-]/.test(a)) return 'Dosya adında Türkçe harf, boşluk veya özel karakter olamaz; yalnızca A-Z, 0-9, nokta, tire ve alt çizgi.';
      if (/^(con|prn|aux|nul|com\d|lpt\d)(\.|$)/i.test(a)) return 'Bu ad Windows’ta ayrılmış; başka bir ad seçin.';
      if (/[. ]$/.test(a)) return 'Ad nokta veya boşlukla bitemez.';
      return '';
    }
    function gKaydet() {
      const s = S(), sepet = (ctx.sepet && ctx.sepet()) || [], kesimler = sepet.length ? sepet : (secili().size ? [{ headler: [...secili()] }] : []);
      if (d.kayit) {
        const k = d.kayit;
        return [el('div', { class: 'ns-kutu' }, el('h4', { text: 'Kaydedildi' }),
          el('p', { class: 'ns-m', text: 'Konum: ' + k.yol }), el('p', { class: 'ns-m', text: 'Ad: ' + k.ad + ' · ' + k.bicim + (k.ollama ? ' · Ollama' : '') }),
          el('p', { class: 'ns-m', text: 'Orijinal modelin parmak izi (SHA-256): ' + k.sha }), el('p', { class: 'ns-not ns-iyi', text: 'Orijinal dosyaya dokunulmadı. (Demo: diske bir şey yazılmadı.)' })),
          tikla(el('span', { class: 'ns-eylem', role: 'button', text: 'Yeni bir kayıt yap' }), () => { d.kayit = null; yenile(); })];
      }
      if (!d.kayitAd) d.kayitAd = s.model.id + '-kesik';
      const ad = el('input', { type: 'text', id: 'ns-kayit-ad', value: d.kayitAd, autocomplete: 'off', spellcheck: 'false' });
      const hata = el('span', { class: 'ns-hata', text: adKontrol(d.kayitAd) });
      ad.addEventListener('input', () => { d.kayitAd = ad.value; hata.textContent = adKontrol(ad.value); });
      const bicim = el('fieldset', null, el('legend', { text: 'Biçim' }), ...[['safetensors', 'safetensors'], ['gguf_q4_k_m', 'GGUF Q4_K_M'], ['gguf_q8_0', 'GGUF Q8_0']].map(([id, a]) => {
        const r = el('input', { type: 'radio', name: 'ns-bicim', value: id }); r.checked = d.bicim === id; r.addEventListener('change', () => { d.bicim = id; });
        return el('label', null, r, a + (id.startsWith('gguf') ? ' (llama.cpp gerekli)' : ''));
      }));
      const ol = el('input', { type: 'checkbox' }); ol.checked = d.ollama; ol.addEventListener('change', () => { d.ollama = ol.checked; });
      const kaydet = tikla(el('span', { class: 'ns-eylem', role: 'button', text: 'Kaydet · 1,0 GB gerekli, 212 GB boş' }), () => {
        if (!kesimler.length || adKontrol(d.kayitAd)) { hata.textContent = kesimler.length ? adKontrol(d.kayitAd) : 'Kaydedilecek kesim yok.'; return; }
        const h = N.hash(s.model.id).toString(16).padStart(8, '0');
        d.kayit = { ad: d.kayitAd, bicim: d.bicim, ollama: d.ollama, yol: 'D:\\Modeller\\' + d.kayitAd, sha: (h + h.split('').reverse().join('')).repeat(4).slice(0, 64) };
        yenile();
      });
      return [el('p', { class: 'ns-not', text: 'Kaydedilecek kesim: ' + kesimler.length + (kesimler.length ? ' · ' + kesimler.map((q) => q.headler.map(N.adK).join('+')).join(' · ') : '') }),
        el('div', { class: 'ns-form' }, el('label', { for: 'ns-kayit-ad' }, 'Ad', ad), hata,
          el('label', null, 'Klasör', el('input', { type: 'text', value: 'D:\\Modeller', readonly: '' })), bicim,
          el('label', { style: 'display:flex;gap:6px;align-items:center;color:var(--n-murekkep)' }, ol, 'Ollama’ya da kaydet'),
          el('p', { class: 'ns-not', text: 'Tarif (.nester.json), model kartı ve lisans her zaman kaydedilir. Orijinal dosya salt okunur; kayıttan sonra parmak izi (SHA-256) kontrol edilir.' }), kaydet)];
    }

    const GOVDE = { sicak: gSicak, '2b': gIki, liste: gListe, '3b': gUc, lens: gLens, dikkat: gDikkat, kavram: gKavram, kars: gKars, kaydet: gKaydet };
    function yenile() {
      const s = S();
      if (!s) return;
      if (s !== sonS) { sonS = s; d.kesik.clear(); d.kavram = null; d.kayit = null; }
      sekmeCiz();
      const odak = document.activeElement && govde.contains(document.activeElement) && document.activeElement.id;
      govde.replaceChildren(...GOVDE[d.sekme]().filter(Boolean));
      if (odak) { const o = document.getElementById(odak); if (o) o.focus(); }
    }
    const ro = new ResizeObserver(() => { const t = govde.querySelector('.ns-tuval'); if (t && t._ciz) t._ciz(); });
    ro.observe(kok);
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => yenile());
    new MutationObserver(() => yenile()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    yenile();
    return { yenile, git: (id) => { d.sekme = id; yenile(); }, karisim: (m) => { d.karisim = m; if (m > 0.02) d.kam = { ...KAMERA.egik }; yenile(); } };
  }

  N.sistem = sistem;
  N.bagGrafi = bagGrafi;
})();
