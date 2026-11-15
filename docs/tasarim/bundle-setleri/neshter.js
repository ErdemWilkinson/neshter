/* Neşter demo çekirdeği: Neşter'in veri biçiminde, tohumlu ve tekrarlanabilir sahte model.
   Gerçek model yok; ama her sayı aynı girdide aynı çıkar ve kesimler birbiriyle tutarlı. */
(function () {
  'use strict';

  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hash(str) {
    let h = 2166136261;
    for (const c of String(str)) { h ^= c.codePointAt(0); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  const sig = (x) => 1 / (1 + Math.exp(-x));

  const MODELLER = [
    { id: 'qwen05', ad: 'Qwen2.5-0.5B-Instruct', kisa: 'Qwen 0.5B', L: 24, H: 14, aygit: 'CUDA', boyut: '0,99 GB' },
    { id: 'gpt2', ad: 'GPT-2 small', kisa: 'GPT-2', L: 12, H: 12, aygit: 'CPU', boyut: '0,52 GB' },
    { id: 'coder05', ad: 'Qwen2.5-Coder-0.5B', kisa: 'Coder 0.5B', L: 24, H: 14, aygit: 'CUDA', boyut: '0,99 GB' },
  ];

  const HAZIR = [
    { metin: "Türkiye'nin başkenti", tokenlar: ['Türk', 'iye', "'nin", ' baş', 'kenti'], hedef: ' Ankara', rakip: ' İstanbul',
      ozellik: ['ülke adı', 'Türkiye', 'iyelik', 'başkent', 'şehir', 'Ankara yolu', 'konum'] },
    { metin: 'Kedi süt içti, köpek de', tokenlar: ['Ked', 'i', ' süt', ' iç', 'ti', ',', ' köpek', ' de'], hedef: ' su', rakip: ' süt',
      ozellik: ['hayvan', 'içecek', 'geçmiş zaman', 'tekrar', 'karşıtlık', 'sıvı', 'özne'] },
    { metin: '2 + 3 =', tokenlar: ['2', ' +', ' 3', ' ='], hedef: ' 5', rakip: ' 6',
      ozellik: ['sayı', 'toplama', 'işlenen', 'eşittir', 'sonuç', 'basamak', 'tek sayı'] },
  ];
  const COP = [' ve', ' bir', ' de', ',', ' bu', ' olan', '.', ' çok', ' için', ' ile'];

  function belirtecle(metin) {
    const parca = String(metin).match(/\s?[^\s]{1,4}/g) || [];
    return parca.slice(0, 14);
  }
  function istem(metin) {
    const m = String(metin || '').trim();
    const h = HAZIR.find((x) => x.metin.toLocaleLowerCase('tr') === m.toLocaleLowerCase('tr'));
    if (h) return h;
    const tok = belirtecle(m);
    return { metin: m, tokenlar: tok.length ? tok : ['…'], hedef: ' ve', rakip: ' bir',
      ozellik: ['sözcük', 'bağlaç', 'özne', 'konu', 'sıra', 'yakın bağlam', 'sonraki sözcük'] };
  }

  const anahtar = (l, h) => l + '.' + h;
  const ac = (k) => { const [l, h] = String(k).split('.').map(Number); return { l, h }; };
  const ad = (l, h) => 'K' + l + '·H' + h;
  const adK = (k) => { const { l, h } = ac(k); return ad(l, h); };

  function oturum(modelId, metin) {
    const model = MODELLER.find((m) => m.id === modelId) || MODELLER[0];
    const ist = istem(metin);
    const L = model.L, H = model.H, T = ist.tokenlar.length;
    const r = rng(hash(model.id + '|' + ist.metin));

    // Isı: katman boyunca yükselen taban + birkaç gerçekten sıcak head.
    const guc = new Float32Array(L * H);
    for (let l = 0; l < L; l++) {
      const egim = 0.35 + 0.65 * sig((l - L * 0.45) / (L * 0.12));
      for (let h = 0; h < H; h++) guc[l * H + h] = 0.04 + 0.2 * r() * egim;
    }
    const sicakSay = 5 + Math.floor(r() * 3);
    for (let i = 0; i < sicakSay; i++) {
      const l = Math.min(L - 1, Math.floor(L * (0.4 + 0.58 * r())));
      const h = Math.floor(r() * H);
      guc[l * H + h] = Math.max(guc[l * H + h], 0.68 + 0.32 * r());
      const yan = (h + 1 + Math.floor(r() * (H - 1))) % H;
      guc[l * H + yan] = Math.max(guc[l * H + yan], 0.3 + 0.25 * r());
    }
    const g = (l, h) => guc[l * H + h];

    const tum = [];
    for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) tum.push({ l, h, k: anahtar(l, h), v: g(l, h) });
    tum.sort((a, b) => b.v - a.v);
    const sicak = tum.slice(0, 10);

    const once = 0.5 + 0.25 * r();
    const rakipOnce = 0.08 + 0.08 * r();

    function etkiOf(anahtarlar, enCok) {
      let kalan = 1;
      for (const k of anahtarlar) {
        const { l, h } = ac(k);
        if (!(l >= 0 && l < L && h >= 0 && h < H)) continue;
        if (enCok != null && l > enCok) continue;
        kalan *= 1 - 0.62 * Math.pow(g(l, h), 1.4);
      }
      return 1 - kalan;
    }

    function olc(anahtarlar) {
      const liste = [...new Set(anahtarlar || [])];
      const etki = etkiOf(liste);
      const sonra = once * (1 - etki) + 0.015;
      const rakipSonra = rakipOnce + (once - sonra) * 0.55;
      const kl = Math.max(0, -Math.log(1 - 0.92 * etki) * 0.6);
      return {
        headler: liste, etki, once, sonra, rakipOnce, rakipSonra, kl,
        tahmin: sonra >= rakipSonra ? ist.hedef : ist.rakip,
        degisti: sonra < rakipSonra,
      };
    }

    function lens(kapali) {
      const k = [...(kapali || [])];
      const satir = [];
      for (let l = 0; l < L; l++) {
        const kalan = 1 - etkiOf(k, l);
        const pt = once * sig((l - L * 0.62) / (L * 0.07)) * kalan;
        const pr = rakipOnce * sig((l - L * 0.5) / (L * 0.1)) + (once * sig((l - L * 0.62) / (L * 0.07)) - pt) * 0.55;
        const cop = COP[(l * 3 + T) % COP.length];
        const pc = Math.max(0.02, 0.42 * (1 - l / L) * (0.7 + 0.3 * Math.sin(l + T)));
        const ust = [{ tok: ist.hedef, p: pt }, { tok: ist.rakip, p: pr }, { tok: cop, p: pc }].sort((a, b) => b.p - a.p);
        satir.push(ust);
      }
      return satir;
    }

    function dikkat(l, h) {
      const rr = rng(hash(model.id + ist.metin + l + ':' + h));
      const tur = (l * 7 + h * 3) % 4; // 0 önceki belirteç, 1 ilk belirteç, 2 içerik, 3 dağınık
      const icerik = Math.min(T - 1, Math.max(0, Math.floor(T * 0.6)));
      const m = [];
      for (let i = 0; i < T; i++) {
        const sat = [];
        let top = 0;
        for (let j = 0; j <= i; j++) {
          let s = rr() * 0.6;
          if (tur === 0 && j === i - 1) s += 3;
          if (tur === 1 && j === 0) s += 3;
          if (tur === 2 && j === icerik) s += 3 * g(l, h) + 1;
          const e = Math.exp(s);
          sat.push(e); top += e;
        }
        for (let j = 0; j < T; j++) sat[j] = j <= i ? sat[j] / top : 0;
        m.push(sat);
      }
      return { tur: ['önceki belirteç', 'ilk belirteç', 'içerik', 'dağınık'][tur], m };
    }

    // Bağ grafiği: girdi belirteçleri → özellik katmanları → çıktı.
    const graf = (function () {
      const rg = rng(hash('graf|' + model.id + ist.metin));
      const bantlar = [0.2, 0.42, 0.62, 0.8].map((f) => Math.round(f * (L - 1)));
      const dug = [];
      for (let i = 0; i < T; i++) dug.push({ id: 'g' + i, katman: -1, konum: i, etiket: ist.tokenlar[i].trim() || '␣', tur: 'girdi', guc: 1 });
      let oz = 0;
      bantlar.forEach((kat, bi) => {
        const say = bi < 2 ? 3 : 2;
        for (let j = 0; j < say; j++) {
          const konum = Math.min(T - 1, Math.max(0, Math.round((T - 1) * (0.3 + 0.7 * (bi / 3)) - j * rg() * 2)));
          dug.push({ id: 'o' + bi + '_' + j, katman: kat, konum, etiket: ist.ozellik[oz++ % ist.ozellik.length], tur: 'ozellik', guc: 0.4 + 0.6 * rg(), bant: bi });
        }
      });
      dug.push({ id: 'c', katman: L, konum: T - 1, etiket: ist.hedef.trim(), tur: 'cikti', guc: 1 });
      const katmanlar = [dug.filter((d) => d.tur === 'girdi')];
      for (let bi = 0; bi < bantlar.length; bi++) katmanlar.push(dug.filter((d) => d.bant === bi));
      katmanlar.push([dug[dug.length - 1]]);
      const bag = [];
      for (let s = 1; s < katmanlar.length; s++) {
        for (const hedefD of katmanlar[s]) {
          const kaynak = [...katmanlar[s - 1]].sort(() => rg() - 0.5).slice(0, Math.min(katmanlar[s - 1].length, s === 1 ? 3 : 2));
          for (const kd of kaynak) bag.push({ id: kd.id + '>' + hedefD.id, k: kd.id, h: hedefD.id, w: 0.25 + 0.75 * rg() });
        }
      }
      // Her düğümün en az bir çıkışı olsun.
      for (let s = 0; s < katmanlar.length - 1; s++) {
        for (const d of katmanlar[s]) {
          if (!bag.some((b) => b.k === d.id)) {
            const hd = katmanlar[s + 1][Math.floor(rg() * katmanlar[s + 1].length)];
            bag.push({ id: d.id + '>' + hd.id, k: d.id, h: hd.id, w: 0.2 + 0.4 * rg() });
          }
        }
      }
      return { dugumler: dug, baglar: bag, bantlar };
    })();

    function akis(kesik) {
      const kes = kesik instanceof Set ? kesik : new Set(kesik || []);
      const ham = (kesSet) => {
        const a = new Map();
        for (const d of graf.dugumler) if (d.tur === 'girdi') a.set(d.id, 1);
        for (const d of graf.dugumler) {
          if (d.tur === 'girdi') continue;
          let s = 0;
          for (const b of graf.baglar) if (b.h === d.id && !kesSet.has(b.id)) s += b.w * (a.get(b.k) || 0);
          a.set(d.id, s);
        }
        return a;
      };
      const tam = ham(new Set());
      const simdi = ham(kes);
      const oran = new Map();
      for (const [id, v] of simdi) oran.set(id, tam.get(id) ? v / tam.get(id) : 0);
      const cikti = oran.get('c') || 0;
      return { oran, cikti, tahminP: once * cikti + 0.015 };
    }

    return { model, istem: ist, L, H, T, guc, g, sicak, once, rakipOnce, olc, lens, dikkat, graf, akis };
  }

  // Renk rampası: hex duraklar → v∈[0,1] için rgb.
  function rampa(duraklar) {
    const cs = duraklar.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
    return function (v) {
      const x = Math.max(0, Math.min(1, v)) * (cs.length - 1);
      const i = Math.min(cs.length - 2, Math.floor(x));
      const f = x - i;
      const c = cs[i].map((a, j) => Math.round(a + (cs[i + 1][j] - a) * f));
      return 'rgb(' + c.join(',') + ')';
    };
  }
  const css = (ad, el) => getComputedStyle(el || document.documentElement).getPropertyValue(ad).trim();

  // Isı haritası: satır = katman (0 üstte), sütun = head. Döner: hücre(x,y) → {l,h}|null.
  function isiCiz(tuval, S, o) {
    o = o || {};
    const dpr = window.devicePixelRatio || 1;
    const w = tuval.clientWidth, h = tuval.clientHeight;
    if (!w || !h) return () => null;
    tuval.width = Math.round(w * dpr); tuval.height = Math.round(h * dpr);
    const c = tuval.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    const pad = o.pad || 0;
    const cw = (w - pad) / S.H, ch = h / S.L, b = o.bosluk == null ? 1 : o.bosluk;
    const renk = o.renk || ((v) => 'rgba(0,0,0,' + v + ')');
    for (let l = 0; l < S.L; l++) for (let hh = 0; hh < S.H; hh++) {
      c.fillStyle = renk(S.g(l, hh));
      c.fillRect(pad + hh * cw + b / 2, l * ch + b / 2, cw - b, ch - b);
    }
    if (o.secili) {
      c.lineWidth = 2; c.strokeStyle = o.cizgi || '#000';
      for (const k of o.secili) { const { l, h: hh } = ac(k); c.strokeRect(pad + hh * cw + 1.5, l * ch + 1.5, cw - 3, ch - 3); }
    }
    if (o.uzerinde) {
      const { l, h: hh } = ac(o.uzerinde);
      c.lineWidth = 1; c.strokeStyle = o.cizgi || '#000'; c.setLineDash([3, 2]);
      c.strokeRect(pad + hh * cw + 0.5, l * ch + 0.5, cw - 1, ch - 1); c.setLineDash([]);
    }
    return function (x, y) {
      const hh = Math.floor((x - pad) / cw), l = Math.floor(y / ch);
      return l >= 0 && l < S.L && hh >= 0 && hh < S.H ? { l, h: hh, k: anahtar(l, hh) } : null;
    };
  }

  const yuzde = (x, ondalik) => '%' + (x * 100).toLocaleString('tr-TR', { maximumFractionDigits: ondalik == null ? 0 : ondalik, minimumFractionDigits: ondalik == null ? 0 : ondalik });
  const sayi = (x, d) => x.toLocaleString('tr-TR', { maximumFractionDigits: d == null ? 2 : d, minimumFractionDigits: d == null ? 2 : d });
  const tok = (t) => (t.startsWith(' ') ? '␣' + t.slice(1) : t);
  const el = (etiket, ozel, ...cocuk) => {
    const e = document.createElement(etiket);
    if (ozel) for (const [a, v] of Object.entries(ozel)) {
      if (a === 'text') e.textContent = v;
      else if (a === 'class') e.className = v;
      else if (a.startsWith('on')) e.addEventListener(a.slice(2), v);
      else e.setAttribute(a, v);
    }
    for (const c of cocuk) if (c != null) e.append(c);
    return e;
  };
  const azalt = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

  window.Nester = { MODELLER, HAZIR, istem, oturum, anahtar, ac, ad, adK, rampa, css, isiCiz, yuzde, sayi, tok, el, rng, hash, azalt };
})();
