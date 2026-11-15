/* Sınıf 3: iskeletler. dom(kap, ctx) küçük ama çalışan bir Neşter ekranı kurar ve ciz() döndürür. */
(function () {
  const K = window.Katalog, N = window.Nester, el = N.el;
  const YUK = 330;
  function cerceve(kap, stil) { const f = el('div', { class: 'mini' }); f.style.cssText = 'height:' + YUK + 'px;position:relative;' + (stil || ''); kap.append(f); return f; }
  function panel(ad, stil) { const p = el('div', { class: 'p' }); if (ad) p.append(el('b', { text: ad })); if (stil) p.style.cssText = stil; return p; }
  const hepsi = (fs) => () => fs.forEach((f) => f());

  K.ekle({ id: 'isk-01', sinif: 'iskelet', ad: 'Kokpit', tarif: 'Her şey aynı anda ekranda, sabit panellerde. Kaydırma yok; göz panelden panele atlar.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;grid-template-columns:1.5fr 1fr 1fr;grid-template-rows:1fr 1fr;gap:6px');
      const a = panel('Isı haritası', 'grid-row:span 2'), b = panel('Sıcak noktalar'), c = panel('Ölçüm'), d = panel('Katman katman'), e = panel('Sepet');
      f.append(a, b, c, d, e);
      return hepsi([ctx.mini.nokta(a, YUK - 40), ctx.mini.sicak(b, 5), ctx.mini.olcum(c),
        ctx.mini.metin(d, () => { const S = ctx.S(), ln = S.lens([...ctx.secili()]); return [0, 6, 12, 18, S.L - 1].filter((l) => l < S.L).map((l) => 'K' + l + ' ' + N.tok(ln[l][0].tok)).join('  ·  '); }),
        ctx.mini.metin(e, () => ctx.secili().size ? [...ctx.secili()].map(N.adK).join(' + ') + ' (taslak)' : 'boş')]);
    } });

  K.ekle({ id: 'isk-02', sinif: 'iskelet', ad: 'Ray ve sekmeler', tarif: 'Solda dar bir ray ekranlar arasında geçer; içerik alanının üstünde sekmeler. Geniş içerik, az krom.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;grid-template-columns:46px 1fr;gap:6px');
      const ray = el('div', { style: 'display:grid;align-content:start;gap:10px;padding:6px 0;border-right:1px solid var(--cizgi);font:600 10px var(--f);color:var(--soluk);text-align:center' });
      const ekr = ['Keşif', 'Kavram', 'Kıyas', 'Kayıt']; let aktifE = 0, aktifS = 0;
      const ana = el('div', { style: 'display:grid;grid-template-rows:auto 1fr;gap:6px;min-width:0' });
      const sek = el('div', { class: 'sekme' }), govde = panel(''); govde.style.minHeight = '0';
      ana.append(sek, govde); f.append(ray, ana);
      let ic = () => {};
      function kur() {
        ray.replaceChildren(...ekr.map((e, i) => el('span', { text: e, style: 'cursor:pointer;padding:4px 0;' + (i === aktifE ? 'color:var(--murekkep);box-shadow:inset 3px 0 0 var(--vurgu)' : ''), onclick: () => { aktifE = i; kur(); } })));
        const sekmeler = aktifE === 0 ? ['2B', 'Sıcak', 'Ölçüm'] : aktifE === 1 ? ['Örnekler', 'Sonuç'] : aktifE === 2 ? ['İki cevap', 'Test seti'] : ['Biçim', 'Sonuç'];
        aktifS = Math.min(aktifS, sekmeler.length - 1);
        sek.replaceChildren(...sekmeler.map((s, i) => el('span', { class: i === aktifS ? 'on' : '', text: s, onclick: () => { aktifS = i; kur(); } })));
        govde.replaceChildren();
        if (aktifE === 0) ic = [() => ctx.mini.isi(govde, YUK - 60), () => ctx.mini.sicak(govde, 9), () => ctx.mini.olcum(govde)][aktifS]();
        else ic = ctx.mini.metin(govde, () => ekr[aktifE] + ' › ' + sekmeler[aktifS] + ': bu ekran bu iskelette ray üzerinden açılır. Seçim: ' + ([...ctx.secili()].map(N.adK).join(' + ') || 'yok'));
        ic();
      }
      kur();
      return () => ic();
    } });

  K.ekle({ id: 'isk-03', sinif: 'iskelet', ad: 'Tek akış', tarif: 'Yukarıdan aşağı tek sütun; her adım bir bölüm. Okur gibi ilerlersin, geri dönmek için yukarı kaydırırsın.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'overflow-y:auto;display:grid;gap:10px;align-content:start;padding-right:4px');
      const adim = (no, ad) => { const p = panel(no + ' · ' + ad); f.append(p); return p; };
      const a = adim('1', 'Model ve istem'), b = adim('2', 'Isı haritası'), c = adim('3', 'Sıcak noktalar'), d = adim('4', 'Ölçüm'), e = adim('5', 'Kaydet');
      return hepsi([ctx.mini.metin(a, () => ctx.S().model.ad + ' · “' + ctx.S().istem.metin + '”'), ctx.mini.isi(b, 160), ctx.mini.sicak(c, 4), ctx.mini.olcum(d),
        ctx.mini.metin(e, () => ctx.S().model.id + '-kesik.safetensors · ' + ctx.secili().size + ' kesim')]);
    } });

  K.ekle({ id: 'isk-04', sinif: 'iskelet', ad: 'Bölünmüş ekran', tarif: 'Ekranın iki yarısı: solda orijinal, sağda kesilmiş model. Her değişiklik iki tarafta birden görünür.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;grid-template-rows:110px 1fr;gap:6px');
      const ust = panel('Kesim seçimi'), alt = el('div', { style: 'display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid var(--cizgi);border-radius:6px;overflow:hidden' });
      const sol = panel('Orijinal', 'border:0;border-radius:0'), sag = panel('Kesilmiş', 'border:0;border-radius:0;border-left:3px solid var(--vurgu)');
      alt.append(sol, sag); f.append(ust, alt);
      const cevap = (kesik) => { const S = ctx.S(), r = S.olc(kesik ? [...ctx.secili()] : []); return S.istem.metin + (r.degisti ? S.istem.rakip : S.istem.hedef) + ' … (' + N.yuzde(r.sonra) + ')'; };
      return hepsi([ctx.mini.isi(ust, 70), ctx.mini.metin(sol, () => cevap(false)), ctx.mini.metin(sag, () => cevap(true))]);
    } });

  K.ekle({ id: 'isk-05', sinif: 'iskelet', ad: 'Komut paleti', tarif: 'Ekranın ortasında tek bir arama kutusu: “k19”, “ölç”, “kaydet” yaz, Enter. Geri kalan her şey arka planda.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;place-items:center');
      const arka = el('div', { style: 'position:absolute;inset:0;opacity:0.35;filter:blur(1.5px);pointer-events:none' });
      const pal = panel('', 'position:relative;width:min(92%,24rem);box-shadow:0 12px 30px rgba(0,0,0,.2);background:var(--kart);gap:6px');
      const g = el('input', { placeholder: 'head ya da komut yaz… (ör. k19, ölç)', 'aria-label': 'Komut' }), sonuc = el('ul', { class: 'liste' }), alt = el('div', { class: 'kucuk' });
      pal.append(g, sonuc, alt); f.append(arka, pal);
      const arkaCiz = ctx.mini.isi(arka, YUK);
      function ara() {
        const q = g.value.trim().toLocaleLowerCase('tr'), S = ctx.S();
        const komut = [['ölç', () => ''], ['sepete koy', () => ''], ['kaydet', () => '']].filter(([a]) => q && a.startsWith(q));
        const head = S.sicak.filter((s) => !q || N.ad(s.l, s.h).toLocaleLowerCase('tr').replace('·', '').includes(q.replace(/[·.\s]/g, ''))).slice(0, 5);
        sonuc.replaceChildren(...komut.map(([a]) => el('li', { onclick: () => { alt.textContent = '“' + a + '” çalıştı: ' + N.yuzde(S.olc([...ctx.secili()]).sonra); } }, el('span', { text: '› ' + a }), el('span', { text: 'komut' }))),
          ...head.map((s) => el('li', { class: ctx.secili().has(s.k) ? 'on' : '', onclick: () => ctx.degis(s.k) }, el('span', { text: N.ad(s.l, s.h) }), el('span', { text: N.sayi(s.v) }))));
        alt.textContent = alt.textContent || 'Enter ilk sonucu seçer';
      }
      g.addEventListener('input', () => { alt.textContent = ''; ara(); });
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter') { const li = sonuc.querySelector('li'); if (li) li.click(); } });
      return () => { arkaCiz(); ara(); };
    } });

  K.ekle({ id: 'isk-06', sinif: 'iskelet', ad: 'Sahne ve şerit', tarif: 'Video düzenleyicisi gibi: üstte büyük sahne, altta deneylerin dizildiği bir zaman şeridi. Şeritteki bir deneye tıkla, seçim geri gelir.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;grid-template-rows:1fr 64px;gap:6px');
      const sahne = panel('Sahne'), serit = el('div', { style: 'display:flex;gap:6px;overflow-x:auto;align-items:stretch;border-top:2px solid var(--cizgi);padding-top:6px' });
      f.append(sahne, serit);
      const deneyler = [];
      function seritCiz() {
        serit.replaceChildren(...deneyler.map((dn, i) => el('div', { class: 'p', style: 'min-width:7rem;cursor:pointer', onclick: () => ctx.ayarla(dn.k) },
          el('b', { text: 'D' + (i + 1) + ' · ' + N.yuzde(dn.r) }), el('span', { class: 'kucuk', text: dn.k.map(N.adK).join('+') }))),
          el('div', { class: 'p', style: 'min-width:5rem;cursor:pointer;place-items:center;color:var(--vurgu)', onclick: () => { if (!ctx.secili().size) return; deneyler.push({ k: [...ctx.secili()], r: ctx.S().olc([...ctx.secili()]).sonra }); seritCiz(); } }, el('b', { text: '+ ekle', style: 'color:var(--vurgu)' })));
      }
      seritCiz();
      return hepsi([ctx.mini.isi(sahne, YUK - 100), seritCiz]);
    } });

  K.ekle({ id: 'isk-07', sinif: 'iskelet', ad: 'Odak ve uydular', tarif: 'Merkezde tek büyük harita; dört köşede onu çevreleyen küçük uydu paneller harita üstünde yüzer.',
    dom(kap, ctx) {
      const f = cerceve(kap);
      const merkez = el('div', { style: 'position:absolute;inset:0' });
      const uydu = (stil, ad) => panel(ad, 'position:absolute;width:42%;max-width:11rem;background:color-mix(in srgb,var(--kart) 88%,transparent);backdrop-filter:blur(4px);box-shadow:0 4px 14px rgba(0,0,0,.15);' + stil);
      const a = uydu('top:8px;left:8px', 'Model'), b = uydu('top:8px;right:8px', 'Ölçüm'), c = uydu('bottom:8px;left:8px', 'Sıcak'), d = uydu('bottom:8px;right:8px', 'Lens');
      f.append(merkez, a, b, c, d);
      return hepsi([ctx.mini.isi(merkez, YUK), ctx.mini.metin(a, () => ctx.S().model.kisa + ' · ' + ctx.S().L + '×' + ctx.S().H), ctx.mini.olcum(b), ctx.mini.sicak(c, 3),
        ctx.mini.metin(d, () => { const S = ctx.S(), ln = S.lens([...ctx.secili()]); return 'son katman: ' + N.tok(ln[S.L - 1][0].tok); })]);
    } });

  K.ekle({ id: 'isk-08', sinif: 'iskelet', ad: 'Kart yığını', tarif: 'Ekranlar üst üste duran kartlar; sağdaki dikey sekmeler hangisini öne çekeceğini seçer. Arkadakiler kenarından görünür.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;grid-template-columns:1fr 34px;gap:4px');
      const yigin = el('div', { style: 'position:relative' }), sek = el('div', { style: 'display:grid;align-content:start;gap:4px' });
      f.append(yigin, sek);
      const ad = ['Isı', 'Sıcak', 'Ölçüm', 'Kayıt']; let on = 0; const kartlar = [], cizler = [];
      ad.forEach((a, i) => {
        const p = panel(a, 'position:absolute;inset:0;background:var(--kart);transition:transform .25s');
        kartlar.push(p); yigin.append(p);
        cizler.push(i === 0 ? ctx.mini.isi(p, YUK - 60) : i === 1 ? ctx.mini.sicak(p, 9) : i === 2 ? ctx.mini.olcum(p) : ctx.mini.metin(p, () => 'Ad: ' + ctx.S().model.id + '-kesik · biçim: safetensors · ' + ctx.secili().size + ' kesim'));
      });
      function diz() {
        kartlar.forEach((p, i) => { const s = (i - on + ad.length) % ad.length; p.style.transform = 'translate(' + s * 8 + 'px,' + s * 8 + 'px) scale(' + (1 - s * 0.04) + ')'; p.style.zIndex = String(10 - s); p.style.opacity = s ? '0.85' : '1'; });
        sek.replaceChildren(...ad.map((a, i) => el('span', { text: a, style: 'writing-mode:vertical-rl;cursor:pointer;padding:6px 4px;border-radius:4px;font:600 10px var(--f);' + (i === on ? 'background:var(--vurgu);color:var(--kart)' : 'background:color-mix(in srgb,var(--murekkep) 6%,transparent);color:var(--soluk)'), onclick: () => { on = i; diz(); cizler[i](); } })));
      }
      diz();
      return () => cizler[on]();
    } });

  K.ekle({ id: 'isk-09', sinif: 'iskelet', ad: 'Radyal menü', tarif: 'Ortada sonuç, çevresinde bir halka üzerinde ekranlar. Bir yöne tıkla, merkez o ekrana dönüşür.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'display:grid;place-items:center');
      const halka = el('div', { style: 'position:relative;width:min(100%,' + YUK + 'px);aspect-ratio:1;max-height:100%' });
      const merkez = panel('', 'position:absolute;left:24%;top:24%;width:52%;height:52%;border-radius:50%;place-content:center;text-align:center;padding:14px');
      halka.append(el('div', { style: 'position:absolute;inset:8%;border:1px dashed var(--cizgi);border-radius:50%' }), merkez); f.append(halka);
      const ad = ['Ölçüm', 'Isı', 'Sıcak', 'Lens', 'Kavram', 'Kaydet']; let on = 0, ic = () => {};
      function kur() {
        halka.querySelectorAll('.yon').forEach((x) => x.remove());
        ad.forEach((a, i) => { const t = -Math.PI / 2 + i / ad.length * Math.PI * 2; halka.append(el('span', { class: 'yon', text: a, onclick: () => { on = i; kur(); },
          style: 'position:absolute;left:' + (50 + Math.cos(t) * 42) + '%;top:' + (50 + Math.sin(t) * 42) + '%;transform:translate(-50%,-50%);cursor:pointer;font:600 11px var(--f);padding:3px 8px;border-radius:999px;background:' + (i === on ? 'var(--vurgu);color:var(--kart)' : 'var(--kart);color:var(--soluk);border:1px solid var(--cizgi)') })); });
        merkez.replaceChildren();
        ic = on === 0 ? ctx.mini.olcum(merkez) : on === 1 ? ctx.mini.isi(merkez, 90) : on === 2 ? ctx.mini.sicak(merkez, 4) : ctx.mini.metin(merkez, () => ad[on] + ' ekranı · seçim ' + (ctx.secili().size || 'yok'));
        ic();
      }
      kur();
      return () => ic();
    } });

  K.ekle({ id: 'isk-10', sinif: 'iskelet', ad: 'Çekmeceler', tarif: 'Ana alan haritaya ait; liste alttan, ölçüm sağdan çekmece gibi açılır. İşin bitince çekmeceleri kapat, alan geri gelir.',
    dom(kap, ctx) {
      const f = cerceve(kap, 'overflow:hidden');
      const ana = el('div', { style: 'position:absolute;inset:0 0 26px 0' });
      const alt = panel('Sıcak noktalar ▴', 'position:absolute;left:0;right:0;bottom:0;height:150px;transform:translateY(124px);transition:transform .25s;background:var(--kart);z-index:2;cursor:pointer');
      const sag = panel('Ölçüm ◂', 'position:absolute;top:0;right:0;bottom:30px;width:52%;transform:translateX(calc(100% - 26px));transition:transform .25s;background:var(--kart);z-index:3;cursor:pointer');
      f.append(ana, alt, sag);
      let altA = false, sagA = false;
      alt.firstChild.addEventListener('click', () => { altA = !altA; alt.style.transform = altA ? 'none' : 'translateY(124px)'; alt.firstChild.textContent = 'Sıcak noktalar ' + (altA ? '▾' : '▴'); });
      sag.firstChild.addEventListener('click', () => { sagA = !sagA; sag.style.transform = sagA ? 'none' : 'translateX(calc(100% - 26px))'; sag.firstChild.textContent = 'Ölçüm ' + (sagA ? '▸' : '◂'); });
      return hepsi([ctx.mini.isi(ana, YUK - 30), ctx.mini.sicak(alt, 5), ctx.mini.olcum(sag)]);
    } });
})();
