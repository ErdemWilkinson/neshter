/* html-isaret: bir öğeye tıkla, ona bağlı not bırak. Bağımlılıksız; Shadow DOM içinde çalışır.
   Depo sırası: artifact db ("isaretler") → yerel alıcı (127.0.0.1:47824, gerekirse köprü penceresi) → tarayıcı (localStorage).
   İşaret modu kapalıyken sayfaya yalnız kendi düğmesini ekler. */
(function () {
  'use strict';
  var G = typeof globalThis !== 'undefined' ? globalThis : window;

  // ── Saf yardımcılar (testlerde de kullanılır)
  function kacis(s) {
    return G.CSS && G.CSS.escape ? G.CSS.escape(s) : String(s).replace(/[^A-Za-z0-9_-]/g, function (c) { return '\\' + c; });
  }
  function seciciUret(el) {
    if (!el || !el.tagName) return '';
    var parca = [];
    var n = el;
    while (n && n.tagName && n.tagName.toLowerCase() !== 'html') {
      var tag = n.tagName.toLowerCase();
      if (n.id && /^[A-Za-z][\w-]*$/.test(n.id)) { parca.unshift('#' + kacis(n.id)); return parca.join(' > '); }
      if (tag === 'body') { parca.unshift('body'); return parca.join(' > '); }
      var ebeveyn = n.parentElement;
      var k = 1;
      if (ebeveyn && ebeveyn.children) {
        var ç = ebeveyn.children;
        for (var i = 0; i < ç.length && ç[i] !== n; i++) if (ç[i].tagName === n.tagName) k++;
      }
      parca.unshift(tag + ':nth-of-type(' + k + ')');
      n = ebeveyn;
    }
    return parca.join(' > ');
  }
  function siteAdi(loc) {
    var h = (loc && loc.host) || '';
    if (!h) return 'dosya';
    return h.replace(/:/g, '_').replace(/[^A-Za-z0-9._-]/g, '-').slice(0, 120);
  }
  function etiketUret(el) {
    var t = el.tagName ? el.tagName.toLowerCase() : '?';
    if (el.id) return t + '#' + el.id;
    var c = el.classList && el.classList.length ? el.classList[0] : '';
    return c ? t + '.' + c : t;
  }
  function notYap(el, metin, loc, simdi) {
    var z = simdi || new Date();
    var ozet = String(el.textContent || '').replace(/\s+/g, ' ').trim();
    if (ozet.length > 80) ozet = ozet.slice(0, 79) + '…';
    var r = el.getBoundingClientRect ? el.getBoundingClientRect() : { left: 0, top: 0, width: 0, height: 0 };
    var varyant = el.closest ? el.closest('[data-varyant]') : null;
    return {
      id: 'n' + z.getTime().toString(36) + Math.random().toString(36).slice(2, 6),
      site: siteAdi(loc), sayfa: (loc && (loc.pathname || '/')) + ((loc && loc.hash) || ''),
      secici: seciciUret(el), etiket: etiketUret(el), metin_ozeti: ozet, not: String(metin),
      varyant: varyant ? varyant.getAttribute('data-varyant') : '',
      durum: 'acik', zaman: z.toISOString(),
      kutu: { x: Math.round(r.left), y: Math.round(r.top + ((G.scrollY) || 0)), w: Math.round(r.width), h: Math.round(r.height) },
    };
  }
  G.IsaretYardim = { seciciUret: seciciUret, siteAdi: siteAdi, notYap: notYap, etiketUret: etiketUret };
  if (typeof document === 'undefined') return;
  if (G.Isaret) return;

  // ── Depolar
  function tarayiciDepo(site) {
    var anahtar = 'isaret:' + site, dinleyen = null;
    function oku() { try { return JSON.parse(localStorage.getItem(anahtar) || '[]'); } catch (e) { return G.__isaretBellek || []; } }
    return {
      ad: 'tarayıcı (yalnız bu tarayıcıda)',
      liste: function () { return Promise.resolve(oku()); },
      kaydet: function (n) {
        var l = oku().filter(function (x) { return x.id !== n.id; }).concat([n]);
        try { localStorage.setItem(anahtar, JSON.stringify(l)); } catch (e) { G.__isaretBellek = l; }
        if (dinleyen) dinleyen(l);
        return Promise.resolve();
      },
      sil: function (id) {
        var l = oku().filter(function (x) { return x.id !== id; });
        try { localStorage.setItem(anahtar, JSON.stringify(l)); } catch (e) { G.__isaretBellek = l; }
        if (dinleyen) dinleyen(l);
        return Promise.resolve();
      },
      izle: function (fn) { dinleyen = fn; return function () { dinleyen = null; }; },
    };
  }
  function artifactDepo(db) {
    var kol = db.collection('isaretler');
    return {
      ad: 'artifact',
      liste: function () { return kol.get().then(function (s) { return s.docs.map(function (d) { return d.data(); }); }); },
      kaydet: function (n) { return kol.doc(n.id).set(n); },
      sil: function (id) { return kol.doc(id).delete(); },
      izle: function (fn, hata) { return kol.onSnapshot(function (s) { fn(s.docs.map(function (d) { return d.data(); })); }, hata); },
    };
  }
  function yerelDepo(alici, site) {
    return {
      ad: 'yerel alıcı',
      liste: function () { return fetch(alici + '/notlar?site=' + encodeURIComponent(site)).then(function (r) { return r.json(); }); },
      kaydet: function (n) {
        return fetch(alici + '/not', { method: 'POST', headers: { 'Content-Type': 'application/json; charset=utf-8' }, body: JSON.stringify(n) })
          .then(function (r) { if (!r.ok) throw new Error('alıcı ' + r.status); });
      },
      sil: function (id) { return Promise.reject(new Error('Yerel alıcıda silme yok; notu çözüldü olarak bırak.')); },
    };
  }
  function kopruDepo(alici, site) {
    var pen = null, no = 0, bekleyen = {};
    addEventListener('message', function (e) {
      var m = e.data;
      if (!m || m.isaret !== true || m.no == null || !bekleyen[m.no]) return;
      var b = bekleyen[m.no]; delete bekleyen[m.no];
      m.ok ? b.ok(m) : b.hata(new Error(m.hata || 'köprü hatası'));
    });
    function gonder(ileti) {
      if (!pen || pen.closed) pen = open(alici + '/kutu', 'isaret-kopru', 'width=360,height=160');
      if (!pen) return Promise.reject(new Error('Köprü penceresi açılamadı (açılır pencere engelli olabilir).'));
      ileti.isaret = true; ileti.no = ++no;
      return new Promise(function (ok, hata) {
        bekleyen[ileti.no] = { ok: ok, hata: hata };
        var dene = 0;
        (function at() {
          try { pen.postMessage(ileti, alici); } catch (e) { /* henüz yüklenmedi */ }
          if (bekleyen[ileti.no] && ++dene < 20) setTimeout(at, 250);
          else if (bekleyen[ileti.no]) { delete bekleyen[ileti.no]; hata(new Error('Köprü cevap vermedi; alıcı çalışıyor mu?')); }
        })();
      });
    }
    return {
      ad: 'yerel alıcı (köprü penceresi)',
      liste: function () { return gonder({ tur: 'liste', site: site }).then(function (m) { return m.notlar || []; }); },
      kaydet: function (n) { return gonder({ tur: 'kaydet', not: n }); },
      sil: function () { return Promise.reject(new Error('Yerel alıcıda silme yok.')); },
    };
  }
  function zamanAsimi(p, ms) { return Promise.race([p, new Promise(function (ok) { setTimeout(function () { ok(null); }, ms); })]); }
  function depoSec(ayar, site) {
    var alici = ayar.alici || 'http://127.0.0.1:47824';
    if (ayar.depo === 'tarayici') return Promise.resolve(tarayiciDepo(site));
    if (ayar.depo === 'yerel') {
      return fetch(alici + '/notlar?site=' + encodeURIComponent(site)).then(function (r) { if (!r.ok) throw 0; return yerelDepo(alici, site); })
        .catch(function () { return kopruDepo(alici, site); });
    }
    var cl = G.claude;
    var dbSoz = cl && typeof cl.use === 'function' ? zamanAsimi(Promise.resolve(cl.use('db')).catch(function () { return null; }), 6000) : Promise.resolve(null);
    return dbSoz.then(function (db) {
      if (db) return artifactDepo(db);
      if (ayar.depo === 'artifact') return tarayiciDepo(site);
      return zamanAsimi(fetch(alici + '/notlar?site=' + encodeURIComponent(site)).then(function (r) { return r.ok ? yerelDepo(alici, site) : null; }).catch(function () { return null; }), 1500)
        .then(function (d) { return d || tarayiciDepo(site); });
    });
  }

  // ── Arayüz
  var CSS = [
    ':host{all:initial}',
    '*{box-sizing:border-box;font-family:system-ui,"Segoe UI",sans-serif}',
    '.dugmeler{position:fixed;left:max(14px,env(safe-area-inset-left,0px));bottom:max(14px,env(safe-area-inset-bottom,0px));z-index:2147483646;display:flex;gap:6px}',
    '.d{font:600 13px/1 system-ui,sans-serif;padding:9px 12px;border-radius:999px;border:1px solid #c9ccd2;background:#fff;color:#1b1d22;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.15)}',
    '.d:focus-visible{outline:2px solid #e8590c;outline-offset:2px}',
    '.d.on{background:#e8590c;border-color:#e8590c;color:#fff}',
    '.cerceve{position:fixed;pointer-events:none;border:2px solid #e8590c;background:rgba(232,89,12,.08);border-radius:3px;z-index:2147483645;display:none}',
    '.cerceve span{position:absolute;left:-2px;top:-20px;background:#e8590c;color:#fff;font:600 11px/1 ui-monospace,Consolas,monospace;padding:3px 5px;border-radius:3px 3px 0 0;white-space:nowrap}',
    '.kutu{position:fixed;z-index:2147483647;width:min(320px,calc(100vw - 24px));background:#fff;color:#1b1d22;border:1px solid #c9ccd2;border-radius:10px;padding:12px;box-shadow:0 10px 30px rgba(0,0,0,.25);display:grid;gap:8px}',
    '.kutu small{color:#5d6370;font:11px ui-monospace,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
    '.kutu textarea{width:100%;min-height:80px;resize:vertical;font:14px/1.4 system-ui,sans-serif;padding:8px;border:1px solid #c9ccd2;border-radius:6px;color:#1b1d22;background:#fff}',
    '.kutu textarea:focus{outline:2px solid #e8590c;outline-offset:0}',
    '.kutu .sat{display:flex;gap:8px;justify-content:flex-end}',
    '.kutu button{font:600 13px system-ui,sans-serif;padding:7px 12px;border-radius:6px;border:1px solid #c9ccd2;background:#fff;color:#1b1d22;cursor:pointer}',
    '.kutu button.ana{background:#e8590c;border-color:#e8590c;color:#fff}',
    '.rozet{position:fixed;z-index:2147483644;min-width:20px;height:20px;padding:0 5px;border-radius:10px;background:#e8590c;color:#fff;font:700 11px/20px system-ui,sans-serif;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.3);cursor:pointer}',
    '.rozet.cozuldu{background:#2f9e44}',
    '.liste{position:fixed;left:max(14px,env(safe-area-inset-left,0px));bottom:62px;z-index:2147483646;width:min(360px,calc(100vw - 28px));max-height:min(60vh,520px);overflow:auto;background:#fff;color:#1b1d22;border:1px solid #c9ccd2;border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.25);padding:10px 12px;display:grid;gap:8px;align-content:start}',
    '.liste h3{margin:0;font:700 14px system-ui,sans-serif;display:flex;justify-content:space-between;gap:8px}',
    '.liste h3 span{font:400 11px system-ui,sans-serif;color:#5d6370}',
    '.liste ol{margin:0;padding:0;list-style:none;display:grid;gap:6px}',
    '.liste li{display:grid;grid-template-columns:22px 1fr;gap:8px;font:13px/1.4 system-ui,sans-serif;padding:6px;border-radius:6px;cursor:pointer}',
    '.liste li:hover{background:#f2f3f5}',
    '.liste li b{display:inline-block;min-width:20px;height:20px;border-radius:10px;background:#e8590c;color:#fff;font:700 11px/20px system-ui;text-align:center}',
    '.liste li.cozuldu b{background:#2f9e44}',
    '.liste li.cozuldu .m{text-decoration:line-through;color:#5d6370}',
    '.liste li small{display:block;color:#5d6370;font:11px ui-monospace,Consolas,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}',
    '.liste li .yok{color:#c92a2a}',
    '.liste p{margin:0;color:#5d6370;font:13px system-ui,sans-serif}',
    '.ipucu{position:fixed;left:50%;top:max(12px,env(safe-area-inset-top,0px));transform:translateX(-50%);z-index:2147483646;background:#1b1d22;color:#fff;font:600 13px system-ui,sans-serif;padding:8px 14px;border-radius:999px;display:none}',
    '@media (prefers-color-scheme: dark){.d,.kutu,.liste,.kutu textarea,.kutu button{background:#1f2227;color:#eceef1;border-color:#3a3f47}.liste li:hover{background:#2a2e35}.kutu small,.liste li small,.liste p,.liste h3 span{color:#a2a8b3}.kutu button.ana,.d.on{background:#e8590c;color:#fff;border-color:#e8590c}}',
  ].join('\n');

  function baslat(ayar) {
    ayar = ayar || G.ISARET_AYAR || {};
    if (G.Isaret && G.Isaret.calisiyor) return G.Isaret;
    var site = siteAdi(location);
    var host = document.createElement('isaret-katman');
    host.setAttribute('data-isaret', '');
    var kok = host.attachShadow({ mode: 'open' });
    kok.innerHTML = '<style>' + CSS + '</style>' +
      '<div class="dugmeler"><button class="d isaretle" type="button" aria-pressed="false">✎ İşaretle</button><button class="d notlar" type="button" aria-expanded="false">Notlar (0)</button></div>' +
      '<div class="cerceve"><span></span></div><div class="ipucu">Not bırakmak istediğin yere tıkla · Esc çıkar</div><div class="rozetler"></div>';
    document.documentElement.appendChild(host);
    var $ = function (s) { return kok.querySelector(s); };
    var bIsaret = $('.isaretle'), bNotlar = $('.notlar'), cerceve = $('.cerceve'), ipucu = $('.ipucu'), rozetler = $('.rozetler');
    var depo = null, notlar = [], mod = false, hedef = null, kutu = null, liste = null, depoHata = '';

    function bizden(e) { var p = e.composedPath ? e.composedPath() : []; return p.indexOf(host) >= 0; }
    function cerceveGoster(el) {
      if (!el) { cerceve.style.display = 'none'; return; }
      var r = el.getBoundingClientRect();
      cerceve.style.display = 'block';
      cerceve.style.left = r.left - 2 + 'px'; cerceve.style.top = r.top - 2 + 'px';
      cerceve.style.width = r.width + 4 + 'px'; cerceve.style.height = r.height + 4 + 'px';
      cerceve.firstChild.textContent = etiketUret(el);
    }
    function uzerinde(e) { if (bizden(e)) return; hedef = e.target; cerceveGoster(hedef); }
    function tikla(e) {
      if (bizden(e)) return;
      e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
      if (e.type !== 'click') return;
      kutuAc(e.target, e.clientX, e.clientY);
    }
    function tus(e) { if (e.key === 'Escape') { if (kutu) kutuKapat(); else modDegis(false); } }
    var OLAYLAR = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'dblclick'];
    function modDegis(ac) {
      mod = ac;
      bIsaret.classList.toggle('on', ac); bIsaret.setAttribute('aria-pressed', String(ac));
      ipucu.style.display = ac ? 'block' : 'none';
      if (ac) {
        document.addEventListener('pointermove', uzerinde, true);
        OLAYLAR.forEach(function (t) { document.addEventListener(t, tikla, true); });
        document.addEventListener('keydown', tus, true);
      } else {
        document.removeEventListener('pointermove', uzerinde, true);
        OLAYLAR.forEach(function (t) { document.removeEventListener(t, tikla, true); });
        document.removeEventListener('keydown', tus, true);
        cerceveGoster(null); kutuKapat();
      }
    }
    function kutuAc(el, x, y) {
      kutuKapat();
      cerceveGoster(el);
      kutu = document.createElement('div'); kutu.className = 'kutu';
      kutu.innerHTML = '<small></small><textarea placeholder="Burada neyi beğenmedin, ne olsun?" aria-label="Not"></textarea><div class="sat"><button type="button" class="vaz">Vazgeç</button><button type="button" class="ana">Kaydet</button></div>';
      kutu.querySelector('small').textContent = etiketUret(el) + (el.textContent ? ' · ' + String(el.textContent).replace(/\s+/g, ' ').trim().slice(0, 40) : '');
      kok.appendChild(kutu);
      var w = kutu.offsetWidth, h = kutu.offsetHeight;
      kutu.style.left = Math.max(12, Math.min(innerWidth - w - 12, x + 12)) + 'px';
      kutu.style.top = Math.max(12, Math.min(innerHeight - h - 12, y + 12)) + 'px';
      var ta = kutu.querySelector('textarea');
      ta.focus();
      ta.addEventListener('keydown', function (e) { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) kaydet(); e.stopPropagation(); });
      kutu.querySelector('.vaz').onclick = kutuKapat;
      kutu.querySelector('.ana').onclick = kaydet;
      function kaydet() {
        var m = ta.value.trim();
        if (!m) { ta.focus(); return; }
        var n = notYap(el, m, location);
        notlar = notlar.filter(function (q) { return q.id !== n.id; }).concat([n]);
        ciz();
        kutuKapat(); modDegis(false);
        (depo ? depo.kaydet(n) : Promise.reject(new Error('depo hazır değil'))).catch(function (err) {
          depoHata = 'Kaydedilemedi: ' + err.message + '. Not bu tarayıcıda saklandı.';
          depo = tarayiciDepo(site); depo.kaydet(n); listeCiz();
        });
      }
    }
    function kutuKapat() { if (kutu) { kutu.remove(); kutu = null; } if (!mod) cerceveGoster(null); }
    function bul(n) { try { return n.secici ? document.querySelector(n.secici) : null; } catch (e) { return null; } }
    function sirali() { return notlar.slice().sort(function (a, b) { return a.zaman < b.zaman ? -1 : 1; }); }
    function rozetCiz() {
      var s = sirali(), parca = [];
      rozetler.textContent = '';
      s.forEach(function (n, i) {
        if (n.sayfa && n.sayfa.split('#')[0] !== location.pathname) return;
        var el = bul(n); if (!el) return;
        var r = el.getBoundingClientRect();
        if (r.bottom < 0 || r.top > innerHeight || (r.width === 0 && r.height === 0)) return;
        var b = document.createElement('div');
        b.className = 'rozet' + (n.durum === 'cozuldu' ? ' cozuldu' : '');
        b.textContent = n.durum === 'cozuldu' ? '✓' : String(i + 1);
        b.title = n.not;
        b.style.left = Math.min(innerWidth - 26, Math.max(4, r.right - 10)) + 'px';
        b.style.top = Math.max(4, r.top - 10) + 'px';
        b.onclick = function () { listeAc(true); };
        parca.push(b);
      });
      parca.forEach(function (b) { rozetler.appendChild(b); });
    }
    function listeCiz() {
      if (!liste) return;
      var s = sirali();
      liste.innerHTML = '<h3>Notlar <span></span></h3>';
      liste.querySelector('h3 span').textContent = (depo ? depo.ad : 'bağlanıyor…');
      if (depoHata) { var ph = document.createElement('p'); ph.textContent = depoHata; liste.appendChild(ph); }
      if (!s.length) { var p = document.createElement('p'); p.textContent = 'Henüz not yok. “✎ İşaretle”ye bas, sonra sayfada beğenmediğin yere tıkla.'; liste.appendChild(p); return; }
      var ol = document.createElement('ol');
      s.forEach(function (n, i) {
        var li = document.createElement('li'); li.className = n.durum === 'cozuldu' ? 'cozuldu' : '';
        var b = document.createElement('b'); b.textContent = n.durum === 'cozuldu' ? '✓' : String(i + 1);
        var d = document.createElement('div');
        var m = document.createElement('div'); m.className = 'm'; m.textContent = n.not;
        var sm = document.createElement('small'); sm.textContent = (n.varyant ? n.varyant + ' · ' : '') + n.etiket;
        d.appendChild(m); d.appendChild(sm);
        if (!bul(n)) { var y = document.createElement('small'); y.className = 'yok'; y.textContent = 'öğe bulunamadı'; d.appendChild(y); }
        li.appendChild(b); li.appendChild(d);
        li.onclick = function () { var el = bul(n); if (el) { el.scrollIntoView({ block: 'center', behavior: 'smooth' }); setTimeout(function () { cerceveGoster(el); setTimeout(function () { if (!mod) cerceveGoster(null); }, 1200); }, 350); } };
        ol.appendChild(li);
      });
      liste.appendChild(ol);
    }
    function listeAc(ac) {
      if (ac == null) ac = !liste;
      if (ac && !liste) { liste = document.createElement('div'); liste.className = 'liste'; kok.appendChild(liste); listeCiz(); }
      if (!ac && liste) { liste.remove(); liste = null; }
      bNotlar.setAttribute('aria-expanded', String(!!liste));
    }
    function ciz() {
      var acik = notlar.filter(function (n) { return n.durum !== 'cozuldu'; }).length;
      bNotlar.textContent = 'Notlar (' + acik + (notlar.length !== acik ? ' · ✓' + (notlar.length - acik) : '') + ')';
      rozetCiz(); listeCiz();
    }
    var bekle = 0;
    function yeniden() { if (bekle) return; bekle = requestAnimationFrame(function () { bekle = 0; rozetCiz(); }); }
    addEventListener('scroll', yeniden, true); addEventListener('resize', yeniden);
    setInterval(function () { if (notlar.length && !document.hidden) rozetCiz(); }, 700);

    bIsaret.onclick = function () { modDegis(!mod); };
    bNotlar.onclick = function () { listeAc(); };

    depoSec(ayar, site).then(function (d) {
      depo = d;
      if (d.izle) d.izle(function (l) { notlar = l || []; ciz(); }, function (e) { depoHata = 'Notlar okunamıyor: ' + (e && e.code || e); listeCiz(); });
      else d.liste().then(function (l) { notlar = l || []; ciz(); }).catch(function (e) { depoHata = String(e.message || e); listeCiz(); });
      listeCiz();
    });

    G.Isaret = { calisiyor: true, depo: function () { return depo; }, notlar: function () { return notlar.slice(); }, mod: modDegis };
    return G.Isaret;
  }

  G.Isaret = { baslat: baslat };
  if (G.ISARET_ELLE !== true) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { baslat(); });
    else baslat();
  }
})();
