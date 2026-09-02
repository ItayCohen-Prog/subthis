/* subthis site behaviors: caption player, OS detection, tabs, copy, reveal, typing demo. */
(function () {
  "use strict";
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- hero caption player: cycles cues with an SRT clock ---- */
  var cueEl = document.querySelector("[data-cues]");
  if (cueEl) {
    var cues;
    try { cues = JSON.parse(cueEl.getAttribute("data-cues")); } catch (e) { cues = []; }
    var tcEl = document.querySelector(".timecode");
    var index = 0;
    var startedAt = Date.now();
    function srt(ms) {
      var s = Math.floor(ms / 1000), m = Math.floor(s / 60), h = Math.floor(m / 60);
      function pad(n, w) { return String(n).padStart(w, "0"); }
      return pad(h, 2) + ":" + pad(m % 60, 2) + ":" + pad(s % 60, 2) + "," + pad(ms % 1000, 3);
    }
    function show(i) { cueEl.innerHTML = cues[i % cues.length]; }
    if (cues.length) show(0);
    if (!reduced && cues.length > 1) {
      setInterval(function () {
        index += 1;
        cueEl.classList.add("swap");
        setTimeout(function () { show(index); cueEl.classList.remove("swap"); }, 180);
      }, 1900);
      if (tcEl) setInterval(function () { tcEl.textContent = srt(Date.now() - startedAt); }, 90);
    }
  }

  /* ---- OS detection for the install section ---- */
  function detectOS() {
    var ua = navigator.userAgent || "";
    var plat = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
    var hay = (plat + " " + ua).toLowerCase();
    if (hay.indexOf("win") !== -1) return "windows";
    if (hay.indexOf("mac") !== -1 && hay.indexOf("iphone") === -1 && hay.indexOf("ipad") === -1) return "macos";
    if (hay.indexOf("linux") !== -1 || hay.indexOf("x11") !== -1) return "linux";
    return "linux";
  }
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tabs [data-os]"));
  if (tabs.length) {
    var panes = Array.prototype.slice.call(document.querySelectorAll(".os-pane"));
    function select(os) {
      tabs.forEach(function (b) { b.setAttribute("aria-selected", String(b.getAttribute("data-os") === os)); });
      panes.forEach(function (p) { p.hidden = p.getAttribute("data-os") !== os; });
    }
    tabs.forEach(function (b) {
      b.addEventListener("click", function () { select(b.getAttribute("data-os")); });
    });
    var os = detectOS();
    select(os);
    var noteName = document.querySelector("[data-os-name]");
    if (noteName) {
      var names = JSON.parse(noteName.getAttribute("data-os-name"));
      noteName.textContent = names[os] || os;
    }
  }

  /* ---- copy buttons ---- */
  Array.prototype.slice.call(document.querySelectorAll(".copy-btn")).forEach(function (btn) {
    btn.addEventListener("click", function () {
      var pre = btn.parentElement.querySelector("pre, code");
      if (!pre) return;
      var text = pre.innerText.replace(/^\$ /gm, "");
      navigator.clipboard.writeText(text).then(function () {
        var was = btn.textContent;
        btn.textContent = btn.getAttribute("data-done") || "copied";
        btn.classList.add("done");
        setTimeout(function () { btn.textContent = was; btn.classList.remove("done"); }, 1400);
      });
    });
  });

  /* ---- scroll reveal ---- */
  var revealed = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
  if ("IntersectionObserver" in window && !reduced) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) { entry.target.classList.add("in"); io.unobserve(entry.target); }
      });
    }, { threshold: 0.12 });
    revealed.forEach(function (el) { io.observe(el); });
  } else {
    revealed.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---- typing terminal demo ---- */
  var demo = document.querySelector("[data-demo]");
  if (demo) {
    var lines = JSON.parse(demo.getAttribute("data-demo"));
    var pre = demo.querySelector("pre");
    if (reduced) {
      pre.innerHTML = lines.map(function (l) { return l.html; }).join("\n");
    } else {
      var out = [];
      var li = 0;
      function nextLine() {
        if (li >= lines.length) { li = 0; out = []; setTimeout(step, 4000); return; }
        var line = lines[li];
        if (line.type === "cmd") {
          var typed = 0;
          var plain = line.text;
          var timer = setInterval(function () {
            typed += 1;
            pre.innerHTML = out.join("\n") + (out.length ? "\n" : "") +
              '<span class="prompt">$</span> ' + plain.slice(0, typed) + '<span class="cursor"></span>';
            if (typed >= plain.length) {
              clearInterval(timer);
              out.push('<span class="prompt">$</span> ' + plain);
              li += 1;
              setTimeout(nextLine, 350);
            }
          }, 45);
        } else {
          out.push(line.html);
          pre.innerHTML = out.join("\n") + '<span class="cursor"></span>';
          li += 1;
          setTimeout(nextLine, line.wait || 300);
        }
      }
      function step() { pre.innerHTML = ""; nextLine(); }
      if ("IntersectionObserver" in window) {
        var seen = false;
        new IntersectionObserver(function (entries, obs) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting && !seen) { seen = true; step(); obs.disconnect(); }
          });
        }, { threshold: 0.3 }).observe(demo);
      } else { step(); }
    }
  }

  /* ---- docs: highlight current section in TOC ---- */
  var toc = document.querySelector(".toc");
  if (toc && "IntersectionObserver" in window) {
    var links = Array.prototype.slice.call(toc.querySelectorAll("a[href^='#']"));
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var sectionObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          links.forEach(function (a) { a.classList.remove("here"); });
          var a = map[entry.target.id];
          if (a) a.classList.add("here");
        }
      });
    }, { rootMargin: "-15% 0px -75% 0px" });
    Array.prototype.slice.call(document.querySelectorAll(".doc-body section[id]"))
      .forEach(function (s) { sectionObserver.observe(s); });
  }
})();
