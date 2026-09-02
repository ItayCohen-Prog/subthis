/* subthis site behaviors: OS detection, tabs, copy, reveal, demo scene. */
(function () {
  "use strict";
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
        var was = btn.title;
        btn.title = btn.getAttribute("data-done") || "copied";
        btn.classList.add("done");
        setTimeout(function () { btn.title = was; btn.classList.remove("done"); }, 1400);
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

  /* ---- demo scene: drag the video into the terminal, get an .srt, drag it into the editor ---- */
  var scene = document.querySelector("[data-scene]");
  if (scene) {
    var pre = scene.querySelector("[data-term] pre");
    var fileMp4 = scene.querySelector('[data-file="mp4"]');
    var fileSrt = scene.querySelector('[data-file="srt"]');
    var subs = scene.querySelector("[data-subs]");
    var caption = scene.querySelector("[data-caption]");
    var ghost = scene.querySelector("[data-ghost]");
    var pointer = scene.querySelector("[data-pointer]");
    var PATH = "~/Videos/lecture.mp4";
    var OUT = [
      ["Transcribing chunk 1/1...", 1400],
      ["", 150],
      ['<span class="ok">  ✓ </span>Done! 214 subtitles were created.', 350],
      ["  Your subtitle file is here:", 250],
      ['    <span class="yel">~/Videos/lecture.srt</span>', 500]
    ];
    var CUES = [[1,5],[8,4],[14,6],[22,3],[27,5],[34,4],[40,6],[48,4],[54,5],[61,3],[66,6],[74,4],[80,5],[87,3],[92,5]];
    var CUR = '<span class="cursor"></span>';
    function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
    function line(cmd) { return '<span class="prompt">$</span> ' + cmd; }
    function fullTerm() { return line("subthis " + PATH) + "\n" + OUT.map(function (o) { return o[0]; }).join("\n") + CUR; }
    function buildCues() {
      subs.querySelectorAll(".ed-cue").forEach(function (e) { e.remove(); });
      return CUES.map(function (c) {
        var el = document.createElement("i"); el.className = "ed-cue";
        el.style.left = "calc(34px + (100% - 40px) * " + (c[0] / 100) + ")";
        el.style.width = "calc((100% - 40px) * " + (c[1] / 100) + ")";
        subs.appendChild(el); return el;
      });
    }
    function reserve() {
      var probe = pre.cloneNode(false);
      probe.innerHTML = fullTerm();
      probe.style.position = "absolute"; probe.style.visibility = "hidden"; probe.style.minHeight = "0";
      pre.parentNode.appendChild(probe);
      pre.style.minHeight = probe.offsetHeight + "px";
      pre.parentNode.removeChild(probe);
    }
    reserve();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(reserve);

    function finalState() {
      pre.innerHTML = fullTerm();
      fileSrt.classList.add("on");
      buildCues().forEach(function (c) { c.classList.add("on"); });
      caption.classList.add("on");
    }

    if (reduced) {
      finalState();
    } else {
      /* positions are read live from the layout, so the same script works
         for the 3-column, 2-column and stacked arrangements */
      function at(el, fx, fy) {
        var s = scene.getBoundingClientRect(), r = el.getBoundingClientRect();
        return [r.left - s.left + r.width * fx, r.top - s.top + r.height * fy];
      }
      function place(el, xy, ms) {
        el.style.transitionDuration = (ms || 0) + "ms, 150ms";
        el.style.transform = "translate(" + Math.round(xy[0]) + "px," + Math.round(xy[1]) + "px)";
      }
      function pointerTo(xy, ms, withGhost) {
        place(pointer, [xy[0] - 4, xy[1] - 3], ms);
        if (withGhost) place(ghost, [xy[0] + 12, xy[1] + 10], ms);
        return sleep(ms);
      }
      function typeInto(prefix, text, done) {
        return new Promise(function (resolve) {
          var i = 0;
          var t = setInterval(function () {
            i += 1;
            pre.innerHTML = prefix + text.slice(0, i) + CUR;
            if (i >= text.length) { clearInterval(t); resolve(); }
          }, 45);
        });
      }
      async function run() {
        pre.innerHTML = line("") + CUR;
        fileSrt.classList.remove("on"); fileMp4.classList.remove("grab");
        caption.classList.remove("on"); subs.classList.remove("over");
        var cues = buildCues();
        ghost.classList.remove("on");
        pointerTo(at(fileMp4, 0.5, 1.5), 0); pointer.classList.add("on");
        await sleep(600);
        await typeInto(line(""), "subthis ");
        await pointerTo(at(fileMp4, 0.5, 0.45), 600);
        fileMp4.classList.add("grab"); ghost.textContent = "lecture.mp4";
        place(ghost, at(fileMp4, 0.5, 0.45).map(function (v, i) { return v + (i ? 10 : 12); }), 0);
        ghost.classList.add("on");
        await sleep(250);
        var drop = at(pre, 0, 0);
        await pointerTo([drop[0] + Math.min(110, pre.clientWidth * 0.4), drop[1] + 26], 900, true);
        ghost.classList.remove("on"); fileMp4.classList.remove("grab");
        pre.innerHTML = line("subthis " + PATH) + CUR;
        await sleep(650);
        var shown = line("subthis " + PATH);
        for (var k = 0; k < OUT.length; k += 1) {
          shown += "\n" + OUT[k][0];
          pre.innerHTML = shown + CUR;
          await sleep(OUT[k][1]);
        }
        fileSrt.classList.add("on");
        await sleep(700);
        await pointerTo(at(fileSrt, 0.5, 0.45), 600);
        fileSrt.classList.add("grab"); ghost.textContent = "lecture.srt";
        place(ghost, at(fileSrt, 0.5, 0.45).map(function (v, i) { return v + (i ? 10 : 12); }), 0);
        ghost.classList.add("on");
        await sleep(250);
        var subsXY = at(subs, 0.45, 0.5);
        var p = pointerTo(subsXY, 900, true);
        setTimeout(function () { subs.classList.add("over"); }, 650);
        await p;
        ghost.classList.remove("on"); fileSrt.classList.remove("grab"); subs.classList.remove("over");
        for (var q = 0; q < cues.length; q += 1) { cues[q].classList.add("on"); await sleep(55); }
        caption.classList.add("on");
        await pointerTo(at(subs, 0.85, 1.9), 500);
        await sleep(3600);
        pointer.classList.remove("on");
        await sleep(300);
      }
      async function loop() { for (;;) { await run(); } }
      if ("IntersectionObserver" in window) {
        var started = false;
        new IntersectionObserver(function (entries, obs) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting && !started) { started = true; loop(); obs.disconnect(); }
          });
        }, { threshold: 0.3 }).observe(scene);
      } else { loop(); }
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
