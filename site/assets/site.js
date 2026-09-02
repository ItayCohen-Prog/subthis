/* subthis site behaviors: OS detection, tabs, copy, reveal, typing demo. */
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
      /* Reserve the terminal's full height up front so the page below it never
         shifts while lines are typed in or the loop restarts. */
      function reserve() {
        var probe = pre.cloneNode(false);
        probe.innerHTML = lines.map(function (l) { return l.type === "cmd" ? "$ " + l.text : l.html; }).join("\n") + '<span class="cursor"></span>';
        probe.style.position = "absolute"; probe.style.visibility = "hidden"; probe.style.minHeight = "0";
        pre.parentNode.appendChild(probe);
        pre.style.minHeight = probe.offsetHeight + "px";
        pre.parentNode.removeChild(probe);
      }
      reserve();
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(reserve);
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
