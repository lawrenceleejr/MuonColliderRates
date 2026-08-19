/* Muon Collider Rates — interactive figure.
 *
 * No dependencies: the chart is plain SVG built from web/data/curves.json,
 * which make_web_data.py generates from the text files in data/.
 */
(function () {
  "use strict";

  var REPO = document.documentElement.dataset.repo;
  var STORE_KEY = "mcr.custom.v2";
  var THEME_KEY = "mcr.theme";
  var SNOWMASS_YEAR_S = 1e7;

  var GROUPS = [
    { id: "background", title: "Machine-induced & inclusive" },
    { id: "sm", title: "Standard Model" },
    { id: "bsm", title: "BSM benchmarks" },
    { id: "extra", title: "Also in the dataset" },
    { id: "custom", title: "Your curves" }
  ];

  var DASHES = {
    "-": null,
    "--": "6 3.5",
    ":": "1 3.5",
    "-.": "8 3 1 3"
  };

  var DEFAULT_VIEW = { x0: 1, x1: 10, y0: 1e-4, y1: 1e12 };
  var W = 940, H = 660;
  var M = { top: 26, right: 118, bottom: 54, left: 78 };

  var UNIT_TO_FB = { fb: 1, pb: 1e3, nb: 1e6, ub: 1e9, "µb": 1e9, mb: 1e12, b: 1e15 };

  var state = {
    meta: null,
    series: [],
    view: Object.assign({}, DEFAULT_VIEW),
    lumi: 2e35,
    refLines: [],
    showRefs: true,
    showInlineLabels: true,
    hover: null,          // key of the series under the pointer
    pinned: null,         // key of the pinned series
    cursorX: null,        // data-space sqrt(s) under the pointer
    pending: null         // parsed-but-not-yet-added custom curve
  };

  var svg = document.getElementById("plot");
  var tooltip = document.getElementById("tooltip");
  var legendEl = document.getElementById("legend");
  var NS = "http://www.w3.org/2000/svg";

  // ---------------------------------------------------------------- helpers

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    if (attrs) {
      for (var k in attrs) {
        if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function fbToHz(sigmaFb) { return sigmaFb * 1e-39 * state.lumi; }

  function sigFigs(v, n) {
    if (!isFinite(v) || v === 0) return "0";
    var d = Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(v))));
    return v.toFixed(Math.min(20, d));
  }

  /** "1.27 × 10⁷" as HTML, or a plain decimal when that reads better. */
  function fmtValue(v, unit) {
    if (!isFinite(v)) return "&mdash;";
    var suffix = unit ? " " + unit : "";
    var abs = Math.abs(v);
    if (abs >= 0.01 && abs < 10000) return sigFigs(v, 4).replace(/\.?0+$/, "") + suffix;
    var e = Math.floor(Math.log10(abs));
    var m = v / Math.pow(10, e);
    var mantissa = (Math.round(m * 100) / 100).toString();
    return mantissa + " &times; 10<sup>" + e + "</sup>" + suffix;
  }

  function fmtRate(hz) {
    if (!isFinite(hz) || hz <= 0) return "&mdash;";
    if (hz >= 1e9) return sigFigs(hz / 1e9, 3) + " GHz";
    if (hz >= 1e6) return sigFigs(hz / 1e6, 3) + " MHz";
    if (hz >= 1e3) return sigFigs(hz / 1e3, 3) + " kHz";
    if (hz >= 1) return sigFigs(hz, 3) + " Hz";
    if (hz >= 1e-3) return sigFigs(hz * 1e3, 3) + " mHz";
    var perYear = hz * SNOWMASS_YEAR_S;
    return fmtValue(perYear) + " / Snowmass yr";
  }

  function fmtCompact(v) {
    if (!isFinite(v)) return "";
    var abs = Math.abs(v);
    if (abs === 0) return "0";
    if (abs >= 0.1 && abs < 1000) return sigFigs(v, 3).replace(/\.?0+$/, "");
    var e = Math.floor(Math.log10(abs));
    var m = Math.round((v / Math.pow(10, e)) * 10) / 10;
    return m + "e" + e;
  }

  // ------------------------------------------------------------ data model

  function makeSeries(raw, custom) {
    var pts = raw.points
      .filter(function (p) { return p[0] > 0 && p[1] > 0; })
      .sort(function (a, b) { return a[0] - b[0]; });
    return {
      key: raw.key,
      label: raw.label,
      group: custom ? "custom" : raw.group,
      color: raw.color,
      dash: raw.dash === undefined ? "-" : raw.dash,
      marker: raw.marker || null,
      points: pts,
      process: raw.process || "",
      source: raw.source || "",
      notes: raw.notes || "",
      file: raw.file || "",
      shown: raw.shown !== false,
      defaultShown: raw.shown !== false,
      custom: !!custom,
      screen: []
    };
  }

  /** Linear interpolation in log-log space; null outside the series range. */
  function valueAt(series, x) {
    var p = series.points;
    if (!p.length) return null;
    if (p.length === 1) return Math.abs(x - p[0][0]) / p[0][0] < 0.02 ? p[0][1] : null;
    if (x < p[0][0] || x > p[p.length - 1][0]) return null;
    for (var i = 0; i < p.length - 1; i++) {
      if (x >= p[i][0] && x <= p[i + 1][0]) {
        var lx0 = Math.log(p[i][0]), lx1 = Math.log(p[i + 1][0]);
        if (lx1 === lx0) return p[i][1];
        var t = (Math.log(x) - lx0) / (lx1 - lx0);
        return Math.exp(Math.log(p[i][1]) * (1 - t) + Math.log(p[i + 1][1]) * t);
      }
    }
    return p[p.length - 1][1];
  }

  function visible() {
    return state.series.filter(function (s) { return s.shown && s.points.length; });
  }

  // ------------------------------------------------------------------ axes

  function scaleX(x) {
    var v = state.view;
    return M.left + (Math.log10(x) - Math.log10(v.x0)) /
      (Math.log10(v.x1) - Math.log10(v.x0)) * (W - M.left - M.right);
  }

  function scaleY(y) {
    var v = state.view;
    return H - M.bottom - (Math.log10(y) - Math.log10(v.y0)) /
      (Math.log10(v.y1) - Math.log10(v.y0)) * (H - M.top - M.bottom);
  }

  function unscaleX(px) {
    var v = state.view;
    var f = (px - M.left) / (W - M.left - M.right);
    return Math.pow(10, Math.log10(v.x0) + f * (Math.log10(v.x1) - Math.log10(v.x0)));
  }

  function decades(lo, hi) {
    var out = [];
    for (var d = Math.ceil(Math.log10(lo) - 1e-9); d <= Math.floor(Math.log10(hi) + 1e-9); d++) {
      out.push(d);
    }
    return out;
  }

  function powerLabel(parent, x, y, exp, anchor, cls) {
    var t = el("text", { x: x, y: y, "text-anchor": anchor, class: cls }, parent);
    el("tspan", {}, t).textContent = "10";
    var sup = el("tspan", { dy: "-5", "font-size": "8.5" }, t);
    sup.textContent = String(exp);
    return t;
  }

  /** Render an HTML label (with <sub>/<sup>) into an SVG <text> as tspans. */
  var BASELINE = { normal: 0, sub: 3, sup: -4.5 };

  function setSvgLabel(node, html) {
    while (node.firstChild) node.removeChild(node.firstChild);
    var holder = document.createElement("div");
    holder.innerHTML = html;

    var runs = [];
    (function walk(parent, style) {
      Array.prototype.forEach.call(parent.childNodes, function (child) {
        if (child.nodeType === 3) {
          if (child.nodeValue) runs.push({ text: child.nodeValue, style: style });
          return;
        }
        var tag = child.nodeName.toLowerCase();
        walk(child, tag === "sub" ? "sub" : tag === "sup" ? "sup" : style);
      });
    })(holder, "normal");

    var offset = 0;
    runs.forEach(function (run) {
      var want = BASELINE[run.style] || 0;
      var t = el("tspan", {}, node);
      if (want !== offset) t.setAttribute("dy", (want - offset).toFixed(1));
      if (run.style !== "normal") t.setAttribute("font-size", "8.5");
      t.textContent = run.text;
      offset = want;
    });
    if (offset !== 0) {
      var tail = el("tspan", { dy: (-offset).toFixed(1) }, node);
      tail.textContent = "\u200a";
    }
  }

  // ----------------------------------------------------------------- chart

  function draw() {
    // Rebuild from scratch, but keep <title>/<desc> for assistive tech.
    Array.prototype.slice.call(svg.childNodes).forEach(function (node) {
      if (node.nodeName !== "title" && node.nodeName !== "desc") svg.removeChild(node);
    });

    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

    var v = state.view;
    var gGrid = el("g", { class: "g-grid" }, svg);
    var gAxes = el("g", { class: "g-axes" }, svg);
    var gRefs = el("g", { class: "g-refs" }, svg);
    var gData = el("g", { class: "g-data" }, svg);
    var gLabels = el("g", { class: "g-labels" }, svg);
    var gHit = el("g", { class: "g-hit" }, svg);
    var gCursor = el("g", { class: "g-cursor" }, svg);

    var x0 = M.left, x1 = W - M.right, y0 = M.top, y1 = H - M.bottom;

    // -- y grid + left axis
    decades(v.y0, v.y1).forEach(function (d) {
      var y = scaleY(Math.pow(10, d));
      el("line", { class: "grid-line", x1: x0, x2: x1, y1: y, y2: y }, gGrid);
      el("line", { class: "tick-mark", x1: x0 - 5, x2: x0, y1: y, y2: y }, gAxes);
      powerLabel(gAxes, x0 - 9, y + 4, d, "end", "tick-label");
    });

    // -- x grid + bottom axis
    var xDecades = Math.log10(v.x1) - Math.log10(v.x0);
    var majors = [];
    for (var d = Math.floor(Math.log10(v.x0)); d <= Math.ceil(Math.log10(v.x1)); d++) {
      for (var m = 1; m <= 9; m++) {
        var xv = m * Math.pow(10, d);
        if (xv >= v.x0 * 0.999 && xv <= v.x1 * 1.001) majors.push({ v: xv, minor: m !== 1 });
      }
    }
    majors.forEach(function (t) {
      var x = scaleX(t.v);
      var label = xDecades <= 1.35 || !t.minor;
      el("line", { class: "grid-line", x1: x, x2: x, y1: y0, y2: y1, opacity: t.minor ? 0.5 : 1 }, gGrid);
      el("line", { class: "tick-mark", x1: x, x2: x, y1: y1, y2: y1 + (t.minor ? 3 : 5) }, gAxes);
      if (!label) return;
      if (t.v >= 1000 || t.v < 0.01) {
        powerLabel(gAxes, x, y1 + 19, Math.round(Math.log10(t.v)), "middle", "tick-label");
      } else {
        el("text", { x: x, y: y1 + 19, "text-anchor": "middle", class: "tick-label" }, gAxes)
          .textContent = String(+t.v.toPrecision(3));
      }
    });

    el("line", { class: "axis-line", x1: x0, x2: x1, y1: y1, y2: y1 }, gAxes);
    el("line", { class: "axis-line", x1: x0, x2: x0, y1: y0, y2: y1 }, gAxes);
    el("line", { class: "axis-line", x1: x1, x2: x1, y1: y0, y2: y1 }, gAxes);

    // -- right axis: the same numbers as a rate
    var rateLo = fbToHz(v.y0), rateHi = fbToHz(v.y1);
    decades(rateLo, rateHi).forEach(function (d) {
      var sigma = Math.pow(10, d) / (1e-39 * state.lumi);
      var y = scaleY(sigma);
      if (y < y0 - 1 || y > y1 + 1) return;
      el("line", { class: "tick-mark", x1: x1, x2: x1 + 5, y1: y, y2: y }, gAxes);
      powerLabel(gAxes, x1 + 9, y + 4, d, "start", "tick-label");
    });

    var xt = el("text", {
      x: (x0 + x1) / 2, y: H - 14, "text-anchor": "middle", class: "axis-title"
    }, gAxes);
    el("tspan", { "font-style": "italic" }, xt).textContent = "√s";
    el("tspan", {}, xt).textContent = "  [TeV]";

    var yt = el("text", {
      x: 0, y: 0, "text-anchor": "middle", class: "axis-title",
      transform: "translate(" + (x0 - 46) + "," + (y0 + y1) / 2 + ") rotate(-90)"
    }, gAxes);
    el("tspan", { "font-style": "italic" }, yt).textContent = "σ";
    el("tspan", {}, yt).textContent = "  [fb]";

    var rt = el("text", {
      x: 0, y: 0, "text-anchor": "middle", class: "axis-title",
      transform: "translate(" + (x1 + 62) + "," + (y0 + y1) / 2 + ") rotate(90)"
    }, gAxes);
    rt.textContent = "Rate [Hz]";
    var rt2 = el("text", {
      x: 0, y: 0, "text-anchor": "middle", class: "ref-label",
      transform: "translate(" + (x1 + 76) + "," + (y0 + y1) / 2 + ") rotate(90)"
    }, gAxes);
    rt2.textContent = "at L = 2 × 10³⁵ cm⁻² s⁻¹";

    // -- reference lines
    if (state.showRefs) {
      state.refLines.forEach(function (ref) {
        var sigma = ref.rate_hz / (1e-39 * state.lumi);
        if (sigma < v.y0 || sigma > v.y1) return;
        var y = scaleY(sigma);
        el("line", { class: "ref-line", x1: x0, x2: x1, y1: y, y2: y }, gRefs);
        el("text", {
          x: x0 + 6, y: y - 4, "text-anchor": "start", class: "ref-label"
        }, gRefs).textContent = ref.label;
      });
    }

    // -- clip so curves never spill past the frame
    var defs = el("defs", {}, svg);
    var clip = el("clipPath", { id: "plot-clip" }, defs);
    el("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }, clip);
    gData.setAttribute("clip-path", "url(#plot-clip)");
    gHit.setAttribute("clip-path", "url(#plot-clip)");
    gLabels.setAttribute("clip-path", "url(#plot-clip)");

    var active = state.pinned || state.hover;
    var list = visible();

    list.forEach(function (s) {
      s.screen = s.points.map(function (p) { return [scaleX(p[0]), scaleY(p[1])]; });
      var faded = active && active !== s.key;
      var isActive = active === s.key;
      var base = s.group === "background" ? 1.4 : 1.8;
      var opacity = faded ? 0.3 : (s.group === "background" ? 0.75 : 1);

      if (s.screen.length > 1) {
        var dstr = s.screen.map(function (p, i) {
          return (i ? "L" : "M") + p[0].toFixed(2) + " " + p[1].toFixed(2);
        }).join(" ");
        el("path", {
          class: "series", d: dstr, stroke: s.color,
          "stroke-width": isActive ? base + 1.6 : base,
          "stroke-dasharray": isActive ? null : DASHES[s.dash] || null,
          opacity: opacity
        }, gData);
        var hit = el("path", { class: "series-hit", d: dstr, "data-key": s.key }, gHit);
        hit.addEventListener("click", function () { togglePin(s.key); });
      }

      if (s.marker || s.screen.length === 1) {
        s.screen.forEach(function (p) {
          el("circle", {
            cx: p[0], cy: p[1], r: isActive ? 5 : 3.6, fill: s.color, opacity: opacity
          }, gData);
        });
        if (s.screen.length === 1) {
          var dot = el("circle", {
            cx: s.screen[0][0], cy: s.screen[0][1], r: 12,
            fill: "transparent", class: "series-hit", "data-key": s.key
          }, gHit);
          dot.addEventListener("click", function () { togglePin(s.key); });
        }
      }
    });

    if (state.showInlineLabels) drawInlineLabels(gLabels, list, active);
    if (state.cursorX !== null) drawCursor(gCursor, active);
  }

  /** End-of-curve labels, pushed apart vertically so they stay readable. */
  function drawInlineLabels(parent, list, active) {
    var x1 = W - M.right;
    var placed = [];
    list.forEach(function (s) {
      if (!s.screen.length) return;
      var last = s.screen[s.screen.length - 1];
      if (last[0] < M.left || last[1] < M.top - 6 || last[1] > H - M.bottom + 6) return;
      placed.push({ s: s, x: Math.min(last[0], x1) - 8, y: last[1] });
    });
    placed.sort(function (a, b) { return a.y - b.y; });
    var minGap = 14;
    for (var i = 1; i < placed.length; i++) {
      if (placed[i].y - placed[i - 1].y < minGap) placed[i].y = placed[i - 1].y + minGap;
    }
    var overflow = placed.length ? placed[placed.length - 1].y - (H - M.bottom - 4) : 0;
    if (overflow > 0) placed.forEach(function (p) { p.y -= overflow; });

    placed.forEach(function (p) {
      var faded = active && active !== p.s.key;
      var t = el("text", {
        x: p.x, y: p.y + 3.5, "text-anchor": "end", class: "inline-label",
        opacity: faded ? 0.28 : 0.95,
        "font-weight": active === p.s.key ? 700 : 400
      }, parent);
      // Inline style, not a fill attribute: the stylesheet's `#plot text` rule
      // would otherwise win over a presentation attribute.
      t.style.fill = p.s.color;
      setSvgLabel(t, p.s.label);
    });
  }

  function drawCursor(parent, activeKey) {
    var x = scaleX(state.cursorX);
    if (x < M.left || x > W - M.right) return;
    el("line", {
      class: "crosshair", x1: x, x2: x, y1: M.top, y2: H - M.bottom
    }, parent);
    if (!activeKey) return;
    var s = state.series.filter(function (t) { return t.key === activeKey; })[0];
    if (!s) return;
    var y = valueAt(s, state.cursorX);
    if (y === null || y < state.view.y0 || y > state.view.y1) return;
    el("circle", {
      cx: x, cy: scaleY(y), r: 5.5, fill: "none", stroke: s.color, "stroke-width": 2
    }, parent);
    el("circle", { cx: x, cy: scaleY(y), r: 2.2, fill: s.color }, parent);
  }

  // ----------------------------------------------------------- interaction

  function pointerData(evt) {
    var rect = svg.getBoundingClientRect();
    var sx = (evt.clientX - rect.left) / rect.width * W;
    var sy = (evt.clientY - rect.top) / rect.height * H;
    return { sx: sx, sy: sy };
  }

  function distToSegment(px, py, ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay;
    var len2 = dx * dx + dy * dy;
    var t = len2 ? clamp(((px - ax) * dx + (py - ay) * dy) / len2, 0, 1) : 0;
    var qx = ax + t * dx, qy = ay + t * dy;
    return Math.hypot(px - qx, py - qy);
  }

  function nearestSeries(sx, sy) {
    var best = null, bestD = 42;
    visible().forEach(function (s) {
      var pts = s.screen;
      if (!pts.length) return;
      var d;
      if (pts.length === 1) {
        d = Math.hypot(sx - pts[0][0], sy - pts[0][1]);
      } else {
        d = Infinity;
        for (var i = 0; i < pts.length - 1; i++) {
          var dd = distToSegment(sx, sy, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
          if (dd < d) d = dd;
        }
      }
      if (d < bestD) { bestD = d; best = s.key; }
    });
    return best;
  }

  function onPointerMove(evt) {
    var p = pointerData(evt);
    if (p.sx < M.left - 4 || p.sx > W - M.right + 4 || p.sy < M.top - 4 || p.sy > H - M.bottom + 4) {
      return onPointerLeave();
    }
    state.cursorX = clamp(unscaleX(p.sx), state.view.x0, state.view.x1);
    var near = nearestSeries(p.sx, p.sy);
    var changed = near !== state.hover;
    state.hover = near;
    draw();
    updateReadout();
    showTooltip(evt, state.pinned || near);
    if (changed) syncLegendActive();
  }

  function onPointerLeave() {
    state.hover = null;
    state.cursorX = null;
    draw();
    updateReadout();
    tooltip.classList.remove("visible");
    syncLegendActive();
  }

  function showTooltip(evt, key) {
    if (!key) { tooltip.classList.remove("visible"); return; }
    var s = state.series.filter(function (t) { return t.key === key; })[0];
    if (!s) { tooltip.classList.remove("visible"); return; }

    var x = state.cursorX, y = valueAt(s, x);
    if (y === null && s.points.length) {
      var last = s.points[s.points.length - 1], first = s.points[0];
      var target = x < first[0] ? first : last;
      x = target[0]; y = target[1];
    }
    if (y === null) { tooltip.classList.remove("visible"); return; }

    var html = '<div class="tt-title">' + s.label + "</div><dl>";
    html += "<dt>&radic;s</dt><dd>" + sigFigs(x, 3) + " TeV</dd>";
    html += "<dt>&sigma;</dt><dd>" + fmtValue(y, "fb") + "</dd>";
    html += "<dt>Rate</dt><dd>" + fmtRate(fbToHz(y)) + "</dd>";
    html += "</dl>";
    if (s.process) html += '<div class="tt-note">' + escapeHtml(s.process) + "</div>";
    if (state.pinned === key) html += '<div class="tt-note">Pinned &mdash; click again to release</div>';
    tooltip.innerHTML = html;
    tooltip.style.borderLeftColor = s.color;
    tooltip.classList.add("visible");

    var wrap = svg.parentNode.getBoundingClientRect();
    var tw = tooltip.offsetWidth, th = tooltip.offsetHeight;
    var lx = evt.clientX - wrap.left + 16;
    var ly = evt.clientY - wrap.top - th - 12;
    if (lx + tw > wrap.width - 4) lx = evt.clientX - wrap.left - tw - 16;
    if (ly < 4) ly = evt.clientY - wrap.top + 18;
    tooltip.style.left = Math.max(4, lx) + "px";
    tooltip.style.top = Math.max(4, ly) + "px";
  }

  function togglePin(key) {
    state.pinned = state.pinned === key ? null : key;
    document.getElementById("pin-hint").textContent =
      state.pinned ? "Pinned — click the curve again to release" : "Click a curve to pin it";
    draw();
    syncLegendActive();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // ---------------------------------------------------------------- legend

  function swatchSvg(s) {
    var dash = DASHES[s.dash];
    var stroke = s.dash === null && !s.marker
      ? ""
      : '<line x1="1" y1="7" x2="29" y2="7" stroke="' + s.color +
        '" stroke-width="2.4"' + (dash ? ' stroke-dasharray="' + dash + '"' : "") + ' />';
    var dot = (s.marker || s.points.length === 1)
      ? '<circle cx="15" cy="7" r="3.4" fill="' + s.color + '" />' : "";
    return '<svg class="swatch" viewBox="0 0 30 14" aria-hidden="true">' + stroke + dot + "</svg>";
  }

  function buildLegend() {
    legendEl.innerHTML = "";
    GROUPS.forEach(function (g) {
      var members = state.series.filter(function (s) { return s.group === g.id; });
      if (!members.length) return;
      var box = document.createElement("div");
      box.className = "legend-group";
      var h = document.createElement("h3");
      h.innerHTML = "<span>" + g.title + "</span>";
      var acts = document.createElement("span");
      acts.className = "actions";
      ["All", "None"].forEach(function (word) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "btn btn-mini";
        b.textContent = word;
        b.addEventListener("click", function () {
          members.forEach(function (s) { s.shown = word === "All"; });
          buildLegend(); draw(); updateReadout();
        });
        acts.appendChild(b);
      });
      h.appendChild(acts);
      box.appendChild(h);

      members.forEach(function (s) { box.appendChild(legendRow(s)); });
      legendEl.appendChild(box);
    });
    syncLegendActive();
    updateReadout();
  }

  function legendRow(s) {
    var row = document.createElement("div");
    row.className = "legend-item" + (s.shown ? "" : " off");
    row.dataset.key = s.key;
    row.setAttribute("role", "group");

    var cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = s.shown;
    cb.setAttribute("aria-label", "Show " + s.label.replace(/<[^>]+>/g, ""));
    cb.addEventListener("change", function () {
      s.shown = cb.checked;
      row.classList.toggle("off", !s.shown);
      draw(); updateReadout();
    });
    row.appendChild(cb);

    var sw = document.createElement("span");
    sw.innerHTML = swatchSvg(s);
    row.appendChild(sw.firstChild);

    var lab = document.createElement("span");
    lab.className = "legend-label";
    lab.innerHTML = s.label;
    lab.title = [s.label.replace(/<[^>]+>/g, ""), s.process, s.source].filter(Boolean).join(" — ");
    row.appendChild(lab);

    var tail = document.createElement("span");
    tail.style.display = "flex";
    tail.style.alignItems = "center";
    tail.style.gap = "0.15rem";
    var val = document.createElement("span");
    val.className = "legend-value";
    tail.appendChild(val);
    if (s.custom) {
      var dl = document.createElement("button");
      dl.type = "button";
      dl.className = "remove";
      dl.textContent = "⤓";
      dl.title = "Download as a repository data file";
      dl.addEventListener("click", function (e) { e.stopPropagation(); downloadSeries(s); });
      tail.appendChild(dl);
      var rm = document.createElement("button");
      rm.type = "button";
      rm.className = "remove";
      rm.textContent = "×";
      rm.title = "Remove this curve";
      rm.addEventListener("click", function (e) {
        e.stopPropagation();
        state.series = state.series.filter(function (t) { return t !== s; });
        if (state.pinned === s.key) state.pinned = null;
        saveCustom(); buildLegend(); draw();
      });
      tail.appendChild(rm);
    }
    row.appendChild(tail);

    row.addEventListener("mouseenter", function () {
      if (state.pinned) return;
      state.hover = s.key; draw(); syncLegendActive();
    });
    row.addEventListener("mouseleave", function () {
      if (state.pinned) return;
      state.hover = null; draw(); syncLegendActive();
    });
    lab.addEventListener("click", function () { togglePin(s.key); });
    return row;
  }

  function syncLegendActive() {
    var active = state.pinned || state.hover;
    Array.prototype.forEach.call(legendEl.querySelectorAll(".legend-item"), function (row) {
      row.classList.toggle("active", row.dataset.key === active);
    });
  }

  function updateReadout() {
    var x = state.cursorX === null ? state.view.x1 : state.cursorX;
    document.getElementById("readout-x").innerHTML = "&radic;s = " + sigFigs(x, 3) + " TeV";
    Array.prototype.forEach.call(legendEl.querySelectorAll(".legend-item"), function (row) {
      var s = state.series.filter(function (t) { return t.key === row.dataset.key; })[0];
      var out = row.querySelector(".legend-value");
      if (!s || !out) return;
      var y = s.shown ? valueAt(s, x) : null;
      out.textContent = y === null ? "" : fmtCompact(y);
      out.title = y === null ? "" : fmtRate(fbToHz(y));
    });
  }

  // ----------------------------------------------------------- custom data

  function parseCurve(text, forcedUnit) {
    var lines = String(text).split(/\r?\n/);
    var unit = "fb", title = null, meta = {};
    var pts = [];
    lines.forEach(function (line) {
      var t = line.trim();
      if (!t) return;
      if (t.charAt(0) === "#") {
        var body = t.replace(/^#+\s*/, "");
        var m = body.match(/^([A-Za-z_ ]+)\s*:\s*(.*)$/);
        if (m) {
          var field = m[1].trim().toLowerCase();
          meta[field] = m[2].trim();
          if (field === "title") title = m[2].trim();
          if (field === "columns") {
            var units = m[2].match(/\[([^\]]+)\]/g) || [];
            if (units.length > 1) {
              var u = units[1].replace(/[[\]]/g, "").trim();
              if (UNIT_TO_FB[u]) unit = u;
            }
          }
        }
        return;
      }
      var fields = t.split(/[,;\s]+/).filter(function (f) { return f.length; });
      if (fields.length < 2) return;
      var x = parseFloat(fields[0]), y = parseFloat(fields[1]);
      if (!isFinite(x) || !isFinite(y)) return;
      pts.push([x, y]);
    });
    if (forcedUnit && forcedUnit !== "header") unit = forcedUnit;
    var scale = UNIT_TO_FB[unit] || 1;
    return {
      points: pts.map(function (p) { return [p[0], p[1] * scale]; }),
      title: title,
      unit: unit,
      meta: meta
    };
  }

  function addCustom(parsed, title, color) {
    var usable = parsed.points.filter(function (p) { return p[0] > 0 && p[1] > 0; });
    if (usable.length < 1) throw new Error("no usable rows: need positive √s and σ values");
    var key = "custom:" + title + ":" + Date.now();
    var s = makeSeries({
      key: key,
      label: escapeHtml(title),
      color: color,
      dash: usable.length > 1 ? "-" : null,
      marker: usable.length <= 12 ? "o" : null,
      points: usable,
      process: parsed.meta.process || "",
      source: parsed.meta.source || "added in the browser",
      notes: parsed.meta.notes || "",
      shown: true
    }, true);
    s.unit = parsed.unit;
    state.series.push(s);
    saveCustom();
    buildLegend();
    draw();
    return s;
  }

  function saveCustom() {
    try {
      var mine = state.series.filter(function (s) { return s.custom; }).map(function (s) {
        return {
          key: s.key, label: s.label, color: s.color, points: s.points,
          dash: s.dash, marker: s.marker, shown: s.shown,
          process: s.process, source: s.source, notes: s.notes, unit: s.unit
        };
      });
      localStorage.setItem(STORE_KEY, JSON.stringify(mine));
    } catch (e) { /* private mode, quota — not worth interrupting the user */ }
  }

  function loadCustom() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return;
      JSON.parse(raw).forEach(function (r) { state.series.push(makeSeries(r, true)); });
    } catch (e) { /* ignore corrupt storage */ }
  }

  function slug(text) {
    return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 32) || "mycurve";
  }

  function downloadSeries(s) {
    var plain = s.label.replace(/<[^>]+>/g, "");
    var lines = [
      "# title: " + plain,
      "# process: " + (s.process || "TODO: describe the process and any cuts"),
      "# source: " + (s.source || "TODO: calculation, paper or private communication"),
      "# columns: sqrt_s [TeV], sigma [fb]"
    ];
    s.points.forEach(function (p) { lines.push(p[0] + ", " + p[1]); });
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = slug(plain) + ".txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ---------------------------------------------------------------- sundry

  function fitView() {
    var list = visible();
    if (!list.length) return;
    var xs = [], ys = [];
    list.forEach(function (s) {
      s.points.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); });
    });
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    state.view = {
      x0: Math.pow(10, Math.floor(Math.log10(xmin) * 20) / 20),
      x1: Math.pow(10, Math.ceil(Math.log10(xmax) * 20) / 20),
      y0: Math.pow(10, Math.floor(Math.log10(ymin)) - 0.3),
      y1: Math.pow(10, Math.ceil(Math.log10(ymax)) + 0.3)
    };
    draw();
  }

  function buildSourceList() {
    var ul = document.getElementById("source-list");
    ul.innerHTML = "";
    state.series.filter(function (s) { return !s.custom; }).forEach(function (s) {
      var li = document.createElement("li");
      var dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = s.color;
      li.appendChild(dot);
      var body = document.createElement("div");
      var name = document.createElement("div");
      name.className = "name";
      name.innerHTML = s.label;
      body.appendChild(name);
      var detail = document.createElement("div");
      detail.className = "detail";
      detail.innerHTML = linkify(escapeHtml([s.process, s.source].filter(Boolean).join(" — ")));
      if (s.file) {
        detail.innerHTML += ' &middot; <a href="' + REPO + "/blob/main/" + s.file + '">' + s.file + "</a>";
      }
      body.appendChild(detail);
      li.appendChild(body);
      ul.appendChild(li);
    });
  }

  function linkify(text) {
    return text.replace(/https?:\/\/[^\s,;)]+/g, function (u) {
      return '<a href="' + u + '" rel="noopener">' + u + "</a>";
    });
  }

  function wireLinks() {
    document.getElementById("link-repo").href = REPO;
    document.getElementById("link-repo-2").href = REPO;
    document.getElementById("link-issue").href = REPO + "/issues/new?title=" +
      encodeURIComponent("Curve request: ") + "&body=" +
      encodeURIComponent(
        "**Process**\n\n**Centre-of-mass energy range**\n\n" +
        "**Where the numbers come from** (paper, arXiv id, generator + settings, or private communication)\n\n" +
        "**Cuts / definitions**\n");
    document.getElementById("link-bug").href = REPO + "/issues/new?title=" +
      encodeURIComponent("Problem with: ") + "&body=" +
      encodeURIComponent("**What looks wrong**\n\n**Which curve**\n\n**Expected value**\n");
    document.getElementById("link-newfile").href = REPO + "/new/main/data";
  }

  function applyTheme(mode) {
    if (mode === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", mode);
    try { localStorage.setItem(THEME_KEY, mode); } catch (e) { /* ignore */ }
  }

  function downloadSvg() {
    var clone = svg.cloneNode(true);
    Array.prototype.forEach.call(clone.querySelectorAll(".series-hit"), function (n) {
      n.parentNode.removeChild(n);
    });
    var css = document.createElementNS(NS, "style");
    var dark = getComputedStyle(document.body).backgroundColor;
    var ink = getComputedStyle(document.body).color;
    css.textContent =
      "text{font-family:'Source Sans 3',Helvetica,Arial,sans-serif;fill:" + ink + "}" +
      ".axis-title{font-family:'Source Serif 4',Georgia,serif;font-size:13px}" +
      ".tick-label{font-size:11px}" +
      ".grid-line{stroke:#e0e3e6;stroke-width:1}" +
      ".axis-line,.tick-mark{stroke:#8a9098;stroke-width:1}" +
      ".ref-line{stroke:#8a9098;stroke-width:1;stroke-dasharray:2 4;opacity:.75}" +
      ".ref-label,.inline-label{font-size:11px}" +
      ".series{fill:none;stroke-linejoin:round;stroke-linecap:round}";
    clone.insertBefore(css, clone.firstChild);
    var bg = document.createElementNS(NS, "rect");
    bg.setAttribute("width", W);
    bg.setAttribute("height", H);
    bg.setAttribute("fill", dark);
    clone.insertBefore(bg, css.nextSibling);
    clone.setAttribute("xmlns", NS);
    var text = new XMLSerializer().serializeToString(clone);
    var blob = new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n', text], { type: "image/svg+xml" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "MuonColliderRates.svg";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ------------------------------------------------------------------ form

  function wireForm() {
    var form = document.getElementById("add-form");
    var fileInput = document.getElementById("custom-file");
    var dropzone = document.getElementById("dropzone");
    var fileLabel = document.getElementById("dropzone-file");
    var paste = document.getElementById("custom-paste");
    var titleInput = document.getElementById("custom-title");
    var status = document.getElementById("add-status");
    var loadedText = null;
    var loadedName = null;

    function say(message, kind) {
      status.textContent = message;
      status.className = "status" + (kind ? " " + kind : "");
    }

    function takeFile(file) {
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        loadedText = String(reader.result);
        loadedName = file.name;
        fileLabel.textContent = file.name;
        if (!titleInput.value) {
          var parsed = parseCurve(loadedText, null);
          titleInput.value = parsed.title || file.name.replace(/\.[^.]+$/, "");
        }
        say("Loaded " + file.name + " — press Add curve.", "ok");
      };
      reader.onerror = function () { say("Could not read that file.", "error"); };
      reader.readAsText(file);
    }

    dropzone.addEventListener("click", function () { fileInput.click(); });
    dropzone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });
    fileInput.addEventListener("change", function () { takeFile(fileInput.files[0]); });
    ["dragenter", "dragover"].forEach(function (t) {
      dropzone.addEventListener(t, function (e) {
        e.preventDefault(); dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (t) {
      dropzone.addEventListener(t, function (e) {
        e.preventDefault(); dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) takeFile(e.dataTransfer.files[0]);
    });

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var text = paste.value.trim() ? paste.value : loadedText;
      if (!text || !text.trim()) {
        say("Add a file or paste some numbers first.", "error");
        return;
      }
      var parsed = parseCurve(text, document.getElementById("custom-unit").value);
      var title = titleInput.value.trim() || parsed.title ||
        (loadedName ? loadedName.replace(/\.[^.]+$/, "") : "My curve");
      try {
        var s = addCustom(parsed, title, document.getElementById("custom-color").value);
        say("Added “" + title + "” with " + s.points.length +
          " point" + (s.points.length === 1 ? "" : "s") +
          (parsed.unit !== "fb" ? " (converted from " + parsed.unit + ")" : "") + ".", "ok");
        paste.value = "";
        titleInput.value = "";
        fileLabel.textContent = "";
        fileInput.value = "";
        loadedText = null;
        loadedName = null;
      } catch (err) {
        say(err.message, "error");
      }
    });

    document.getElementById("btn-clear-custom").addEventListener("click", function () {
      state.series = state.series.filter(function (s) { return !s.custom; });
      saveCustom(); buildLegend(); draw();
      say("Removed your curves.", "ok");
    });

    document.getElementById("btn-export-custom").addEventListener("click", function () {
      var mine = state.series.filter(function (s) { return s.custom; });
      if (!mine.length) { say("Nothing of yours to download yet.", "error"); return; }
      mine.forEach(downloadSeries);
      say("Downloaded " + mine.length + " data file" + (mine.length === 1 ? "" : "s") +
        " ready for a pull request.", "ok");
    });
  }

  function wireToolbar() {
    document.getElementById("btn-reset-view").addEventListener("click", function () {
      state.view = Object.assign({}, DEFAULT_VIEW);
      draw();
    });
    document.getElementById("btn-fit-view").addEventListener("click", fitView);
    document.getElementById("toggle-refs").addEventListener("change", function (e) {
      state.showRefs = e.target.checked; draw();
    });
    document.getElementById("toggle-inline-labels").addEventListener("change", function (e) {
      state.showInlineLabels = e.target.checked; draw();
    });
    document.getElementById("btn-download-svg").addEventListener("click", downloadSvg);

    Array.prototype.forEach.call(document.querySelectorAll("[data-all]"), function (b) {
      b.addEventListener("click", function () {
        var mode = b.dataset.all;
        state.series.forEach(function (s) {
          s.shown = mode === "on" ? true : mode === "off" ? false : s.defaultShown || s.custom;
        });
        buildLegend(); draw();
      });
    });

    var order = ["auto", "light", "dark"];
    document.getElementById("theme-toggle").addEventListener("click", function () {
      var current = localStorage.getItem(THEME_KEY) || "auto";
      var next = order[(order.indexOf(current) + 1) % order.length];
      applyTheme(next);
      this.textContent = next === "auto" ? "Theme" : next === "light" ? "Light" : "Dark";
      draw();
    });

    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerleave", onPointerLeave);
    window.addEventListener("resize", function () { draw(); });
  }

  // ------------------------------------------------------------------ boot

  fetch("data/curves.json", { cache: "no-cache" })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (payload) {
      state.meta = payload;
      state.lumi = payload.nominal_lumi_cm2_s || 2e35;
      state.refLines = payload.reference_lines || [];
      state.series = payload.curves.map(function (c) { return makeSeries(c, false); });
      loadCustom();

      var stamp = document.getElementById("build-stamp");
      if (payload.generated_date) {
        stamp.textContent = "Data as of " + payload.generated_date +
          " (" + payload.generated_from + ")";
      }

      applyTheme(localStorage.getItem(THEME_KEY) || "auto");
      wireLinks();
      wireToolbar();
      wireForm();
      buildLegend();
      buildSourceList();
      draw();
    })
    .catch(function (err) {
      var wrap = svg.parentNode;
      var p = document.createElement("p");
      p.className = "status error";
      p.style.padding = "2rem";
      p.textContent = "Could not load the curve data (" + err.message +
        "). If you are opening this file straight from disk, serve the folder over " +
        "HTTP instead — for example: python3 -m http.server.";
      wrap.appendChild(p);
    });
})();
