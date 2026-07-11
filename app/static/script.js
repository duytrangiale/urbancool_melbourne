// UrbanCool Melbourne — console frontend interactivity.
// No framework: the map/KPIs/lists are baked into index.html at build time (static,
// only changes when the model is retrained — see app/build_static.py); this file only
// handles the genuinely dynamic parts (view switching, map zoom/layers/highlighting,
// and the two panels that call the live FastAPI backend: Feature Explorer's per-suburb
// SHAP and the What-If Simulator's re-prediction).

(function () {
  "use strict";

  const layerRanges = JSON.parse(document.getElementById("layer-ranges-data").textContent);
  const suburbLookup = JSON.parse(document.getElementById("suburb-lookup-data").textContent);

  // ---------- Glossary modal ----------
  const glossaryOverlay = document.getElementById("glossary-overlay");
  document.getElementById("glossary-open").addEventListener("click", () => glossaryOverlay.classList.add("open"));
  document.getElementById("glossary-close").addEventListener("click", () => glossaryOverlay.classList.remove("open"));
  glossaryOverlay.addEventListener("click", (e) => {
    if (e.target === glossaryOverlay) glossaryOverlay.classList.remove("open");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") glossaryOverlay.classList.remove("open");
  });

  // ---------- View tab switching ----------
  const viewTabs = document.querySelectorAll(".view-tab");
  const viewPanels = document.querySelectorAll(".view-panel");

  viewTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      viewTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const targetId = tab.dataset.view;
      viewPanels.forEach((p) => {
        const isTarget = p.id === targetId;
        p.style.display = isTarget ? "contents" : "none";
        p.classList.toggle("active", isTarget);
      });
      if (targetId === "view-explorer") ensureExplorerLoaded();
      if (targetId === "view-whatif") ensureWhatIfLoaded();
    });
  });

  // ============================================================
  // HEAT MAP: layer toggle, hover tooltip, zoom/pan, click-highlight
  // ============================================================
  const mapSvg = document.getElementById("map-svg");
  const mapZoomGroup = document.getElementById("map-zoom-group");
  const mapStage = document.getElementById("map-stage");
  const tooltip = document.getElementById("map-tooltip");
  const allPolys = Array.from(document.querySelectorAll(".sa2-poly"));

  // --- Layer toggle ---
  const LAYER_CONFIG = {
    heat: { fillAttr: "data-fill-heat", valAttr: "data-val-heat", unit: "°C", title: "°C above baseline", caption: "SA1→SA2 aggregated prediction · 361 SA2 areas" },
    tree: { fillAttr: "data-fill-tree", valAttr: "data-val-tree", unit: "%", title: "Tree cover (%)", caption: "State vegetation data · full coverage" },
    veg: { fillAttr: "data-fill-veg", valAttr: "data-val-veg", unit: "%", title: "Vegetation cover (%)", caption: "State vegetation data · full coverage" },
  };
  let currentLayer = "heat";

  function applyLayer(layer) {
    currentLayer = layer;
    const cfg = LAYER_CONFIG[layer];
    allPolys.forEach((p) => {
      const fill = p.getAttribute(cfg.fillAttr);
      p.setAttribute("fill", fill || "#22394a"); // grey-ish fallback for suburbs missing this layer's data
      p.style.opacity = fill ? "1" : "0.35";
    });
    document.getElementById("legend-title").textContent = cfg.title;
    document.getElementById("map-layer-caption").textContent = cfg.caption;
    const [vmin, vmax] = layerRanges[layer];
    const scale = document.getElementById("legend-scale");
    scale.innerHTML = `<span>${vmin.toFixed(1)}</span><span>${((vmin + vmax) / 2).toFixed(1)}</span><span>${vmax.toFixed(1)}</span>`;
    const rampCss = layer === "heat"
      ? "linear-gradient(90deg, var(--heat-cool), var(--heat-mid), var(--heat-hot))"
      : "linear-gradient(90deg, #0f1e29, #2d7a5a, #6edcaa)";
    document.getElementById("legend-ramp").style.background = rampCss;
  }

  document.querySelectorAll(".layer-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".layer-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyLayer(btn.dataset.layer);
    });
  });

  // --- Hover tooltip ---
  if (mapStage && tooltip) {
    mapStage.addEventListener("mousemove", (e) => {
      const target = e.target.closest(".sa2-poly");
      if (!target) {
        tooltip.style.display = "none";
        return;
      }
      const rect = mapStage.getBoundingClientRect();
      const cfg = LAYER_CONFIG[currentLayer];
      const val = target.getAttribute(cfg.valAttr);
      tooltip.style.display = "block";
      tooltip.style.left = e.clientX - rect.left + "px";
      tooltip.style.top = e.clientY - rect.top + "px";
      tooltip.innerHTML = val
        ? `${target.dataset.name}<span class="t-val">${val}${cfg.unit}</span>`
        : `${target.dataset.name}<span class="t-val">no data</span>`;
    });
    mapStage.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
  }

  // --- Zoom / pan ---
  // SVG-native transform (user-space units matching the 0..900/0..760 viewBox), not a
  // CSS transform, so panning/zooming math stays in the same coordinate system as the
  // baked polygon paths.
  let zoom = { scale: 1, x: 0, y: 0 };
  const ZOOM_MIN = 1, ZOOM_MAX = 10;

  function applyZoom() {
    mapZoomGroup.setAttribute("transform", `translate(${zoom.x},${zoom.y}) scale(${zoom.scale})`);
  }

  function svgPoint(evt) {
    const pt = mapSvg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    return pt.matrixTransform(mapSvg.getScreenCTM().inverse());
  }

  function zoomAt(clientPoint, newScale) {
    newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newScale));
    // Keep the point under the cursor/centre fixed while scale changes.
    zoom.x = clientPoint.x - ((clientPoint.x - zoom.x) / zoom.scale) * newScale;
    zoom.y = clientPoint.y - ((clientPoint.y - zoom.y) / zoom.scale) * newScale;
    zoom.scale = newScale;
    applyZoom();
  }

  mapStage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const pt = svgPoint(e);
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    zoomAt(pt, zoom.scale * factor);
  }, { passive: false });

  document.getElementById("zoom-in").addEventListener("click", () => zoomAt({ x: 450, y: 380 }, zoom.scale * 1.4));
  document.getElementById("zoom-out").addEventListener("click", () => zoomAt({ x: 450, y: 380 }, zoom.scale / 1.4));
  document.getElementById("zoom-reset").addEventListener("click", () => {
    zoom = { scale: 1, x: 0, y: 0 };
    applyZoom();
  });

  let dragging = false;
  let dragStart = null;
  mapSvg.addEventListener("mousedown", (e) => {
    dragging = true;
    mapSvg.classList.add("panning");
    dragStart = { x: e.clientX, y: e.clientY, zoomX: zoom.x, zoomY: zoom.y };
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    // Screen-pixel delta scaled into SVG user-space (viewBox width / rendered width).
    const rect = mapSvg.getBoundingClientRect();
    const scaleFactor = 900 / rect.width;
    zoom.x = dragStart.zoomX + (e.clientX - dragStart.x) * scaleFactor;
    zoom.y = dragStart.zoomY + (e.clientY - dragStart.y) * scaleFactor;
    applyZoom();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    mapSvg.classList.remove("panning");
  });

  // --- Click-to-highlight from the ranked lists ---
  function highlightAndFrame(suburbName) {
    allPolys.forEach((p) => p.classList.toggle("highlighted", p.dataset.name === suburbName));
    const target = allPolys.find((p) => p.dataset.name === suburbName);
    if (!target) return;
    const bbox = target.getBBox();
    const cx = bbox.x + bbox.width / 2;
    const cy = bbox.y + bbox.height / 2;
    const targetScale = Math.min(ZOOM_MAX, Math.max(3, 700 / Math.max(bbox.width, bbox.height)));
    zoom.scale = targetScale;
    zoom.x = 450 - cx * targetScale;
    zoom.y = 380 - cy * targetScale;
    applyZoom();
  }

  function clearHighlight() {
    allPolys.forEach((p) => p.classList.remove("highlighted"));
    document.querySelectorAll("ul.rank-list li[data-suburb]").forEach((el) => el.classList.remove("active-row"));
  }

  document.querySelectorAll("#hottest-list li[data-suburb], #coolest-list li[data-suburb]").forEach((li) => {
    const activate = () => {
      document.querySelectorAll("ul.rank-list li[data-suburb]").forEach((el) => el.classList.remove("active-row"));
      li.classList.add("active-row");
      // Make sure we're actually looking at the Heat Map view when jumping to a suburb.
      document.querySelector('.view-tab[data-view="view-heatmap"]').click();
      highlightAndFrame(li.dataset.suburb);
    };
    li.addEventListener("click", (e) => { e.stopPropagation(); activate(); });
    li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
  });

  // Clicking anywhere outside a ranked-list row clears the highlight (so it doesn't get stuck on).
  document.addEventListener("click", clearHighlight);

  // ============================================================
  // FEATURE EXPLORER
  // ============================================================
  let explorerLoaded = false;
  const explorerSuburbSelect = document.getElementById("explorer-suburb");
  const subtabs = document.querySelectorAll(".subtab");
  const subpanels = document.querySelectorAll(".subpanel");

  subtabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      subtabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      subpanels.forEach((p) => p.classList.toggle("active", p.id === "subpanel-" + tab.dataset.subtab));
    });
  });

  function renderSnapshot(listEl, snapshot) {
    listEl.innerHTML = `
      <li><span class="rank-name">Tree cover</span><span class="rank-val">${snapshot.tree_cover_pct_state.toFixed(1)}%</span></li>
      <li><span class="rank-name">Vegetation cover</span><span class="rank-val">${snapshot.vegetation_cover_pct_state.toFixed(1)}%</span></li>
      <li><span class="rank-name">Impervious ratio</span><span class="rank-val">${(snapshot.impervious_ratio * 100).toFixed(1)}%</span></li>
      <li><span class="rank-name">Population density</span><span class="rank-val">${Math.round(snapshot.population_density).toLocaleString()}/km&sup2;</span></li>
      <li><span class="rank-name">Predicted heat</span><span class="rank-val v-hot">${snapshot.predicted_mean_uhi_2018.toFixed(2)}&deg;C</span></li>
    `;
  }

  async function loadExplorerSuburb(suburb) {
    const wrap = document.getElementById("shap-wrap");
    wrap.innerHTML = '<span class="loading-text">Computing SHAP explanation&hellip;</span>';
    const [snapshotResp, shapImg] = await Promise.all([
      fetch(`/api/snapshot/${encodeURIComponent(suburb)}`).then((r) => r.json()),
      Promise.resolve(`/api/shap/${encodeURIComponent(suburb)}?t=${Date.now()}`),
    ]);
    renderSnapshot(document.getElementById("explorer-snapshot"), snapshotResp);
    wrap.innerHTML = `<img src="${shapImg}" alt="SHAP waterfall for ${suburb}" />`;
  }

  function ensureExplorerLoaded() {
    if (explorerLoaded) return;
    explorerLoaded = true;
    loadExplorerSuburb(explorerSuburbSelect.value);
  }

  explorerSuburbSelect.addEventListener("change", () => loadExplorerSuburb(explorerSuburbSelect.value));

  // ============================================================
  // WHAT-IF SIMULATOR
  // ============================================================
  let whatIfLoaded = false;
  const whatIfSuburbSelect = document.getElementById("whatif-suburb");
  const sliderTree = document.getElementById("slider-tree");
  const sliderVeg = document.getElementById("slider-veg");
  const sliderGreen = document.getElementById("slider-green");
  const valTree = document.getElementById("val-tree");
  const valVeg = document.getElementById("val-veg");
  const valGreen = document.getElementById("val-green");

  let whatIfDebounce = null;
  let currentSnapshot = null;

  // Mirrors app/core.py::heat_ramp exactly (three colour stops) so the mini-map shading
  // matches the main map's language without a round-trip to the server on every slider
  // move — see DAY_6.md for why this one small duplication was judged worth it.
  function heatRampJs(t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [[0, [255, 210, 138]], [0.5, [255, 138, 76]], [1, [232, 72, 58]]];
    for (let i = 0; i < stops.length - 1; i++) {
      const [t0, c0] = stops[i];
      const [t1, c1] = stops[i + 1];
      if (t >= t0 && t <= t1) {
        const f = (t - t0) / (t1 - t0);
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
        return `rgb(${r},${g},${b})`;
      }
    }
    return "rgb(232,72,58)";
  }

  function renderMiniMap(suburb, predictedHeat) {
    const svg = document.getElementById("whatif-mini-map");
    const entry = suburbLookup[suburb];
    document.getElementById("whatif-map-title").textContent = suburb;
    if (!entry) {
      svg.innerHTML = "";
      return;
    }
    const [vmin, vmax] = layerRanges.heat;
    const t = (predictedHeat - vmin) / (vmax - vmin);
    svg.innerHTML = `<path d="${entry.d}" fill="${heatRampJs(t)}" stroke="rgba(10,20,28,0.6)" stroke-width="1.2"></path>`;
  }

  function renderProportionBar(tree, veg, impervious) {
    const bar = document.getElementById("whatif-proportion-bar");
    const treePct = Math.max(0, tree);
    const otherVegPct = Math.max(0, veg - tree);
    const imperviousPct = Math.max(0, impervious * 100);
    const otherPct = Math.max(0, 100 - treePct - otherVegPct - imperviousPct);

    const segments = [
      { pct: treePct, color: "#6edcaa", label: "Tree canopy" },
      { pct: otherVegPct, color: "#2d7a5a", label: "Other vegetation" },
      { pct: imperviousPct, color: "#8fa8b3", label: "Impervious surface" },
      { pct: otherPct, color: "#22394a", label: "Other (bare/water/etc.)" },
    ];
    bar.innerHTML = segments.map((s) => `<div style="width:${s.pct}%;background:${s.color}" title="${s.label}: ${s.pct.toFixed(1)}%"></div>`).join("");
    document.getElementById("whatif-proportion-legend").innerHTML = segments
      .map((s) => `<span><span class="swatch" style="background:${s.color}"></span>${s.label} (${s.pct.toFixed(1)}%)</span>`)
      .join("");
  }

  function renderWhatIfSnapshot(baseline, modifiedValues) {
    const listEl = document.getElementById("whatif-snapshot");
    const rows = [
      ["Tree cover", baseline.tree_cover_pct_state, modifiedValues.tree, "%"],
      ["Vegetation cover", baseline.vegetation_cover_pct_state, modifiedValues.veg, "%"],
      ["Impervious ratio", baseline.impervious_ratio * 100, modifiedValues.impervious * 100, "%"],
    ];
    listEl.innerHTML = rows.map(([label, before, after, unit]) => {
      const changed = Math.abs(before - after) > 0.001;
      const afterHtml = changed
        ? `<span class="snap-arrow">&rarr;</span><span class="snap-value snap-after changed">${after.toFixed(1)}${unit}</span>`
        : "";
      return `<li><span class="rank-name">${label}</span><span><span class="snap-value">${before.toFixed(1)}${unit}</span>${afterHtml}</span></li>`;
    }).join("") + `<li><span class="rank-name">Population density</span><span class="snap-value">${Math.round(baseline.population_density).toLocaleString()}/km&sup2;</span></li>`;
  }

  async function runWhatIf() {
    const suburb = whatIfSuburbSelect.value;
    const extraTree = Number(sliderTree.value);
    const extraVeg = Number(sliderVeg.value);
    const greenConversion = Number(sliderGreen.value);

    const resp = await fetch("/api/whatif", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ suburb, extra_tree: extraTree, extra_veg: extraVeg, green_conversion: greenConversion }),
    }).then((r) => r.json());

    document.getElementById("whatif-current").textContent = resp.current.toFixed(1) + "°C";
    const afterEl = document.getElementById("whatif-after");
    afterEl.textContent = resp.modified.toFixed(1) + "°C";
    afterEl.className = "kpi-value " + (resp.delta < 0 ? "delta-good" : resp.delta > 0 ? "delta-bad" : "");
    const reductionEl = document.getElementById("whatif-reduction");
    reductionEl.textContent = (-resp.delta).toFixed(2) + "°C";
    reductionEl.className = "kpi-value " + (resp.delta < 0 ? "delta-good" : resp.delta > 0 ? "delta-bad" : "");

    if (currentSnapshot) {
      const conversionFraction = greenConversion / 100;
      const modifiedValues = {
        tree: Math.min(100, Math.max(0, currentSnapshot.tree_cover_pct_state + extraTree)),
        veg: Math.min(100, Math.max(0, currentSnapshot.vegetation_cover_pct_state + extraVeg)),
        impervious: Math.min(1, Math.max(0, currentSnapshot.impervious_ratio - conversionFraction)),
      };
      renderWhatIfSnapshot(currentSnapshot, modifiedValues);
      renderMiniMap(suburb, resp.modified);
      renderProportionBar(modifiedValues.tree, modifiedValues.veg, modifiedValues.impervious);
    }
  }

  function scheduleWhatIf() {
    clearTimeout(whatIfDebounce);
    whatIfDebounce = setTimeout(runWhatIf, 150);
  }

  [sliderTree, sliderVeg, sliderGreen].forEach((slider) => {
    slider.addEventListener("input", () => {
      valTree.textContent = sliderTree.value + "pp";
      valVeg.textContent = sliderVeg.value + "pp";
      valGreen.textContent = sliderGreen.value + "pp";
      scheduleWhatIf();
    });
  });

  async function loadWhatIfSuburb(suburb) {
    currentSnapshot = await fetch(`/api/snapshot/${encodeURIComponent(suburb)}`).then((r) => r.json());
    sliderTree.value = 0; sliderVeg.value = 0; sliderGreen.value = 0;
    valTree.textContent = "0pp"; valVeg.textContent = "0pp"; valGreen.textContent = "0pp";
    renderWhatIfSnapshot(currentSnapshot, { tree: currentSnapshot.tree_cover_pct_state, veg: currentSnapshot.vegetation_cover_pct_state, impervious: currentSnapshot.impervious_ratio });
    renderMiniMap(suburb, currentSnapshot.predicted_mean_uhi_2018);
    renderProportionBar(currentSnapshot.tree_cover_pct_state, currentSnapshot.vegetation_cover_pct_state, currentSnapshot.impervious_ratio);
    runWhatIf();
  }

  whatIfSuburbSelect.addEventListener("change", () => loadWhatIfSuburb(whatIfSuburbSelect.value));

  function ensureWhatIfLoaded() {
    if (whatIfLoaded) return;
    whatIfLoaded = true;
    loadWhatIfSuburb(whatIfSuburbSelect.value);
    fetch("/api/suburbs").then((r) => r.json()).then((data) => {
      const excluded = data.suburbs.length - data.complete_suburbs.length;
      if (excluded > 0) {
        document.getElementById("whatif-excluded-note").textContent =
          `(${excluded} suburbs omitted from this list: incomplete vegetation/impervious data.)`;
      }
    });
  }
})();
