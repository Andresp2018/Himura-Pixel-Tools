// Himura Pixel Tools â€” desktop UI controller.
// Talks only to the local FastAPI backend on 127.0.0.1.

const API = "";  // same origin

async function api(method, path, body) {
  const opt = { method, headers: {} };
  const tok = window.HIMURA_TOKEN;
  if (tok) opt.headers["Authorization"] = `Bearer ${tok}`;
  if (body !== undefined) {
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(API + path, opt);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`HTTP ${r.status}: ${t}`);
  }
  return r.json();
}

function $(id) { return document.getElementById(id); }
function setStatus(id, text, isErr) {
  const el = $(id); if (!el) return;
  el.textContent = text || "";
  el.style.color = isErr ? "var(--danger)" : "var(--muted)";
}
function applySavedOutputOption(req, checkboxId) {
  const el = $(checkboxId);
  if (el) req.exclude_from_saved_outputs = !el.checked;
  return req;
}

function parseSize(sel, wId, hId) {
  const v = $(sel).value;
  if (v === "custom") {
    $(wId).parentElement.parentElement.classList.remove("hidden");
    return [parseInt($(wId).value) || 64, parseInt($(hId).value) || 64];
  }
  $(wId).parentElement.parentElement.classList.add("hidden");
  const [w, h] = v.split("x").map(Number);
  return [w, h];
}

// â”€â”€ tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
document.querySelectorAll("#tabs button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    document.querySelector(`[data-panel="${btn.dataset.tab}"]`).classList.add("active");
    if (btn.dataset.tab === "models") refreshModels();
    if (btn.dataset.tab === "objects") refreshObjects();
    if (btn.dataset.tab === "jobs") refreshJobs();
    if (btn.dataset.tab === "characters") refreshCharacters();
    if (btn.dataset.tab === "export") refreshExportAssets();
    if (btn.dataset.tab === "settings") refreshSettings();
    refreshLoraSelectors();
  });
});

// â”€â”€ LoRA selectors (base-aware) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function familyFromCompat(compat) {
  const s = (compat || []).join(" ").toLowerCase();
  if (s.includes("flux")) return "flux";
  if (s.includes("sd15")) return "sd15";
  if (s.includes("sdxl")) return "sdxl";
  return "other";
}
function isFlux2Name(value) {
  const blob = String(value || "").toLowerCase();
  return ["flux.2", "flux2", "flux-2", "klein"].some(t => blob.includes(t));
}
function loraBlob(l) {
  return [l.model_id, l.display_name, l.source_url, l.local_path, l.trigger, l.notes,
          (l.base_compatibility || []).join(" ")].join(" ").toLowerCase();
}
function loraFluxGeneration(l) {
  if (familyFromCompat(l.base_compatibility) !== "flux") return null;
  return isFlux2Name(loraBlob(l)) ? "flux2" : "flux1";
}
function baseFluxGeneration(baseModelId) {
  if (!String(baseModelId || "").toLowerCase().includes("flux")) return null;
  return isFlux2Name(baseModelId) ? "flux2" : "flux1";
}
function baseFamilyLabel(b) {
  const fam = baseFamily(b);
  if (fam !== "flux") return fam;
  return baseFluxGeneration((b && b.model_id) || "") || "flux";
}
function loraCompatibility(l, activeFam, activeFlux) {
  const f = familyFromCompat(l.base_compatibility);
  if (f !== "other" && activeFam !== "other" && f !== activeFam) {
    const label = f === "flux" ? "FLUX" : (f === "sdxl" ? "SDXL" : "SD 1.5");
    return { show: true, enabled: true, reason: `${label} LoRA: switch active base for effect` };
  }
  if (activeFlux && f === "flux") {
    const gen = loraFluxGeneration(l);
    if (gen !== activeFlux) {
      return {
        show: true,
        enabled: true,
        reason: gen === "flux1" ? "FLUX.1 LoRA: use a FLUX.1 base for effect"
                                : "FLUX.2/Klein LoRA: use a FLUX.2/Klein base for effect",
      };
    }
  }
  return { show: true, enabled: true, reason: "" };
}
async function refreshLoraSelectors() {
  if (!document.querySelector(".lora-select")) return;
  let data;
  try { data = await api("GET", "/api/loras"); } catch (e) { return; }
  const activeBase = data.active_base_model || "";
  const activeFam = baseFamily({ model_id: activeBase });
  const activeFlux = baseFluxGeneration(activeBase);
  const loras = (data.loras || [])
    .map(l => ({ ...l, _compat: loraCompatibility(l, activeFam, activeFlux) }))
    .filter(l => l._compat.show);
  const selectable = loras.filter(l => l._compat.enabled);
  document.querySelectorAll(".lora-select").forEach(sel => {
    const prev = sel.value;
    sel.innerHTML = '<option value="">(none / global default)</option>';
    if (!selectable.length && loras.length) {
      const info = document.createElement("option");
      info.value = "";
      info.disabled = true;
      info.textContent = activeFlux === "flux2"
        ? "No FLUX.2/Klein-compatible LoRA installed"
        : "No compatible LoRA installed";
      sel.appendChild(info);
    }
    loras.forEach(l => {
      const o = document.createElement("option");
      o.value = l._compat.enabled ? l.model_id : "";
      o.disabled = !l._compat.enabled;
      const trigger = l.trigger ? ` - ${l.trigger}` : "";
      o.textContent = l.display_name + trigger + (l._compat.reason ? ` (${l._compat.reason})` : "");
      sel.appendChild(o);
    });
    sel.value = selectable.some(l => l.model_id === prev) ? prev : "";
  });
}

// â”€â”€ health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function refreshHealth() {
  try {
    const h = await api("GET", "/api/health");
    $("version").textContent = `v${h.version}`;
    $("device").textContent = h.device || "cpu";
    $("loaded-model").textContent = h.model_loaded || "no model";
  } catch (e) {
    setStatus("g-status", "backend offline: " + e.message, true);
  }
}
refreshHealth();
refreshLoraSelectors();
setInterval(refreshHealth, 5000);

// â”€â”€ helper: run a job and poll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function runJob(path, req, statusId, onDone) {
  setStatus(statusId, "Queuedâ€¦");
  let job;
  try {
    job = await api("POST", path, req);
  } catch (e) {
    setStatus(statusId, e.message, true);
    return null;
  }
  const jobId = job.job_id;
  // poll
  for (let i = 0; i < 1200; i++) {
    await new Promise(r => setTimeout(r, 800));
    try {
      const st = await api("GET", `/api/jobs/${jobId}`);
      setStatus(statusId, `${st.status} ${(st.progress * 100).toFixed(0)}%`);
      if (["succeeded", "failed", "cancelled", "needs_review"].includes(st.status)) {
        if (st.status === "failed") {
          setStatus(statusId, "Failed: " + (st.error || ""), true);
        } else if (st.status === "needs_review") {
          setStatus(statusId, "Done with warnings: " + (st.warnings || []).join("; "));
        } else {
          setStatus(statusId, "Done");
        }
        if (onDone) onDone(st);
        return st;
      }
    } catch (e) {
      setStatus(statusId, "poll error: " + e.message, true);
      return null;
    }
  }
  return null;
}

function fileUrl(p) {
  if (!p) return null;
  // map absolute project path to /api/files/... if possible
  const m = p.replace(/\\/g, "/").match(/projects\/(.*)$/);
  if (!m) return null;
  // <img> tags can't send an Authorization header, so authenticate via the
  // token query param (accepted by require_token's fallback).
  const tok = window.HIMURA_TOKEN ? `?token=${encodeURIComponent(window.HIMURA_TOKEN)}` : "";
  return `/api/files/${m[1]}${tok}`;
}
function showImage(parentId, url, label) {
  const el = $(parentId);
  if (!url) { return; }
  const img = document.createElement("img");
  img.src = url; img.alt = label || "";
  img.onerror = () => img.remove();
  el.appendChild(img);
}

function baseName(p) {
  return (p || "").replace(/\\/g, "/").split("/").pop();
}

// Render a set of images into a preview area. `items` is an array of
// { path | url, label } objects; entries without a resolvable file are skipped.
function renderImages(parentId, items, emptyText) {
  const el = $(parentId);
  if (!el) return;
  el.innerHTML = "";
  const resolved = (items || [])
    .map(it => ({ url: it.url || fileUrl(it.path), label: it.label }))
    .filter(it => it.url);
  if (resolved.length === 0) {
    el.innerHTML = `<span class="muted">${emptyText || "No preview available."}</span>`;
    return;
  }
  resolved.forEach(it => {
    const fig = document.createElement("figure");
    fig.className = "thumb";
    const a = document.createElement("a");
    a.href = it.url; a.target = "_blank"; a.rel = "noopener";
    const img = document.createElement("img");
    img.src = it.url; img.alt = it.label || "";
    img.onerror = () => fig.remove();
    a.appendChild(img);
    fig.appendChild(a);
    if (it.label) {
      const cap = document.createElement("figcaption");
      cap.textContent = it.label;
      fig.appendChild(cap);
    }
    el.appendChild(fig);
  });
}

// Build a row of download links for every produced file, preferring the
// consolidated copies in final_output/. Returns an element (possibly empty).
function outputLinks(outputs) {
  const wrap = document.createElement("div");
  wrap.className = "out-links";
  if (!outputs) return wrap;
  const seen = new Set();
  const files = (outputs.final_files && outputs.final_files.length)
    ? outputs.final_files
    : (outputs.files || []);
  files.forEach(path => {
    const url = fileUrl(path);
    if (!url) return;
    const key = (path || "").toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    const a = document.createElement("a");
    a.href = url; a.target = "_blank"; a.rel = "noopener";
    a.textContent = baseName(path);
    a.title = path;
    wrap.appendChild(a);
  });
  return wrap;
}

// "Open folder" button that reveals an output folder in the OS file browser.
function revealButton(folder, label) {
  if (!folder) return null;
  const btn = document.createElement("button");
  btn.className = "btn-mini";
  btn.textContent = label || "Open folder";
  btn.title = folder;
  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      await api("POST", "/api/reveal", { path: folder });
    } catch (err) {
      alert("Could not open folder: " + err.message);
    }
  });
  return btn;
}

// Fill a "<id>-meta" element with download links + the final output folder.
function showOutputMeta(metaId, st, extraText) {
  const meta = $(metaId);
  if (!meta) return;
  meta.innerHTML = "";
  if (extraText) {
    const info = document.createElement("div");
    info.className = "small";
    info.textContent = extraText;
    meta.appendChild(info);
  }
  const outputs = (st && st.outputs) || {};
  const links = outputLinks(outputs);
  if (links.childNodes.length) meta.appendChild(links);
  if (st && st.saved_outputs_excluded) {
    const excluded = document.createElement("div");
    excluded.className = "final-path";
    excluded.textContent = "Excluded from final saved outputs";
    meta.appendChild(excluded);
  }
  const folder = outputs.final_output_folder;
  if (folder) {
    const row = document.createElement("div");
    row.className = "final-path";
    const span = document.createElement("span");
    span.textContent = folder;
    row.appendChild(span);
    const btn = revealButton(folder);
    if (btn) row.appendChild(btn);
    meta.appendChild(row);
  }
}

// â”€â”€ generate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$("g-size").addEventListener("change", () => parseSize("g-size", "g-w", "g-h"));

$("g-generate").addEventListener("click", async () => {
  const [w, h] = parseSize("g-size", "g-w", "g-h");
  const req = {
    asset_type: $("g-type").value,
    prompt: $("g-prompt").value,
    target_size: { width: w, height: h },
    transparent: $("g-transparent").checked,
    style_profile_id: $("g-style").value || null,
    palette_limit: parseInt($("g-colors").value) || 24,
    output_root: $("g-output").value || null,
    seed: parseInt($("g-seed").value) >= 0 ? parseInt($("g-seed").value) : null,
    negative_prompt: $("g-negative").value,
    quality_preset: $("g-fast").checked ? "fast" : "quality",
    palette_preset: $("g-palette").value || null,
    dither: $("g-dither").value || "none",
    protect_extremes: $("g-protect").checked,
    lora_id: ($("g-lora") || {}).value || null,
    generate_preview_scale: [4, 8],
  };
  applySavedOutputOption(req, "g-save-final");
  $("g-preview").innerHTML = "";
  $("g-meta").textContent = "";
  const st = await runJob("/api/jobs/generate", req, "g-status", (s) => {
    const o = s.outputs;
    renderImages("g-preview", [
      { path: o.preview_png_8x || o.preview_png, label: "preview" },
      { path: o.production_png, label: "production (1Ã—)" },
    ], "Generation finished but no image was produced.");
    showOutputMeta("g-meta", s,
      `seed ${s.seed ?? "â€”"} Â· model ${s.model_profile_id || "â€”"}`);
  });
  if (st && st.status === "succeeded") refreshHealth();
});

// â”€â”€ characters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$("c-create").addEventListener("click", async () => {
  const [w, h] = $("c-size").value.split("x").map(Number);
  const req = {
    name: $("c-name").value,
    description: $("c-desc").value,
    width: w, height: h,
    directions: parseInt($("c-dir").value),
    seed: parseInt($("c-seed").value) >= 0 ? parseInt($("c-seed").value) : null,
    lora_id: ($("c-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "c-save-final");
  $("c-preview").innerHTML = "";
  $("c-meta").textContent = "";
  await runJob("/api/jobs/create-character", req, "c-status", async (s) => {
    const o = s.outputs;
    renderImages("c-preview", [
      { path: o.preview_png_8x || o.preview_png, label: "canonical reference" },
      { path: o.production_png, label: "production (1Ã—)" },
    ], "Character created but no reference image was produced.");
    showOutputMeta("c-meta", s);
    await refreshCharacters();
  });
});

async function refreshCharacters() {
  try {
    const list = await api("GET", "/api/characters");
    $("c-list").innerHTML = "";
    ["c-pick", "a-char", "p-pick"].forEach(id => { if ($(id)) $(id).innerHTML = ""; });
    list.forEach(c => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${c.name}</span><span class="tag">${c.character_id}</span>`;
      $("c-list").appendChild(li);
      ["c-pick", "a-char", "p-pick"].forEach(id => {
        if (!$(id)) return;
        const o = document.createElement("option");
        o.value = c.character_id; o.textContent = c.name;
        $(id).appendChild(o);
      });
    });
  } catch (e) { /* ignore */ }
}

$("c-turnaround").addEventListener("click", async () => {
  if (!$("c-pick").value) {
    setStatus("c-turn-status", "Create or select a character first.", true);
    return;
  }
  const req = {
    character_profile_id: $("c-pick").value,
    directions: parseInt($("c-tdir").value),
    lora_id: ($("c-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "c-save-final");
  $("c-turn-grid").innerHTML = "";
  $("c-turn-meta").textContent = "";
  await runJob("/api/jobs/turnaround", req, "c-turn-status", (s) => {
    const dirs = s.outputs.turnaround_pngs || {};
    const items = Object.keys(dirs).map(k => ({ path: dirs[k], label: k }));
    renderImages("c-turn-grid", items, "Turnaround finished but no frames were produced.");
    showOutputMeta("c-turn-meta", s);
    refreshHealth();
  });
});

// â”€â”€ portrait â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$("p-generate").addEventListener("click", async () => {
  if (!$("p-pick").value) {
    setStatus("p-status", "Create or select a character first.", true);
    return;
  }
  const [w, h] = $("p-size").value.split("x").map(Number);
  const req = {
    character_profile_id: $("p-pick").value,
    width: w, height: h,
    palette_limit: parseInt($("p-colors").value) || 32,
    expression: $("p-expr").value,
    transparent: $("p-transparent").checked,
    seed: parseInt($("p-seed").value) >= 0 ? parseInt($("p-seed").value) : null,
    lora_id: ($("c-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "c-save-final");
  $("p-preview").innerHTML = "";
  $("p-meta").textContent = "";
  await runJob("/api/jobs/portrait", req, "p-status", (s) => {
    const o = s.outputs;
    renderImages("p-preview", [
      { path: o.preview_png_8x || o.preview_png, label: "portrait" },
      { path: o.production_png, label: "production (1Ã—)" },
    ], "Portrait finished but no image was produced.");
    showOutputMeta("p-meta", s);
    refreshHealth();
  });
});

// â”€â”€ animation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$("a-generate").addEventListener("click", async () => {
  if (!$("a-char").value) {
    setStatus("a-status", "Create or select a character first.", true);
    return;
  }
  const req = {
    character_profile_id: $("a-char").value,
    animation: $("a-anim").value,
    directions: parseInt($("a-dir").value),
    lora_id: ($("a-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "a-save-final");
  $("a-preview").innerHTML = "";
  $("a-meta").textContent = "";
  await runJob("/api/jobs/animate-character", req, "a-status", (s) => {
    const o = s.outputs;
    renderImages("a-preview", [
      { path: o.gif_preview, label: "animated preview" },
      { path: o.webp_preview, label: "webp" },
      { path: o.sprite_sheet_png, label: "sprite sheet" },
    ], "Animation finished but no preview was produced.");
    showOutputMeta("a-meta", s);
    refreshHealth();
  });
});

// â”€â”€ tileset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$("t-create").addEventListener("click", async () => {
  const req = {
    description: $("t-desc").value,
    tile_width: parseInt($("t-w").value),
    tile_height: parseInt($("t-h").value),
    tileset_type: $("t-type").value,
    tile_count: parseInt($("t-count").value),
    lora_id: ($("t-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "t-save-final");
  $("t-preview").innerHTML = "";
  $("t-meta").textContent = "";
  await runJob("/api/jobs/tileset", req, "t-status", (s) => {
    const o = s.outputs;
    renderImages("t-preview", [
      { path: o.sprite_sheet_png, label: "tileset sheet" },
    ], "Tileset finished but no sheet was produced.");
    showOutputMeta("t-meta", s);
    refreshHealth();
  });
});

// Batch recipes
if ($("r-run")) $("r-run").addEventListener("click", async () => {
  const req = {
    recipe: $("r-recipe").value,
    theme: $("r-theme").value,
    count: parseInt($("r-count").value) || 6,
    size: parseInt($("r-size").value) || 64,
    directions: parseInt($("r-dir").value) || 4,
    animations: $("r-anims").value.split(",").map(x => x.trim()).filter(Boolean),
    seed: parseInt($("r-seed").value) >= 0 ? parseInt($("r-seed").value) : null,
    lora_id: ($("r-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "r-save-final");
  $("r-preview").innerHTML = "";
  $("r-meta").textContent = "";
  await runJob("/api/jobs/batch-recipe", req, "r-status", (s) => {
    renderJobOutputs("r-preview", "r-meta", s, "Recipe finished but no preview was produced.");
    refreshHealth();
  });
});

async function refreshObjects() {
  const sel = $("o-state-pick");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '<option value="">(upload image or choose object)</option>';
  try {
    const objects = await api("GET", "/api/objects");
    (objects || []).slice().reverse().forEach(obj => {
      const o = document.createElement("option");
      o.value = obj.asset_id;
      const label = obj.prompt ? obj.prompt.slice(0, 44) : obj.asset_type;
      o.textContent = `${label} (${obj.asset_id})`;
      sel.appendChild(o);
    });
    if (prev) sel.value = prev;
  } catch (e) {
    // Keep upload path available if the object list cannot load.
  }
}

function readFileAsDataURL(input) {
  const file = input && input.files && input.files[0];
  if (!file) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read image."));
    reader.readAsDataURL(file);
  });
}
// PixelLab-style object workflows
function renderJobOutputs(previewId, metaId, s, emptyText) {
  const o = s.outputs || {};
  const dirs = o.turnaround_pngs || {};
  const items = [];
  const seen = new Set();
  function add(path, label) {
    if (!path || !/\.(png|gif|webp|jpe?g)$/i.test(path)) return;
    const key = String(path).toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    items.push({ path, label });
  }
  Object.keys(dirs).forEach(k => add(dirs[k], k));
  add(o.gif_preview, "animated preview");
  add(o.webp_preview, "webp");
  add(o.preview_png_8x || o.preview_png, "preview");
  add(o.sprite_sheet_png, "sprite sheet");
  add(o.production_png, "production");
  Object.entries(o.named_files || {}).forEach(([k, v]) => add(v, k));
  (o.files || []).forEach((v) => add(v, baseName(v)));
  renderImages(previewId, items, emptyText || "Job finished but no preview was produced.");
  showOutputMeta(metaId, s, `job ${s.job_id}`);
}

if ($("o-create")) $("o-create").addEventListener("click", async () => {
  const req = {
    description: $("o-desc").value,
    size: parseInt($("o-size").value) || 64,
    view: $("o-view").value,
    seed: parseInt($("o-seed").value) >= 0 ? parseInt($("o-seed").value) : null,
    lora_id: ($("o-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "o-save-final");
  $("o-preview").innerHTML = ""; $("o-meta").textContent = "";
  await runJob("/api/jobs/object", req, "o-status", async (s) => { renderJobOutputs("o-preview", "o-meta", s); await refreshObjects(); });
});

if ($("o-rot")) $("o-rot").addEventListener("click", async () => {
  const req = {
    description: $("o-desc").value,
    size: parseInt($("o-size").value) || 64,
    view: ["low top-down", "high top-down", "side"].includes($("o-view").value) ? $("o-view").value : "low top-down",
    seed: parseInt($("o-seed").value) >= 0 ? parseInt($("o-seed").value) : null,
    lora_id: ($("o-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "o-save-final");
  $("o-preview").innerHTML = ""; $("o-meta").textContent = "";
  await runJob("/api/jobs/object-8dir", req, "o-status", async (s) => { renderJobOutputs("o-preview", "o-meta", s); await refreshObjects(); });
});

if ($("o-state")) $("o-state").addEventListener("click", async () => {
  const picked = ($("o-state-pick") || {}).value || "";
  let uploaded = null;
  try {
    uploaded = await readFileAsDataURL($("o-state-upload"));
  } catch (e) {
    setStatus("o-status", e.message || String(e), true);
    return;
  }
  if (!picked && !uploaded) {
    setStatus("o-status", "Choose a saved object or upload an image first.", true);
    return;
  }
  const req = {
    object_id: picked || null,
    source_image: uploaded,
    edit_description: $("o-state-edit").value,
    seed: parseInt($("o-seed").value) >= 0 ? parseInt($("o-seed").value) : null,
    lora_id: ($("o-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "o-save-final");
  $("o-preview").innerHTML = ""; $("o-meta").textContent = "";
  await runJob("/api/jobs/object-state", req, "o-status", async (s) => {
    renderJobOutputs("o-preview", "o-meta", s);
    await refreshObjects();
  });
});

if ($("mo-create")) $("mo-create").addEventListener("click", async () => {
  const req = {
    description: $("mo-desc").value,
    width: parseInt($("mo-w").value) || 64,
    height: parseInt($("mo-h").value) || 64,
    view: $("mo-view").value,
    lora_id: ($("o-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "o-save-final");
  $("o-preview").innerHTML = ""; $("o-meta").textContent = "";
  await runJob("/api/jobs/map-object", req, "o-status", async (s) => { renderJobOutputs("o-preview", "o-meta", s); await refreshObjects(); });
});

if ($("u-create")) $("u-create").addEventListener("click", async () => {
  const req = {
    description: $("u-desc").value,
    name: $("u-name").value,
    width: parseInt($("u-w").value) || 256,
    height: parseInt($("u-h").value) || 128,
    color_palette: $("u-palette").value || null,
    elements: $("u-elements").value.split(",").map(x => x.trim()).filter(Boolean),
    no_background: $("u-bg").checked,
    lora_id: ($("u-lora") || {}).value || null,
  };
  applySavedOutputOption(req, "u-save-final");
  $("u-preview").innerHTML = ""; $("u-meta").textContent = "";
  await runJob("/api/jobs/ui-asset", req, "u-status", (s) => renderJobOutputs("u-preview", "u-meta", s));
});

async function runAdvancedTile(path, req) {
  req.lora_id = ($("adv-lora") || {}).value || null;
  applySavedOutputOption(req, "adv-save-final");
  $("adv-preview").innerHTML = ""; $("adv-meta").textContent = "";
  await runJob(path, req, "adv-status", (s) => renderJobOutputs("adv-preview", "adv-meta", s));
}

if ($("td-create")) $("td-create").addEventListener("click", () => runAdvancedTile("/api/jobs/topdown-tileset", {
  lower_description: $("td-lower").value,
  upper_description: $("td-upper").value,
  transition_description: $("td-trans").value,
  tile_size: { width: parseInt($("adv-size").value) || 32, height: parseInt($("adv-size").value) || 32 },
}));

if ($("sd-create")) $("sd-create").addEventListener("click", () => runAdvancedTile("/api/jobs/sidescroller-tileset", {
  lower_description: $("sd-lower").value,
  transition_description: $("sd-trans").value,
  tile_size: { width: parseInt($("adv-size").value) || 32, height: parseInt($("adv-size").value) || 32 },
}));

if ($("iso-create")) $("iso-create").addEventListener("click", () => runAdvancedTile("/api/jobs/isometric-tile", {
  description: $("iso-desc").value,
  size: parseInt($("adv-size").value) || 32,
  tile_shape: "thick tile",
}));

if ($("pro-create")) $("pro-create").addEventListener("click", () => runAdvancedTile("/api/jobs/tiles-pro", {
  description: $("iso-desc").value,
  tile_size: parseInt($("adv-size").value) || 32,
  tile_type: "square_topdown",
  tile_view: "top-down",
}));

// â”€â”€ true-pixel snapper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$("sn-run").addEventListener("click", async () => {
  const f = $("sn-file").files[0];
  if (!f) { setStatus("sn-status", "Choose an image file first.", true); return; }
  setStatus("sn-status", "Reading imageâ€¦");
  const dataUrl = await new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(f);
  });
  const req = { image: dataUrl };
  const px = parseFloat($("sn-px").value);
  if (px > 0) req.pixel_size = px;
  const kc = parseInt($("sn-colors").value);
  if (kc > 1) req.k_colors = kc;
  $("sn-preview").innerHTML = "";
  $("sn-meta").textContent = "";
  setStatus("sn-status", "Snappingâ€¦");
  try {
    const r = await api("POST", "/api/snap", req);
    renderImages("sn-preview", [
      { path: r.preview_png, label: "snapped (8Ã—)" },
      { path: r.production_png, label: "true pixels (1Ã—)" },
    ], "No output produced.");
    const meta = $("sn-meta");
    meta.innerHTML = "";
    const info = document.createElement("div");
    info.className = "small";
    info.textContent = `detected pixel size ${r.detected_pixel_size.x}Ã—${r.detected_pixel_size.y} Â· `
      + `output ${r.output_size.width}Ã—${r.output_size.height} Â· ${r.colors} colors`;
    meta.appendChild(info);
    const links = outputLinks({ files: [r.production_png, r.preview_png] });
    if (links.childNodes.length) meta.appendChild(links);
    setStatus("sn-status", "Done");
  } catch (e) {
    setStatus("sn-status", e.message, true);
  }
});

// â”€â”€ export / build pack â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function refreshExportAssets() {
  try {
    const assets = await api("GET", "/api/assets");
    const ul = $("e-assets");
    ul.innerHTML = "";
    if (!assets.length) {
      ul.innerHTML = `<li><span class="muted">No assets yet â€” generate something, then refresh.</span></li>`;
      return;
    }
    assets.slice().reverse().forEach(a => {
      const li = document.createElement("li");
      const url = fileUrl(a.preview_path || a.production_path);
      const thumb = url ? `<img src="${url}" alt="" style="height:32px;width:auto;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px;" onerror="this.remove()"/>` : "";
      li.innerHTML = `<label class="check" style="margin:0;flex:1;">
          <input type="checkbox" class="e-asset" data-id="${a.asset_id}" checked />
          ${thumb}
          <span>${a.asset_type} <span class="tag">${a.asset_id}</span></span>
        </label>`;
      ul.appendChild(li);
    });
  } catch (e) { setStatus("e-status", "could not load assets: " + e.message, true); }
}

$("e-refresh").addEventListener("click", refreshExportAssets);
$("e-build").addEventListener("click", async () => {
  const ids = Array.from(document.querySelectorAll(".e-asset:checked")).map(c => c.dataset.id);
  if (!ids.length) { setStatus("e-status", "Select at least one asset.", true); return; }
  $("e-result").innerHTML = "";
  const req = { asset_ids: ids, engine: $("e-engine").value };
  applySavedOutputOption(req, "e-save-final");
  await runJob("/api/export/pack", req, "e-status", (s) => {
    const zip = s.outputs.zip_path;
    const res = $("e-result");
    res.innerHTML = "";
    const links = outputLinks(s.outputs);
    if (links.childNodes.length) res.appendChild(links);
    const folder = s.outputs.final_output_folder;
    if (folder) {
      const row = document.createElement("div");
      row.className = "final-path";
      const span = document.createElement("span");
      span.textContent = zip || folder;
      row.appendChild(span);
      const btn = revealButton(folder);
      if (btn) row.appendChild(btn);
      res.appendChild(row);
    }
  });
});

// â”€â”€ models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
let _activePoll = null;

function baseFamily(b) {
  const compat = (b.base_compatibility || []).join(" ").toLowerCase();
  const blob = (compat + " " + (b.model_id || "")).toLowerCase();
  if (blob.includes("flux")) return "flux";
  if (compat.includes("sd15") || blob.includes("sd1") || blob.includes("1-5")) return "sd15";
  if (blob.includes("xl") || blob.includes("sdxl")) return "sdxl";
  return "other";
}

async function refreshModels() {
  try {
    const m = await api("GET", "/api/models");
    if (m.models_root) $("m-root").textContent = m.models_root;
    $("m-vram").textContent = JSON.stringify(m.vram, null, 2);

    // active base selector â€” grouped by family (SDXL / FLUX / SD1.5) with
    // <optgroup> separators so the two stacks are visually distinct.
    const sel = $("m-active-base");
    const prev = sel.value;
    sel.innerHTML = "";
    const bases = (m.installed || []).filter(x => x.type === "base");
    if (bases.length === 0) {
      const o = document.createElement("option");
      o.value = ""; o.textContent = "(no base model installed)";
      sel.appendChild(o);
    } else {
      const groups = { sdxl: [], flux1: [], flux2: [], flux: [], sd15: [], other: [] };
      bases.forEach(b => groups[baseFamilyLabel(b)].push(b));
      const order = [["sdxl", "SDXL models"], ["flux1", "FLUX.1 models (LoRA-compatible)"],
                     ["flux2", "FLUX.2/Klein models"], ["flux", "FLUX models"],
                     ["sd15", "SD 1.5 models"], ["other", "Other base models"]];
      order.forEach(([key, label]) => {
        if (!groups[key].length) return;
        const og = document.createElement("optgroup");
        og.label = label;
        groups[key].forEach(b => {
          const o = document.createElement("option");
          o.value = b.model_id; o.textContent = `${b.display_name} (${b.model_id})`;
          og.appendChild(o);
        });
        sel.appendChild(og);
      });
    }
    if (m.active_base_model) {
      sel.value = m.active_base_model;
      // if the stored active id isn't in the list (e.g. HF default), add it
      if (!sel.value) {
        const o = document.createElement("option");
        o.value = m.active_base_model; o.textContent = m.active_base_model + " (not installed)";
        sel.appendChild(o); sel.value = m.active_base_model;
      }
    } else if (prev) {
      sel.value = prev;
    }

    // installed list (with remove button)
    $("m-installed").innerHTML = "";
    (m.installed || []).forEach(mi => {
      const li = document.createElement("li");
      const isActive = mi.type === "base" && mi.model_id === m.active_base_model;
      li.innerHTML = `<span>${mi.display_name} <span class="tag">[${mi.type}]</span>
        ${isActive ? '<span class="ok">â— active</span>' : ''}</span>
        <span>
          <span class="tag">${mi.model_id}</span>
          <button class="m-remove" data-id="${mi.model_id}" style="padding:2px 8px;font-size:11px;">remove</button>
        </span>`;
      $("m-installed").appendChild(li);
    });
    document.querySelectorAll(".m-remove").forEach(b => {
      b.addEventListener("click", async () => {
        if (!confirm("Remove this model from the local store?")) return;
        await api("POST", "/api/models/remove", { model_id: b.dataset.id });
        refreshModels();
      });
    });

    // recommended list (grouped SDXL / FLUX / shared, clickable to download)
    $("m-recommended").innerHTML = "";
    const recGroups = { sdxl: [], flux: [], shared: [] };
    (m.recommendations || []).forEach(r => {
      const fam = baseFamily(r);
      recGroups[fam === "flux" ? "flux" : (fam === "sdxl" ? "sdxl" : "shared")].push(r);
    });
    const recOrder = [["sdxl", "SDXL stack"], ["flux", "FLUX stack"],
                      ["shared", "Shared / other"]];
    recOrder.forEach(([key, label]) => {
      if (!recGroups[key].length) return;
      const head = document.createElement("li");
      head.className = "list-sep";
      head.innerHTML = `<strong>â”€â”€ ${label} â”€â”€</strong>`;
      $("m-recommended").appendChild(head);
      recGroups[key].forEach(r => {
        const li = document.createElement("li");
        const trig = r.trigger ? ` <span class="tag">trigger: ${r.trigger}</span>` : "";
        const btn = r.installed
          ? `<span class="ok">installed</span>`
          : `<button class="m-dl-one primary" data-id="${r.id}" style="padding:2px 10px;font-size:11px;">download</button>`;
        li.innerHTML = `<span>${r.id} <span class="tag">[${r.type}]</span> <span class="tag">(${r.role})</span>${trig}</span><span>${btn}</span>`;
        $("m-recommended").appendChild(li);
      });
    });
    document.querySelectorAll(".m-dl-one").forEach(b => {
      b.addEventListener("click", () => downloadOne(b.dataset.id));
    });
  } catch (e) {
    setStatus("g-status", "models error: " + e.message, true);
  }
}

async function downloadOne(source, forcedType) {
  const mtype = forcedType || $("m-install-type").value || null;
  setStatus("m-download-status", `Downloading ${source} â€¦`);
  try {
    await api("POST", "/api/models/download", { source, model_type: mtype });
  } catch (e) {
    setStatus("m-download-status", e.message, true);
    return;
  }
  pollDownload(source);
}

function pollDownload(source) {
  if (_activePoll) clearInterval(_activePoll);
  _activePoll = setInterval(async () => {
    try {
      const st = await api("GET", `/api/models/download-status?source=${encodeURIComponent(source)}`);
      setStatus("m-download-status", `${st.status}: ${st.message} (${st.pct}%)`,
        st.status === "error");
      if (st.status === "done" || st.status === "error") {
        clearInterval(_activePoll); _activePoll = null;
        refreshModels();
      }
    } catch (e) { /* keep polling */ }
  }, 1500);
}

$("m-refresh").addEventListener("click", refreshModels);
$("m-download-all").addEventListener("click", async () => {
  setStatus("m-download-status", "Downloading recommended set in backgroundâ€¦");
  await api("POST", "/api/models/download-all");
  pollDownload("__all__");
});
if ($("m-download-ideogram")) $("m-download-ideogram").addEventListener("click", () => {
  downloadOne("https://huggingface.co/leejet/ideogram-4-GGUF/tree/main", "base");
});
$("m-install").addEventListener("click", () => {
  const src = $("m-install-id").value.trim();
  if (!src) { setStatus("m-download-status", "Enter an HF id or URL first.", true); return; }
  downloadOne(src);
});
$("m-set-active").addEventListener("click", async () => {
  const id = $("m-active-base").value;
  if (!id) { setStatus("m-active-status", "No base model selected.", true); return; }
  try {
    await api("POST", "/api/models/set-active", { model_id: id });
    setStatus("m-active-status", `Active base model set to ${id}. It will load on next generation.`);
    refreshModels();
    refreshLoraSelectors();
  } catch (e) { setStatus("m-active-status", e.message, true); }
});
$("m-unload").addEventListener("click", async () => {
  try { await api("POST", "/api/runtime/unload"); setStatus("m-active-status", "Unloaded."); refreshHealth(); }
  catch (e) { setStatus("m-active-status", e.message, true); }
});

// â”€â”€ jobs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function refreshGeneratedLists() {
  await Promise.allSettled([
    refreshJobs(),
    refreshCharacters(),
    refreshExportAssets(),
    refreshObjects(),
  ]);
}

async function excludeSavedOutput(jobId) {
  try {
    await api("POST", `/api/jobs/${jobId}/exclude-saved-output`);
    await refreshGeneratedLists();
  } catch (e) {
    alert("Could not exclude generation: " + e.message);
  }
}

async function refreshJobs() {
  try {
    const jobs = await api("GET", "/api/jobs");
    const tbody = $("j-table").querySelector("tbody");
    tbody.innerHTML = "";
    jobs.slice().reverse().forEach(j => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${j.job_id}</td><td>${j.job_type}</td>
        <td class="st-${j.status}">${j.status}</td>
        <td>${(j.progress * 100).toFixed(0)}%</td>`;
      const outTd = document.createElement("td");
      outTd.className = "small";
      const outputs = j.outputs || {};
      const links = outputLinks(outputs);
      if (links.childNodes.length) outTd.appendChild(links);
      const folder = outputs.final_output_folder;
      const row = document.createElement("div");
      row.className = "final-path";
      if (folder) {
        const btn = revealButton(folder);
        if (btn) row.appendChild(btn);
      }
      const excludeBtn = document.createElement("button");
      excludeBtn.className = "btn-mini danger";
      excludeBtn.textContent = "Exclude generation";
      excludeBtn.title = "Delete this generation's files, asset records, and job history entry";
      excludeBtn.addEventListener("click", () => excludeSavedOutput(j.job_id));
      row.appendChild(excludeBtn);
      outTd.appendChild(row);
      if (!outTd.childNodes.length) {
        outTd.textContent = j.error ? "" : "â€”";
      }
      tr.appendChild(outTd);
      tbody.appendChild(tr);
    });
  } catch (e) { /* ignore */ }
}
$("j-refresh").addEventListener("click", refreshJobs);
if ($("j-exclude-all")) $("j-exclude-all").addEventListener("click", async () => {
  try {
    await api("POST", "/api/jobs/exclude-saved-outputs");
    await refreshGeneratedLists();
  } catch (e) {
    alert("Could not exclude all generated outputs: " + e.message);
  }
});

// â”€â”€ settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function refreshSettings() {
  try {
    const cfg = await api("GET", "/api/config");
    const fields = $("s-fields");
    fields.innerHTML = "";
    const labels = {
      precision: "Precision", attention_slicing: "Attention slicing",
      vae_tiling: "VAE tiling",
      sequential_cpu_offload_when_low_vram: "CPU offload (low VRAM)",
      max_parallel_jobs: "Max parallel jobs",
      default_quality_preset: "Default preset",
      auto_download_models_at_startup: "Auto-download models at startup",
      mirror_outputs_to_final: "Save completed jobs to final outputs",
      mcp_http_enabled: "MCP HTTP enabled", mcp_require_token: "Require MCP token",
      last_base_model_id: "Default base model id",
      generation_provider: "Generation provider (local | gemini)",
      gemini_api_key: "Gemini API key (only if provider = gemini)",
      gemini_model: "Gemini model",
      civitai_api_key: "Civitai API key (optional, for gated downloads)",
      pixel_lora_id: "Default pixel-art LoRA id (global fallback)",
      pixel_lora_weight: "Default pixel-art LoRA weight",
    };
    Object.entries(labels).forEach(([k, lbl]) => {
      const wrap = document.createElement("div");
      wrap.className = "row";
      const val = cfg[k];
      const isBool = typeof val === "boolean";
      wrap.innerHTML = `<label>${lbl}${isBool
        ? `<input data-cfg="${k}" type="checkbox" ${val ? "checked" : ""}/>`
        : typeof val === "number"
          ? `<input data-cfg="${k}" type="number" step="any" value="${val}"/>`
          : `<input data-cfg="${k}" type="text" value="${val}"/>`}</label>`;
      fields.appendChild(wrap);
    });
    const save = document.createElement("button");
    save.className = "primary"; save.textContent = "Save settings";
    save.addEventListener("click", async () => {
      const out = {};
      fields.querySelectorAll("[data-cfg]").forEach(el => {
        const k = el.dataset.cfg;
        out[k] = el.type === "checkbox" ? el.checked : (el.type === "number" ? parseFloat(el.value) : el.value);
      });
      await api("POST", "/api/config", out);
      setStatus("g-status", "Settings saved.");
    });
    fields.appendChild(save);

    const tok = await api("GET", "/api/mcp-token");
    $("s-token").value = tok.token;
    $("s-mcp-url").textContent = tok.http_url;
    window.__mcp = { token: tok.token, url: tok.http_url,
                     stdio: tok.stdio_command || "himura-pixel-tools-mcp" };
    renderCliConfig();
  } catch (e) { /* ignore */ }
}

function mcpCliConfig(kind, m) {
  const T = m.token, U = m.url, S = m.stdio;
  const Sj = JSON.stringify(S);  // safely quoted for shell / json
  switch (kind) {
    case "claude_http":
      return `claude mcp add --transport http himura-pixel-tools ${U} --header "Authorization: Bearer ${T}"`;
    case "claude_stdio":
      return `claude mcp add --transport stdio himura-pixel-tools -- ${Sj} --transport stdio`;
    case "claude_desktop":
      return "// claude_desktop_config.json\n" + JSON.stringify(
        { mcpServers: { "himura-pixel-tools": { command: S, args: ["--transport", "stdio"] } } }, null, 2);
    case "codex_http":
      return [
        "# 1) set the token in your shell:",
        `#    export HIMURA_MCP_TOKEN="${T}"   (Windows: setx HIMURA_MCP_TOKEN "${T}")`,
        "# 2) ~/.codex/config.toml:",
        "[mcp_servers.himura_pixel_tools]",
        `url = "${U}"`,
        'bearer_token_env_var = "HIMURA_MCP_TOKEN"',
        "startup_timeout_sec = 20",
        "tool_timeout_sec = 600",
      ].join("\n");
    case "codex_stdio":
      return [
        "# ~/.codex/config.toml",
        "[mcp_servers.himura_pixel_tools]",
        `command = ${Sj}`,
        'args = ["--transport", "stdio"]',
        "startup_timeout_sec = 20",
        "tool_timeout_sec = 600",
      ].join("\n");
    case "antigravity":
      return "// .antigravity/mcp.json\n" + JSON.stringify(
        { mcpServers: { "himura-pixel-tools": {
          httpUrl: U, headers: { Authorization: `Bearer ${T}` }, timeout: 600000 } } }, null, 2);
    case "cursor":
      return "// .cursor/mcp.json\n" + JSON.stringify(
        { mcpServers: { "himura-pixel-tools": {
          url: U, headers: { Authorization: `Bearer ${T}` } } } }, null, 2);
    default: return "";
  }
}

function renderCliConfig() {
  if (!window.__mcp || !$("s-cli") || !$("s-cli-config")) return;
  $("s-cli-config").value = mcpCliConfig($("s-cli").value, window.__mcp);
}

document.addEventListener("change", (e) => {
  if (e.target && e.target.id === "s-cli") renderCliConfig();
});

$("s-copy-token").addEventListener("click", () => {
  $("s-token").select(); document.execCommand("copy");
});
$("s-copy-cli").addEventListener("click", () => {
  const el = $("s-cli-config");
  el.select();
  try { document.execCommand("copy"); } catch (_) {}
  if (navigator.clipboard) navigator.clipboard.writeText(el.value).catch(() => {});
  setStatus("g-status", "MCP setup copied to clipboard.");
});









