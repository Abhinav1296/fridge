/* SmartFridge Vision MVP — frontend logic (vanilla JS).
 *
 * Two input paths feed a single "analyze" flow:
 *   1. File upload (drag-and-drop + file picker)  -> sent as multipart/form-data.
 *   2. Live webcam capture (getUserMedia + canvas) -> sent as base64 JSON.
 * Results are rendered as cards with colour-coded freshness badges. Anything the
 * model couldn't identify is shown in a separate "Needs your input" section.
 */

"use strict";

// --- Element references -----------------------------------------------------
const els = {
  tabFridge: document.getElementById("tab-fridge"),
  tabUpload: document.getElementById("tab-upload"),
  tabWebcam: document.getElementById("tab-webcam"),
  tabHistory: document.getElementById("tab-history"),
  panelFridge: document.getElementById("panel-fridge"),
  panelUpload: document.getElementById("panel-upload"),
  panelWebcam: document.getElementById("panel-webcam"),
  panelHistory: document.getElementById("panel-history"),

  // Wrapper (display:contents) around the whole capture flow. Only the Upload and
  // Webcam tabs show it; Fridge and History hide it.
  captureExtras: document.getElementById("capture-extras"),

  // Fridge tab (current inventory = most recent scan).
  fridgeAsof: document.getElementById("fridge-asof"),
  fridgeLoading: document.getElementById("fridge-loading"),
  fridgeEmpty: document.getElementById("fridge-empty"),
  fridgeError: document.getElementById("fridge-error"),
  fridgeBody: document.getElementById("fridge-body"),
  fridgeSummary: document.getElementById("fridge-summary"),
  fridgeCallout: document.getElementById("fridge-callout"),
  fridgeToggle: document.getElementById("fridge-toggle"),
  fridgeGroups: document.getElementById("fridge-groups"),
  fridgeUnidentified: document.getElementById("fridge-unidentified"),
  refreshFridge: document.getElementById("refresh-fridge"),
  fridgeScanCta: document.getElementById("fridge-scan-cta"),

  // History tab elements.
  historyList: document.getElementById("history-list"),
  historyLoading: document.getElementById("history-loading"),
  historyEmpty: document.getElementById("history-empty"),
  historyError: document.getElementById("history-error"),
  refreshHistory: document.getElementById("refresh-history"),

  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),

  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  startCamera: document.getElementById("start-camera"),
  capture: document.getElementById("capture"),
  webcamHint: document.getElementById("webcam-hint"),

  previewSection: document.getElementById("preview-section"),
  preview: document.getElementById("preview"),
  analyze: document.getElementById("analyze"),
  clear: document.getElementById("clear"),

  loading: document.getElementById("loading"),
  error: document.getElementById("error"),

  resultsSection: document.getElementById("results-section"),
  cards: document.getElementById("cards"),
  itemCount: document.getElementById("item-count"),
  noItems: document.getElementById("no-items"),
  unidentifiedSection: document.getElementById("unidentified-section"),
  unidentifiedCards: document.getElementById("unidentified-cards"),
};

// --- Application state ------------------------------------------------------
// Exactly one image source is "pending" at a time. `source` selects how we send it.
const state = {
  source: null, // "file" | "camera" | null
  file: null, // File object (upload path)
  dataUrl: null, // data URL string (camera path)
  objectUrl: null, // object URL for the upload preview (revoked on clear)
  stream: null, // active MediaStream (camera)
};

const FRESHNESS_LABELS = {
  fresh: "Fresh",
  ripe: "Ripe",
  use_soon: "Use soon",
  spoiled: "Spoiled",
  unknown: "Unknown",
};

// Stable display order for the Fridge view's freshness summary.
const FRESHNESS_ORDER = ["fresh", "ripe", "use_soon", "spoiled", "unknown"];

// Known categories shown first (in this order) when grouping the Fridge view; any
// other category the model returns is appended alphabetically after these.
const CATEGORY_ORDER = ["fruit", "vegetable", "dairy", "beverage", "packaged", "other"];

// My Fridge shows its grouped item list expanded by default (it's the glance view), but
// the "Show all items" toggle lets you collapse it down to just the summary + callout.
let fridgeItemsOpen = true;
let fridgeHasUnidentified = false;

// --- Tab switching ----------------------------------------------------------
function activateTab(which) {
  const tabs = {
    fridge: [els.tabFridge, els.panelFridge],
    upload: [els.tabUpload, els.panelUpload],
    webcam: [els.tabWebcam, els.panelWebcam],
    history: [els.tabHistory, els.panelHistory],
  };

  for (const [name, [tab, panel]] of Object.entries(tabs)) {
    const active = name === which;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    panel.hidden = !active;
  }

  // The camera only belongs to the Webcam tab.
  if (which !== "webcam") {
    stopCamera();
  }

  // The capture flow (preview/loading/results) belongs only to the Upload and Webcam
  // tabs; hide it on Fridge/History. Reload data when entering a data-backed tab.
  els.captureExtras.hidden = which !== "upload" && which !== "webcam";
  if (which === "fridge") {
    loadFridge();
  } else if (which === "history") {
    loadHistory();
  }
}

els.tabFridge.addEventListener("click", () => activateTab("fridge"));
els.tabUpload.addEventListener("click", () => activateTab("upload"));
els.tabWebcam.addEventListener("click", () => activateTab("webcam"));
els.tabHistory.addEventListener("click", () => activateTab("history"));
els.refreshFridge.addEventListener("click", loadFridge);
els.refreshHistory.addEventListener("click", loadHistory);
els.fridgeScanCta.addEventListener("click", () => activateTab("upload"));
els.fridgeToggle.addEventListener("click", () => setFridgeItemsOpen(!fridgeItemsOpen));

// --- Upload: drag & drop + file picker --------------------------------------
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});

els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files && els.fileInput.files[0]) {
    handleFile(els.fileInput.files[0]);
  }
});

["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) {
    handleFile(file);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("That file isn't an image. Please choose a JPEG or PNG.");
    return;
  }
  clearResults();
  hideError();

  // Reset any previous source.
  releaseObjectUrl();
  state.source = "file";
  state.file = file;
  state.dataUrl = null;
  state.objectUrl = URL.createObjectURL(file);

  els.preview.src = state.objectUrl;
  els.previewSection.hidden = false;
}

// --- Webcam -----------------------------------------------------------------
els.startCamera.addEventListener("click", startCamera);
els.capture.addEventListener("click", captureFrame);

async function startCamera() {
  hideError();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError("This browser doesn't support webcam capture. Try the Upload tab instead.");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    els.video.srcObject = state.stream;
    els.capture.disabled = false;
    els.startCamera.textContent = "Restart camera";
    els.webcamHint.textContent = "Point the camera at your fridge/box contents, then press Capture.";
  } catch (err) {
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      showError("Camera permission was denied. Allow camera access, or use the Upload tab.");
    } else if (err && err.name === "NotFoundError") {
      showError("No camera was found on this device. Use the Upload tab instead.");
    } else {
      showError("Couldn't start the camera. Use the Upload tab instead.");
    }
  }
}

function captureFrame() {
  if (!state.stream) {
    showError("Start the camera first, then press Capture.");
    return;
  }
  const w = els.video.videoWidth;
  const h = els.video.videoHeight;
  if (!w || !h) {
    showError("The camera isn't ready yet. Give it a moment and try again.");
    return;
  }
  els.canvas.width = w;
  els.canvas.height = h;
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(els.video, 0, 0, w, h);

  clearResults();
  hideError();

  releaseObjectUrl();
  state.source = "camera";
  state.dataUrl = els.canvas.toDataURL("image/jpeg", 0.9);
  state.file = null;

  els.preview.src = state.dataUrl;
  els.previewSection.hidden = false;
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    els.video.srcObject = null;
    els.capture.disabled = true;
    els.startCamera.textContent = "Start camera";
  }
}

// --- Analyze / clear --------------------------------------------------------
els.analyze.addEventListener("click", analyze);
els.clear.addEventListener("click", clearAll);

async function analyze() {
  if (!state.source) {
    showError("Choose or capture an image first.");
    return;
  }

  hideError();
  clearResults();
  setLoading(true);

  try {
    const response = await sendForAnalysis();
    let payload;
    try {
      payload = await response.json();
    } catch (_e) {
      payload = null;
    }

    if (!response.ok) {
      const message =
        (payload && payload.error) ||
        `The server returned an error (HTTP ${response.status}).`;
      showError(message);
      return;
    }

    renderResults(payload || { items: [], unidentified: [] });
  } catch (_err) {
    showError("Couldn't reach the server. Check that the app is running and try again.");
  } finally {
    setLoading(false);
  }
}

function sendForAnalysis() {
  if (state.source === "file") {
    const form = new FormData();
    form.append("image", state.file);
    return fetch("/analyze", { method: "POST", body: form });
  }
  // camera path
  return fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: state.dataUrl }),
  });
}

function clearAll() {
  clearResults();
  hideError();
  releaseObjectUrl();
  state.source = null;
  state.file = null;
  state.dataUrl = null;
  els.preview.removeAttribute("src");
  els.previewSection.hidden = true;
  els.fileInput.value = "";
}

// --- Rendering --------------------------------------------------------------
function renderResults(data) {
  const items = Array.isArray(data.items) ? data.items : [];
  const unidentified = Array.isArray(data.unidentified) ? data.unidentified : [];

  els.cards.innerHTML = "";
  els.unidentifiedCards.innerHTML = "";

  els.itemCount.textContent = `${items.length} ${items.length === 1 ? "item" : "items"}`;
  els.noItems.hidden = items.length !== 0;

  for (const item of items) {
    els.cards.appendChild(buildItemCard(item));
  }

  if (unidentified.length > 0) {
    for (const entry of unidentified) {
      els.unidentifiedCards.appendChild(buildUnidentifiedCard(entry));
    }
    els.unidentifiedSection.hidden = false;
  } else {
    els.unidentifiedSection.hidden = true;
  }

  els.resultsSection.hidden = false;
}

function buildItemCard(item) {
  const freshness = FRESHNESS_LABELS[item.freshness] ? item.freshness : "unknown";
  const confidence = item.freshness_confidence || "low";

  const card = document.createElement("div");
  card.className = "card";

  const head = document.createElement("div");
  head.className = "card-head";
  const name = document.createElement("h3");
  name.className = "card-name";
  name.textContent = item.name || "Unknown item";
  const count = document.createElement("span");
  count.className = "card-count";
  count.textContent = `×${Number.isFinite(item.count) ? item.count : 1}`;
  head.append(name, count);

  const meta = document.createElement("div");
  meta.className = "card-meta";

  const badge = document.createElement("span");
  badge.className = `badge badge-${freshness}${confidence === "low" ? " conf-low" : ""}`;
  badge.textContent = FRESHNESS_LABELS[freshness];
  meta.appendChild(badge);

  if (item.category) {
    const cat = document.createElement("span");
    cat.className = "chip";
    cat.textContent = item.category;
    meta.appendChild(cat);
  }

  const conf = document.createElement("span");
  conf.className = "chip";
  conf.textContent = `${confidence} confidence`;
  meta.appendChild(conf);

  card.append(head, meta);

  if (item.notes && String(item.notes).trim()) {
    const notes = document.createElement("p");
    notes.className = "card-notes";
    notes.textContent = item.notes;
    card.appendChild(notes);
  }

  return card;
}

function buildUnidentifiedCard(entry) {
  const card = document.createElement("div");
  card.className = "card";

  const name = document.createElement("h3");
  name.className = "card-name";
  name.textContent = entry.description || "Unidentified object";
  card.appendChild(name);

  if (entry.reason && String(entry.reason).trim()) {
    const reason = document.createElement("p");
    reason.className = "card-notes";
    reason.textContent = entry.reason;
    card.appendChild(reason);
  }
  return card;
}

// --- Fridge (current inventory) ---------------------------------------------
async function loadFridge() {
  els.fridgeError.hidden = true;
  els.fridgeEmpty.hidden = true;
  els.fridgeBody.hidden = true;
  els.fridgeLoading.hidden = false;

  try {
    const response = await fetch("/api/current");
    let payload;
    try {
      payload = await response.json();
    } catch (_e) {
      payload = null;
    }

    if (!response.ok) {
      els.fridgeError.textContent =
        (payload && payload.error) || "Couldn't load your fridge right now.";
      els.fridgeError.hidden = false;
      return;
    }

    const scan = payload && payload.scan;
    if (!scan) {
      els.fridgeAsof.textContent = "Your most recent scan is your current inventory.";
      els.fridgeEmpty.hidden = false;
      return;
    }
    renderFridge(scan);
  } catch (_err) {
    els.fridgeError.textContent = "Couldn't reach the server to load your fridge.";
    els.fridgeError.hidden = false;
  } finally {
    els.fridgeLoading.hidden = true;
  }
}

function renderFridge(scan) {
  const items = Array.isArray(scan.items) ? scan.items : [];
  const unidentified = Array.isArray(scan.unidentified) ? scan.unidentified : [];

  els.fridgeSummary.innerHTML = "";
  els.fridgeGroups.innerHTML = "";
  els.fridgeCallout.hidden = true;

  // "As of" header line.
  const sourceLabel = scan.source === "webcam" ? "📷 Webcam" : "⬆️ Upload";
  els.fridgeAsof.textContent =
    `As of ${formatTimestamp(scan.created_at)} · ${sourceLabel}`;

  // Headline count (distinct entries, matching the History tab).
  const total = items.length;
  const totalEl = document.createElement("div");
  totalEl.className = "fridge-total";
  totalEl.textContent = `${total} ${total === 1 ? "item" : "items"} in your fridge`;
  els.fridgeSummary.appendChild(totalEl);

  // Freshness breakdown pills (non-zero only, in a stable order).
  const counts = {};
  for (const f of FRESHNESS_ORDER) counts[f] = 0;
  for (const item of items) {
    const f = FRESHNESS_LABELS[item.freshness] ? item.freshness : "unknown";
    counts[f] += 1;
  }
  const freshRow = document.createElement("div");
  freshRow.className = "fridge-freshness";
  for (const f of FRESHNESS_ORDER) {
    if (!counts[f]) continue;
    const pill = document.createElement("span");
    pill.className = `badge badge-${f}`;
    pill.textContent = `${FRESHNESS_LABELS[f]} ${counts[f]}`;
    freshRow.appendChild(pill);
  }
  if (freshRow.children.length) els.fridgeSummary.appendChild(freshRow);

  // "Use these first" callout for spoiled / use-soon items.
  const urgent = [];
  if (counts.spoiled) urgent.push(`${counts.spoiled} spoiled`);
  if (counts.use_soon) urgent.push(`${counts.use_soon} to use soon`);
  if (urgent.length) {
    els.fridgeCallout.className =
      `fridge-callout ${counts.spoiled ? "fridge-callout-spoiled" : "fridge-callout-soon"}`;
    els.fridgeCallout.textContent = `⚠️ Use these first — ${urgent.join(" · ")}.`;
    els.fridgeCallout.hidden = false;
  }

  // Items grouped by category.
  if (total === 0) {
    const none = document.createElement("p");
    none.className = "muted";
    none.textContent = "No food items were in the latest scan.";
    els.fridgeGroups.appendChild(none);
  } else {
    for (const { category, groupItems } of groupByCategory(items)) {
      const section = document.createElement("div");
      section.className = "fridge-group";

      const title = document.createElement("h3");
      title.className = "section-title fridge-group-title";
      title.textContent = category;
      const pill = document.createElement("span");
      pill.className = "count-pill";
      pill.textContent = String(groupItems.length);
      title.appendChild(pill);
      section.appendChild(title);

      const grid = document.createElement("div");
      grid.className = "cards";
      for (const item of groupItems) grid.appendChild(buildItemCard(item));
      section.appendChild(grid);

      els.fridgeGroups.appendChild(section);
    }
  }

  // Prepare the "couldn't identify" note; the toggle governs whether it's shown.
  fridgeHasUnidentified = unidentified.length > 0;
  if (fridgeHasUnidentified) {
    const n = unidentified.length;
    els.fridgeUnidentified.textContent =
      `${n} item${n === 1 ? "" : "s"} in the latest scan couldn't be identified — see the History tab for details.`;
  }

  // Offer the "Show all items" collapse only when there's a grouped list to hide;
  // an empty fridge just shows its "no items" message. Reset to expanded each render.
  if (total > 0) {
    els.fridgeToggle.hidden = false;
    setFridgeItemsOpen(true);
  } else {
    els.fridgeToggle.hidden = true;
    els.fridgeGroups.hidden = false;
    els.fridgeUnidentified.hidden = !fridgeHasUnidentified;
  }

  els.fridgeBody.hidden = false;
}

// Expand/collapse the My Fridge item list (groups + the unidentified note), keeping the
// toggle's label and aria state in sync. The summary and callout stay visible either way.
function setFridgeItemsOpen(open) {
  fridgeItemsOpen = open;
  els.fridgeToggle.setAttribute("aria-expanded", String(open));
  els.fridgeToggle.textContent = open ? "Hide items ▴" : "Show all items ▾";
  els.fridgeGroups.hidden = !open;
  els.fridgeUnidentified.hidden = !(open && fridgeHasUnidentified);
}

// Group items by category, ordering known categories first then any extras A→Z.
function groupByCategory(items) {
  const map = new Map();
  for (const item of items) {
    const cat = (item.category && String(item.category).trim().toLowerCase()) || "other";
    if (!map.has(cat)) map.set(cat, []);
    map.get(cat).push(item);
  }
  const known = CATEGORY_ORDER.filter((c) => map.has(c));
  const extras = [...map.keys()].filter((c) => !CATEGORY_ORDER.includes(c)).sort();
  return [...known, ...extras].map((category) => ({
    category,
    groupItems: map.get(category),
  }));
}

// --- History ----------------------------------------------------------------
async function loadHistory() {
  els.historyError.hidden = true;
  els.historyEmpty.hidden = true;
  els.historyList.innerHTML = "";
  els.historyLoading.hidden = false;

  try {
    const response = await fetch("/api/history");
    let payload;
    try {
      payload = await response.json();
    } catch (_e) {
      payload = null;
    }

    if (!response.ok) {
      els.historyError.textContent =
        (payload && payload.error) || "Couldn't load history right now.";
      els.historyError.hidden = false;
      return;
    }

    const scans = (payload && Array.isArray(payload.scans)) ? payload.scans : [];
    if (scans.length === 0) {
      els.historyEmpty.hidden = false;
      return;
    }
    for (const scan of scans) {
      els.historyList.appendChild(buildScanEntry(scan));
    }
  } catch (_err) {
    els.historyError.textContent = "Couldn't reach the server to load history.";
    els.historyError.hidden = false;
  } finally {
    els.historyLoading.hidden = true;
  }
}

function buildScanEntry(scan) {
  const items = Array.isArray(scan.items) ? scan.items : [];
  const unidentified = Array.isArray(scan.unidentified) ? scan.unidentified : [];

  const entry = document.createElement("div");
  entry.className = "scan-entry";

  // Header: timestamp (local) + source + item count.
  const head = document.createElement("div");
  head.className = "scan-head";

  const when = document.createElement("span");
  when.className = "scan-when";
  when.textContent = formatTimestamp(scan.created_at);

  const meta = document.createElement("span");
  meta.className = "scan-meta muted small";
  const count = items.length;
  const sourceLabel = scan.source === "webcam" ? "📷 Webcam" : "⬆️ Upload";
  meta.textContent = `${sourceLabel} · ${count} ${count === 1 ? "item" : "items"}`;

  head.append(when, meta);
  entry.appendChild(head);

  // Quick peek: item chips (name ×count) with a freshness dot.
  if (items.length > 0) {
    const list = document.createElement("div");
    list.className = "scan-items";
    for (const item of items) {
      const freshness = FRESHNESS_LABELS[item.freshness] ? item.freshness : "unknown";
      const chip = document.createElement("span");
      chip.className = `badge badge-${freshness}`;
      const qty = Number.isFinite(item.count) ? item.count : 1;
      chip.textContent = `${item.name || "item"} ×${qty}`;
      chip.title = `${FRESHNESS_LABELS[freshness]}${item.category ? " · " + item.category : ""}`;
      list.appendChild(chip);
    }
    entry.appendChild(list);
  } else {
    const none = document.createElement("p");
    none.className = "muted small";
    none.textContent = "No food items detected in this scan.";
    entry.appendChild(none);
  }

  if (unidentified.length > 0) {
    const note = document.createElement("p");
    note.className = "muted small scan-unidentified";
    note.textContent = `${unidentified.length} item${unidentified.length === 1 ? "" : "s"} needed your input.`;
    entry.appendChild(note);
  }

  // Expandable detail: everything in this particular image, grouped by category
  // (Vegetables, Fruit, Dairy, …) with full cards, plus any unidentified objects.
  if (items.length > 0 || unidentified.length > 0) {
    const detail = document.createElement("div");
    detail.className = "scan-detail";
    detail.id = `scan-detail-${scan.id}`;
    detail.hidden = true;

    for (const { category, groupItems } of groupByCategory(items)) {
      detail.appendChild(
        buildDetailGroup(category, groupItems.length, groupItems.map(buildItemCard))
      );
    }
    if (unidentified.length > 0) {
      detail.appendChild(
        buildDetailGroup("Needs your input", unidentified.length, unidentified.map(buildUnidentifiedCard))
      );
    }

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "scan-toggle";
    toggle.setAttribute("aria-controls", detail.id);
    const setLabel = (open) => {
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Hide items ▴" : "Show all items ▾";
    };
    setLabel(false);
    toggle.addEventListener("click", () => setLabel((detail.hidden = !detail.hidden) === false));

    entry.append(toggle, detail);
  }

  return entry;
}

// One labelled category block inside a scan's expanded detail: a heading with a count
// pill and a grid of the given cards.
function buildDetailGroup(label, count, cards) {
  const group = document.createElement("div");
  group.className = "scan-detail-group";

  const title = document.createElement("h4");
  title.className = "scan-detail-title";
  title.textContent = label;
  const pill = document.createElement("span");
  pill.className = "count-pill";
  pill.textContent = String(count);
  title.appendChild(pill);
  group.appendChild(title);

  const grid = document.createElement("div");
  grid.className = "cards";
  for (const card of cards) grid.appendChild(card);
  group.appendChild(grid);

  return group;
}

function formatTimestamp(iso) {
  if (!iso) return "Unknown time";
  const date = new Date(iso);
  if (isNaN(date.getTime())) return iso;
  // Render in the viewer's local time zone.
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// --- UI helpers -------------------------------------------------------------
function setLoading(isLoading) {
  els.loading.hidden = !isLoading;
  els.analyze.disabled = isLoading;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}

function hideError() {
  els.error.hidden = true;
  els.error.textContent = "";
}

function clearResults() {
  els.resultsSection.hidden = true;
  els.cards.innerHTML = "";
  els.unidentifiedCards.innerHTML = "";
  els.unidentifiedSection.hidden = true;
}

function releaseObjectUrl() {
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
}

// Release the camera if the user navigates away.
window.addEventListener("pagehide", stopCamera);

// The Fridge tab is the home view: load the current inventory on startup.
activateTab("fridge");
