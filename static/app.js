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
  tabSuggestions: document.getElementById("tab-suggestions"),
  tabPlan: document.getElementById("tab-plan"),
  tabNutrition: document.getElementById("tab-nutrition"),
  tabAnalytics: document.getElementById("tab-analytics"),
  tabChef: document.getElementById("tab-chef"),
  tabUpload: document.getElementById("tab-upload"),
  tabWebcam: document.getElementById("tab-webcam"),
  tabHistory: document.getElementById("tab-history"),
  panelFridge: document.getElementById("panel-fridge"),
  panelSuggestions: document.getElementById("panel-suggestions"),
  panelPlan: document.getElementById("panel-plan"),
  panelNutrition: document.getElementById("panel-nutrition"),
  panelAnalytics: document.getElementById("panel-analytics"),
  panelChef: document.getElementById("panel-chef"),
  panelUpload: document.getElementById("panel-upload"),
  panelWebcam: document.getElementById("panel-webcam"),
  panelHistory: document.getElementById("panel-history"),

  // Real-time status indicator + transient "just updated" toast.
  liveStatus: document.getElementById("live-status"),
  liveLabel: document.getElementById("live-label"),
  liveToast: document.getElementById("live-toast"),

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

  // Suggestions tab elements.
  refreshSuggestions: document.getElementById("refresh-suggestions"),
  perishForm: document.getElementById("perish-form"),
  perishName: document.getElementById("perish-name"),
  perishDate: document.getElementById("perish-date"),
  perishError: document.getElementById("perish-error"),
  perishList: document.getElementById("perish-list"),
  perishEmpty: document.getElementById("perish-empty"),
  suggestLoading: document.getElementById("suggest-loading"),
  suggestError: document.getElementById("suggest-error"),
  suggestBody: document.getElementById("suggest-body"),
  useSoonSection: document.getElementById("use-soon-section"),
  useSoonList: document.getElementById("use-soon-list"),
  useSoonCount: document.getElementById("use-soon-count"),
  recipesSection: document.getElementById("recipes-section"),
  recipesList: document.getElementById("recipes-list"),
  recipesCount: document.getElementById("recipes-count"),
  recipesEmpty: document.getElementById("recipes-empty"),
  shoppingSection: document.getElementById("shopping-section"),
  shoppingList: document.getElementById("shopping-list"),
  suggestMl: document.getElementById("suggest-ml"),

  // Semantic recipe search elements.
  recipeSearchForm: document.getElementById("recipe-search-form"),
  recipeSearchInput: document.getElementById("recipe-search-input"),
  recipeSearchError: document.getElementById("recipe-search-error"),
  recipeSearchStatus: document.getElementById("recipe-search-status"),
  recipeSearchResults: document.getElementById("recipe-search-results"),

  // Meal Plan tab elements.
  planDays: document.getElementById("plan-days"),
  planMeals: document.getElementById("plan-meals"),
  planBuild: document.getElementById("plan-build"),
  planLoading: document.getElementById("plan-loading"),
  planError: document.getElementById("plan-error"),
  planBody: document.getElementById("plan-body"),
  planMetrics: document.getElementById("plan-metrics"),
  planNotes: document.getElementById("plan-notes"),
  planEmpty: document.getElementById("plan-empty"),
  planMealsList: document.getElementById("plan-meals-list"),
  planShoppingSection: document.getElementById("plan-shopping-section"),
  planShoppingCount: document.getElementById("plan-shopping-count"),
  planShopping: document.getElementById("plan-shopping"),
  planAtriskSection: document.getElementById("plan-atrisk-section"),
  planAtrisk: document.getElementById("plan-atrisk"),
  planNutritionSection: document.getElementById("plan-nutrition-section"),
  planNutrition: document.getElementById("plan-nutrition"),
  planNutritionBasis: document.getElementById("plan-nutrition-basis"),

  // Nutrition tab elements.
  nutriRefresh: document.getElementById("nutri-refresh"),
  nutriLoading: document.getElementById("nutri-loading"),
  nutriError: document.getElementById("nutri-error"),
  nutriBody: document.getElementById("nutri-body"),
  nutriMetrics: document.getElementById("nutri-metrics"),
  nutriEmpty: document.getElementById("nutri-empty"),
  nutriTableSection: document.getElementById("nutri-table-section"),
  nutriTable: document.getElementById("nutri-table"),
  nutriCount: document.getElementById("nutri-count"),

  // Barcode scanner (inside the Nutrition tab).
  barcodeForm: document.getElementById("barcode-form"),
  barcodeInput: document.getElementById("barcode-input"),
  barcodeLookup: document.getElementById("barcode-lookup"),
  barcodeScan: document.getElementById("barcode-scan"),
  barcodeStop: document.getElementById("barcode-stop"),
  barcodeCamera: document.getElementById("barcode-camera"),
  barcodeVideo: document.getElementById("barcode-video"),
  barcodeError: document.getElementById("barcode-error"),
  barcodeResult: document.getElementById("barcode-result"),

  // Analytics tab elements.
  analyticsRefresh: document.getElementById("analytics-refresh"),
  analyticsLoading: document.getElementById("analytics-loading"),
  analyticsError: document.getElementById("analytics-error"),
  analyticsBody: document.getElementById("analytics-body"),
  analyticsHeadline: document.getElementById("analytics-headline"),
  analyticsMetrics: document.getElementById("analytics-metrics"),
  analyticsEmpty: document.getElementById("analytics-empty"),
  analyticsTrendSection: document.getElementById("analytics-trend-section"),
  analyticsTrend: document.getElementById("analytics-trend"),
  analyticsTopSection: document.getElementById("analytics-top-section"),
  analyticsTop: document.getElementById("analytics-top"),
  analyticsTopCount: document.getElementById("analytics-top-count"),
  wasteForm: document.getElementById("waste-form"),
  wasteName: document.getElementById("waste-name"),
  wasteCost: document.getElementById("waste-cost"),
  wasteUsed: document.getElementById("waste-used"),
  wasteWasted: document.getElementById("waste-wasted"),
  wasteError: document.getElementById("waste-error"),
  wasteNotice: document.getElementById("waste-notice"),

  // Chef (agent) tab elements.
  chefLog: document.getElementById("chef-log"),
  chefIntro: document.getElementById("chef-intro"),
  chefError: document.getElementById("chef-error"),
  chefForm: document.getElementById("chef-form"),
  chefInput: document.getElementById("chef-input"),
  chefSend: document.getElementById("chef-send"),

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

// The Meal Plan tab solves on first open, then only when the user hits "Build plan".
let planLoadedOnce = false;
// The Nutrition tab loads its cached dashboard on first open; the "Refresh" button pulls
// live from Open Food Facts. The barcode scanner keeps its own camera stream + detect loop.
let nutritionLoadedOnce = false;
let nutritionBusy = false;
let barcodeStream = null;
let barcodeScanning = false;
let barcodeRaf = null;
// The Analytics tab loads the waste-log summary on first open; "Refresh", logging an item,
// or resolving a tracked item re-pulls it. `wasteBusy` guards the manual log buttons.
let analyticsLoadedOnce = false;
let wasteBusy = false;
// The Chef holds a short running transcript so follow-up questions have context.
const chefHistory = [];
let chefBusy = false;

// Which tabs the user has opened at least once. A real-time "state_changed" push only
// re-fetches a tab's data if that tab has actually been loaded, so a background update
// never eagerly populates a panel the user has never visited (matching the lazy-load model).
const tabsSeen = new Set(["fridge"]);

// --- Tab switching ----------------------------------------------------------
function activateTab(which) {
  const tabs = {
    fridge: [els.tabFridge, els.panelFridge],
    suggestions: [els.tabSuggestions, els.panelSuggestions],
    plan: [els.tabPlan, els.panelPlan],
    nutrition: [els.tabNutrition, els.panelNutrition],
    analytics: [els.tabAnalytics, els.panelAnalytics],
    chef: [els.tabChef, els.panelChef],
    upload: [els.tabUpload, els.panelUpload],
    webcam: [els.tabWebcam, els.panelWebcam],
    history: [els.tabHistory, els.panelHistory],
  };

  tabsSeen.add(which);

  for (const [name, [tab, panel]] of Object.entries(tabs)) {
    const active = name === which;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    panel.hidden = !active;
  }

  // The webcam camera belongs only to the Webcam tab; the barcode camera only to Nutrition.
  if (which !== "webcam") {
    stopCamera();
  }
  if (which !== "nutrition") {
    stopBarcodeScan();
  }

  // The capture flow (preview/loading/results) belongs only to the Upload and Webcam
  // tabs; hide it on Fridge/History. Reload data when entering a data-backed tab.
  els.captureExtras.hidden = which !== "upload" && which !== "webcam";
  if (which === "fridge") {
    loadFridge();
  } else if (which === "history") {
    loadHistory();
  } else if (which === "suggestions") {
    loadPerishables();
    loadSuggestions();
  } else if (which === "plan") {
    // Build a plan automatically the first time the tab is opened; after that the user
    // rebuilds explicitly (so changing days/meals doesn't re-solve on every tab switch).
    if (!planLoadedOnce) {
      planLoadedOnce = true;
      loadPlan();
    }
  } else if (which === "nutrition") {
    // Load the cached dashboard on first open; the "Refresh" button pulls live data.
    if (!nutritionLoadedOnce) {
      nutritionLoadedOnce = true;
      loadNutrition();
    }
  } else if (which === "analytics") {
    // Load the waste-log summary on first open; the "Refresh" button re-pulls it.
    if (!analyticsLoadedOnce) {
      analyticsLoadedOnce = true;
      loadAnalytics();
    }
  } else if (which === "chef") {
    els.chefInput.focus();
  }
}

els.tabFridge.addEventListener("click", () => activateTab("fridge"));
els.tabSuggestions.addEventListener("click", () => activateTab("suggestions"));
els.tabPlan.addEventListener("click", () => activateTab("plan"));
els.tabNutrition.addEventListener("click", () => activateTab("nutrition"));
els.tabAnalytics.addEventListener("click", () => activateTab("analytics"));
els.tabChef.addEventListener("click", () => activateTab("chef"));
els.tabUpload.addEventListener("click", () => activateTab("upload"));
els.tabWebcam.addEventListener("click", () => activateTab("webcam"));
els.tabHistory.addEventListener("click", () => activateTab("history"));
els.refreshFridge.addEventListener("click", loadFridge);
els.refreshHistory.addEventListener("click", loadHistory);
els.refreshSuggestions.addEventListener("click", loadSuggestions);
els.perishForm.addEventListener("submit", addPerishable);
els.recipeSearchForm.addEventListener("submit", searchRecipes);
els.fridgeScanCta.addEventListener("click", () => activateTab("upload"));
els.fridgeToggle.addEventListener("click", () => setFridgeItemsOpen(!fridgeItemsOpen));
els.planBuild.addEventListener("click", loadPlan);
els.chefForm.addEventListener("submit", askChef);
els.nutriRefresh.addEventListener("click", refreshNutrition);
els.barcodeForm.addEventListener("submit", lookupBarcode);
els.barcodeScan.addEventListener("click", startBarcodeScan);
els.barcodeStop.addEventListener("click", stopBarcodeScan);
els.analyticsRefresh.addEventListener("click", loadAnalytics);
els.wasteForm.addEventListener("submit", (e) => e.preventDefault());
els.wasteUsed.addEventListener("click", () => logWaste("used"));
els.wasteWasted.addEventListener("click", () => logWaste("wasted"));
// The camera scanner is only offered where the browser can decode barcodes natively.
if ("BarcodeDetector" in window) {
  els.barcodeScan.hidden = false;
}

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
  els.fridgeAsof.textContent =
    `As of ${formatTimestamp(scan.created_at)} · ${sourceLabel(scan.source)}`;
  // Signal when the view reflects meals marked cooked since the last scan.
  if (scan.adjusted_for_consumption) {
    els.fridgeAsof.textContent += " · adjusted after cooking";
  }

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
  meta.textContent = `${sourceLabel(scan.source)} · ${count} ${count === 1 ? "item" : "items"}`;

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

// A short label for how a scan's image arrived: upload or webcam.
function sourceLabel(source) {
  if (source === "webcam") return "📷 Webcam";
  return "⬆️ Upload";
}

// --- Suggestions (recipes + use-soon + shopping) ----------------------------

// Map a recommender "severity" to one of the existing freshness badge colours.
const SEVERITY_BADGE = { overdue: "spoiled", spoiled: "spoiled", soon: "use_soon" };

async function loadPerishables() {
  try {
    const response = await fetch("/api/perishables");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showPerishError((payload && payload.error) || "Couldn't load your tracked items.");
      return;
    }
    renderPerishables((payload && payload.perishables) || []);
  } catch (_err) {
    showPerishError("Couldn't reach the server to load your tracked items.");
  }
}

function renderPerishables(list) {
  els.perishList.innerHTML = "";
  els.perishEmpty.hidden = list.length > 0;
  for (const p of list) {
    els.perishList.appendChild(buildPerishRow(p));
  }
}

function buildPerishRow(p) {
  const row = document.createElement("li");
  row.className = "perish-row";

  const label = document.createElement("span");
  label.className = "perish-label";
  const name = document.createElement("span");
  name.className = "perish-name-text";
  name.textContent = p.name;
  const date = document.createElement("span");
  date.className = "perish-date-text muted small";
  date.textContent = `use by ${formatDate(p.use_by)}`;
  label.append(name, date);

  // Quick actions: close the item out as eaten or thrown away (both log to the waste
  // analytics and stop tracking it), plus the plain "stop tracking" ✕.
  const actions = document.createElement("div");
  actions.className = "perish-actions";

  const used = document.createElement("button");
  used.type = "button";
  used.className = "perish-action perish-action-used";
  used.setAttribute("aria-label", `Mark ${p.name} used`);
  used.textContent = "✅ Used";
  used.addEventListener("click", () => resolvePerishable(p, "used"));

  const wasted = document.createElement("button");
  wasted.type = "button";
  wasted.className = "perish-action perish-action-wasted";
  wasted.setAttribute("aria-label", `Mark ${p.name} wasted`);
  wasted.textContent = "🗑️ Wasted";
  wasted.addEventListener("click", () => resolvePerishable(p, "wasted"));

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "perish-remove";
  remove.setAttribute("aria-label", `Stop tracking ${p.name}`);
  remove.textContent = "✕";
  remove.addEventListener("click", () => deletePerishable(p.id));

  actions.append(used, wasted, remove);
  row.append(label, actions);
  return row;
}

// Close out a tracked perishable as used or wasted: logs it to the waste analytics and
// stops tracking it, then refreshes the tracked list, suggestions, and (if loaded) analytics.
async function resolvePerishable(perishable, event) {
  hidePerishError();
  try {
    const response = await fetch(`/api/perishables/${perishable.id}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event }),
    });
    if (!response.ok && response.status !== 404) {
      const payload = await response.json().catch(() => null);
      showPerishError((payload && payload.error) || "Couldn't update that item.");
      return;
    }
    loadPerishables();
    loadSuggestions();
    // Reflect the new event on the Analytics tab if the user has opened it.
    if (analyticsLoadedOnce) loadAnalytics();
  } catch (_err) {
    showPerishError("Couldn't reach the server to update that item.");
  }
}

async function addPerishable(event) {
  event.preventDefault();
  hidePerishError();
  const name = els.perishName.value.trim();
  const useBy = els.perishDate.value; // <input type=date> gives YYYY-MM-DD
  if (!name) {
    showPerishError("Enter what the item is.");
    return;
  }
  if (!useBy) {
    showPerishError("Pick a use-by date.");
    return;
  }

  try {
    const response = await fetch("/api/perishables", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, use_by: useBy }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showPerishError((payload && payload.error) || "Couldn't save that item.");
      return;
    }
    els.perishForm.reset();
    els.perishName.focus();
    loadPerishables();
    loadSuggestions(); // a new date can change "use soon" and recipe ranking
  } catch (_err) {
    showPerishError("Couldn't reach the server to save that item.");
  }
}

async function deletePerishable(id) {
  try {
    const response = await fetch(`/api/perishables/${id}`, { method: "DELETE" });
    if (!response.ok && response.status !== 404) {
      const payload = await response.json().catch(() => null);
      showPerishError((payload && payload.error) || "Couldn't remove that item.");
      return;
    }
    loadPerishables();
    loadSuggestions();
  } catch (_err) {
    showPerishError("Couldn't reach the server to remove that item.");
  }
}

async function loadSuggestions() {
  els.suggestError.hidden = true;
  els.suggestBody.hidden = true;
  els.suggestLoading.hidden = false;

  try {
    const response = await fetch("/api/recommendations");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.suggestError.textContent =
        (payload && payload.error) || "Couldn't build suggestions right now.";
      els.suggestError.hidden = false;
      return;
    }
    renderSuggestions(payload || {});
  } catch (_err) {
    els.suggestError.textContent = "Couldn't reach the server to build suggestions.";
    els.suggestError.hidden = false;
  } finally {
    els.suggestLoading.hidden = true;
  }
}

function renderSuggestions(data) {
  const useSoon = Array.isArray(data.use_soon) ? data.use_soon : [];
  const recipes = Array.isArray(data.recipes) ? data.recipes : [];
  const shopping = Array.isArray(data.shopping) ? data.shopping : [];

  // Use soon.
  els.useSoonList.innerHTML = "";
  els.useSoonSection.hidden = useSoon.length === 0;
  els.useSoonCount.textContent = String(useSoon.length);
  for (const u of useSoon) els.useSoonList.appendChild(buildUseSoonRow(u));

  // Recipes.
  els.recipesList.innerHTML = "";
  els.recipesCount.textContent = String(recipes.length);
  els.recipesEmpty.hidden = recipes.length > 0;
  for (const r of recipes) els.recipesList.appendChild(buildRecipeCard(r));

  // Shopping.
  els.shoppingList.innerHTML = "";
  els.shoppingSection.hidden = shopping.length === 0;
  for (const s of shopping) els.shoppingList.appendChild(buildShoppingRow(s));

  // ML transparency line.
  const ml = data.ml || {};
  if (ml.backend === "embedding" && ml.dim) {
    els.suggestMl.textContent =
      `Recipes ranked by a ${ml.dim}-dimension ingredient-embedding model we trained on our recipe set, blended with how much of each recipe you already have and what's expiring.`;
  } else {
    els.suggestMl.textContent =
      "Recipes ranked by how much of each recipe you already have and what's expiring.";
  }

  els.suggestBody.hidden = false;
}

function buildUseSoonRow(u) {
  const row = document.createElement("div");
  row.className = "use-soon-row";

  const badgeKind = SEVERITY_BADGE[u.severity] || "use_soon";
  const badge = document.createElement("span");
  badge.className = `badge badge-${badgeKind}`;
  badge.textContent = u.name;

  const reason = document.createElement("span");
  reason.className = "muted small";
  reason.textContent = u.reason || "";

  row.append(badge, reason);
  return row;
}

function buildRecipeCard(recipe) {
  const card = document.createElement("div");
  card.className = "card recipe-card";

  const head = document.createElement("div");
  head.className = "card-head";
  const title = document.createElement("h3");
  title.className = "card-name";
  title.textContent = recipe.title || "Recipe";
  head.appendChild(title);
  if (Number.isFinite(recipe.time_min)) {
    const time = document.createElement("span");
    time.className = "recipe-time muted small";
    time.textContent = `⏱ ${recipe.time_min} min`;
    head.appendChild(time);
  }
  card.appendChild(head);

  // Status tags: ready-to-cook and/or uses-expiring.
  const tags = document.createElement("div");
  tags.className = "recipe-tags";
  if (recipe.can_make) {
    const ready = document.createElement("span");
    ready.className = "recipe-tag recipe-tag-ready";
    ready.textContent = "✓ Ready to cook";
    tags.appendChild(ready);
  }
  if (Array.isArray(recipe.uses_expiring) && recipe.uses_expiring.length) {
    const uses = document.createElement("span");
    uses.className = "recipe-tag recipe-tag-expiring";
    uses.textContent = `Uses your ${recipe.uses_expiring.join(", ")}`;
    tags.appendChild(uses);
  }
  if (tags.children.length) card.appendChild(tags);

  // Have / need chip rows.
  if (Array.isArray(recipe.matched) && recipe.matched.length) {
    card.appendChild(chipRow("Have", recipe.matched, "chip-have"));
  }
  if (Array.isArray(recipe.missing) && recipe.missing.length) {
    card.appendChild(chipRow("Need", recipe.missing, "chip-need"));
  }

  // Coverage bar.
  if (Number.isFinite(recipe.coverage)) {
    const bar = document.createElement("div");
    bar.className = "recipe-cov";
    const fill = document.createElement("div");
    fill.className = "recipe-cov-fill";
    fill.style.width = `${Math.round(recipe.coverage * 100)}%`;
    bar.appendChild(fill);
    card.appendChild(bar);
  }

  return card;
}

// A labelled row of small chips (e.g. "Have: Onion Tomato Garlic").
function chipRow(label, names, chipClass) {
  const row = document.createElement("p");
  row.className = "recipe-chip-row";
  const lead = document.createElement("span");
  lead.className = "recipe-chip-label muted small";
  lead.textContent = `${label}: `;
  row.appendChild(lead);
  for (const name of names) {
    const chip = document.createElement("span");
    chip.className = `chip ${chipClass}`;
    chip.textContent = name;
    row.appendChild(chip);
  }
  return row;
}

// --- Semantic recipe search -------------------------------------------------

async function searchRecipes(e) {
  if (e) e.preventDefault();
  const query = (els.recipeSearchInput.value || "").trim();
  els.recipeSearchError.hidden = true;
  if (!query) {
    els.recipeSearchResults.innerHTML = "";
    els.recipeSearchStatus.hidden = true;
    return;
  }
  els.recipeSearchStatus.hidden = false;
  els.recipeSearchStatus.textContent = "Searching…";
  try {
    const response = await fetch(`/api/recipes/search?q=${encodeURIComponent(query)}`);
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.recipeSearchError.textContent =
        (payload && payload.error) || "Couldn't search recipes right now.";
      els.recipeSearchError.hidden = false;
      els.recipeSearchStatus.hidden = true;
      return;
    }
    const results = (payload && payload.results) || [];
    renderRecipeSearch(
      results,
      results.length
        ? `${results.length} recipe${results.length === 1 ? "" : "s"} for “${query}”`
        : `No recipes matched “${query}”. Try an ingredient or a dish style.`,
    );
  } catch (_err) {
    els.recipeSearchError.textContent = "Couldn't reach the server to search recipes.";
    els.recipeSearchError.hidden = false;
    els.recipeSearchStatus.hidden = true;
  }
}

async function findSimilarRecipes(recipeId, title, button) {
  if (!recipeId) return;
  els.recipeSearchError.hidden = true;
  const original = button ? button.textContent : "";
  if (button) {
    button.disabled = true;
    button.textContent = "Finding…";
  }
  try {
    const response = await fetch(`/api/recipes/${encodeURIComponent(recipeId)}/similar`);
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.recipeSearchError.textContent =
        (payload && payload.error) || "Couldn't find similar recipes right now.";
      els.recipeSearchError.hidden = false;
      return;
    }
    const results = (payload && payload.results) || [];
    renderRecipeSearch(
      results,
      results.length ? `Recipes similar to “${title}”` : `No similar recipes for “${title}”.`,
    );
    els.recipeSearchResults.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (_err) {
    els.recipeSearchError.textContent = "Couldn't reach the server for similar recipes.";
    els.recipeSearchError.hidden = false;
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = original;
    }
  }
}

function renderRecipeSearch(results, statusText) {
  els.recipeSearchStatus.hidden = false;
  els.recipeSearchStatus.textContent = statusText;
  els.recipeSearchResults.innerHTML = "";
  for (const r of results) els.recipeSearchResults.appendChild(buildSearchRecipeCard(r));
}

function buildSearchRecipeCard(r) {
  const card = document.createElement("div");
  card.className = "card recipe-card search-recipe-card";

  const head = document.createElement("div");
  head.className = "card-head";
  const title = document.createElement("h3");
  title.className = "card-name";
  title.textContent = r.title || "Recipe";
  head.appendChild(title);
  if (Number.isFinite(r.similarity) && r.similarity > 0) {
    const match = document.createElement("span");
    match.className = "match-badge";
    match.textContent = `${Math.round(r.similarity * 100)}% match`;
    head.appendChild(match);
  }
  card.appendChild(head);

  // Time + tag chips.
  const meta = document.createElement("div");
  meta.className = "recipe-tags";
  if (Number.isFinite(r.time_min)) {
    const time = document.createElement("span");
    time.className = "recipe-tag";
    time.textContent = `⏱ ${r.time_min} min`;
    meta.appendChild(time);
  }
  for (const tag of Array.isArray(r.tags) ? r.tags : []) {
    const chip = document.createElement("span");
    chip.className = "recipe-tag";
    chip.textContent = tag;
    meta.appendChild(chip);
  }
  if (meta.children.length) card.appendChild(meta);

  // Why it matched.
  const whyBits = [];
  if (Array.isArray(r.matched) && r.matched.length) whyBits.push(r.matched.join(", "));
  if (Array.isArray(r.matched_terms) && r.matched_terms.length) whyBits.push(r.matched_terms.join(", "));
  if (whyBits.length) {
    const why = document.createElement("p");
    why.className = "recipe-why muted small";
    why.textContent = `Matches ${whyBits.join(" · ")}`;
    card.appendChild(why);
  }

  // Full ingredient list (muted).
  if (Array.isArray(r.ingredients) && r.ingredients.length) {
    const ing = document.createElement("p");
    ing.className = "recipe-ingredients muted small";
    ing.textContent = r.ingredients.join(", ");
    card.appendChild(ing);
  }

  const actions = document.createElement("div");
  actions.className = "search-card-actions";
  const similarBtn = document.createElement("button");
  similarBtn.type = "button";
  similarBtn.className = "btn-similar";
  similarBtn.textContent = "Find similar";
  similarBtn.addEventListener("click", () => findSimilarRecipes(r.id, r.title, similarBtn));
  actions.appendChild(similarBtn);
  card.appendChild(actions);

  return card;
}

function buildShoppingRow(s) {
  const row = document.createElement("div");
  row.className = "shopping-row";

  const item = document.createElement("span");
  item.className = "shopping-item";
  item.textContent = s.item;

  const unlocks = document.createElement("span");
  unlocks.className = "muted small";
  const titles = Array.isArray(s.unlocks) ? s.unlocks : [];
  unlocks.textContent = titles.length
    ? `unlocks ${titles.join(", ")}`
    : "";

  row.append(item, unlocks);
  return row;
}

function showPerishError(message) {
  els.perishError.textContent = message;
  els.perishError.hidden = false;
}

function hidePerishError() {
  els.perishError.hidden = true;
  els.perishError.textContent = "";
}

// Format a bare 'YYYY-MM-DD' as a local date without a timezone off-by-one.
function formatDate(ymd) {
  if (!ymd) return "";
  const parts = String(ymd).split("-");
  if (parts.length !== 3) return ymd;
  const [y, m, d] = parts.map((n) => parseInt(n, 10));
  const date = new Date(y, m - 1, d);
  if (isNaN(date.getTime())) return ymd;
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

// --- Meal Plan (zero-waste optimizer) ---------------------------------------
async function loadPlan() {
  const days = parseInt(els.planDays.value, 10) || 3;
  const mealsPerDay = parseInt(els.planMeals.value, 10) || 2;

  els.planError.hidden = true;
  els.planBody.hidden = true;
  els.planLoading.hidden = false;
  els.planBuild.disabled = true;

  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ days, meals_per_day: mealsPerDay }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.planError.textContent =
        (payload && payload.error) || "Couldn't build a meal plan right now.";
      els.planError.hidden = false;
      return;
    }
    renderPlan(payload || {});
  } catch (_err) {
    els.planError.textContent = "Couldn't reach the server to build a meal plan.";
    els.planError.hidden = false;
  } finally {
    els.planLoading.hidden = true;
    els.planBuild.disabled = false;
  }
}

function renderPlan(data) {
  const plan = Array.isArray(data.plan) ? data.plan : [];
  const metrics = data.metrics || {};
  const notes = Array.isArray(data.notes) ? data.notes : [];
  const shopping = Array.isArray(data.shopping_list) ? data.shopping_list : [];
  const atRisk = Array.isArray(data.at_risk_remaining) ? data.at_risk_remaining : [];

  // --- Metric tiles ---
  els.planMetrics.innerHTML = "";
  const solverLabel = data.solver === "ilp" ? "Exact optimizer" : "Fast heuristic";
  els.planMetrics.appendChild(buildStat(solverLabel, "Solver", "plan-stat-solver"));
  if (metrics.slots) {
    els.planMetrics.appendChild(
      buildStat(`${metrics.slots_filled ?? plan.length}/${metrics.slots}`, "Meals planned")
    );
  }
  if (metrics.urgent_total) {
    const pct = metrics.waste_avoided_pct;
    els.planMetrics.appendChild(
      buildStat(
        `${metrics.urgent_used}/${metrics.urgent_total}`,
        pct != null ? `At-risk used (${pct}%)` : "At-risk used",
        "plan-stat-waste"
      )
    );
  }
  if (Number.isFinite(metrics.distinct_ingredients_used)) {
    els.planMetrics.appendChild(
      buildStat(String(metrics.distinct_ingredients_used), "Ingredients used")
    );
  }
  if (Number.isFinite(metrics.new_ingredients_to_buy)) {
    els.planMetrics.appendChild(
      buildStat(String(metrics.new_ingredients_to_buy), "To buy")
    );
  }

  // --- Notes ---
  els.planNotes.innerHTML = "";
  for (const note of notes) {
    const p = document.createElement("p");
    p.className = "plan-note";
    p.textContent = note;
    els.planNotes.appendChild(p);
  }

  // --- Meal cards / empty state ---
  els.planMealsList.innerHTML = "";
  els.planEmpty.hidden = plan.length > 0;
  for (const meal of plan) {
    els.planMealsList.appendChild(buildPlanMealCard(meal));
  }

  // --- Shopping list ---
  els.planShopping.innerHTML = "";
  els.planShoppingSection.hidden = shopping.length === 0;
  els.planShoppingCount.textContent = String(shopping.length);
  for (const s of shopping) {
    els.planShopping.appendChild(buildPlanShoppingRow(s));
  }

  // --- Still at risk ---
  els.planAtrisk.innerHTML = "";
  els.planAtriskSection.hidden = atRisk.length === 0;
  for (const row of atRisk) {
    els.planAtrisk.appendChild(buildUseSoonRow(row));
  }

  // --- Nutrition (optional) ---
  const nutrition = metrics.nutrition;
  if (nutrition) {
    els.planNutrition.innerHTML = "";
    els.planNutrition.appendChild(buildStat(`${nutrition.kcal}`, "kcal"));
    els.planNutrition.appendChild(buildStat(`${nutrition.protein_g} g`, "Protein"));
    els.planNutrition.appendChild(buildStat(`${nutrition.carbs_g} g`, "Carbs"));
    els.planNutrition.appendChild(buildStat(`${nutrition.fat_g} g`, "Fat"));
    els.planNutritionBasis.textContent = nutrition.basis || "";
    els.planNutritionSection.hidden = false;
  } else {
    els.planNutritionSection.hidden = true;
  }

  els.planBody.hidden = false;
}

// One "big number + label" tile for the metrics/nutrition strips.
function buildStat(value, label, extraClass) {
  const stat = document.createElement("div");
  stat.className = `plan-stat${extraClass ? " " + extraClass : ""}`;
  const val = document.createElement("div");
  val.className = "plan-stat-value";
  val.textContent = value;
  const lab = document.createElement("div");
  lab.className = "plan-stat-label muted small";
  lab.textContent = label;
  stat.append(val, lab);
  return stat;
}

function buildPlanMealCard(meal) {
  const card = document.createElement("div");
  card.className = "card plan-meal-card";

  const head = document.createElement("div");
  head.className = "card-head";
  const title = document.createElement("h3");
  title.className = "card-name";
  const slot = Number.isFinite(meal.slot) ? meal.slot : "";
  title.textContent = slot ? `Meal ${slot}: ${meal.title || "Recipe"}` : (meal.title || "Recipe");
  head.appendChild(title);
  if (Number.isFinite(meal.time_min)) {
    const time = document.createElement("span");
    time.className = "recipe-time muted small";
    time.textContent = `⏱ ${meal.time_min} min`;
    head.appendChild(time);
  }
  card.appendChild(head);

  // "Uses your …expiring" highlight — the zero-waste payoff, shown first.
  if (Array.isArray(meal.uses_expiring) && meal.uses_expiring.length) {
    const tags = document.createElement("div");
    tags.className = "recipe-tags";
    const uses = document.createElement("span");
    uses.className = "recipe-tag recipe-tag-expiring";
    uses.textContent = `Uses up ${meal.uses_expiring.join(", ")}`;
    tags.appendChild(uses);
    card.appendChild(tags);
  }

  if (Array.isArray(meal.uses) && meal.uses.length) {
    card.appendChild(chipRow("Uses", meal.uses, "chip-have"));
  }
  if (Array.isArray(meal.to_buy) && meal.to_buy.length) {
    card.appendChild(chipRow("Buy", meal.to_buy, "chip-need"));
  }

  // "Mark cooked" closes the loop: it consumes the on-hand ingredients (decrementing the
  // fridge) and logs each as "used" for the Analytics tab. Only offered when the meal
  // actually draws on inventory — a shopping-only meal has nothing to decrement.
  if (Array.isArray(meal.uses) && meal.uses.length) {
    const actions = document.createElement("div");
    actions.className = "plan-meal-actions";

    const cookBtn = document.createElement("button");
    cookBtn.type = "button";
    cookBtn.className = "btn-cook";
    cookBtn.textContent = "🍳 Mark cooked";

    const status = document.createElement("span");
    status.className = "plan-cook-status muted small";
    status.hidden = true;

    cookBtn.addEventListener("click", () => cookMeal(meal, cookBtn, status));
    actions.append(cookBtn, status);
    card.appendChild(actions);
  }

  return card;
}

// Mark a planned meal cooked: consume its on-hand ingredients (decrementing the fridge)
// and log each as "used" so waste trends down. Closes plan → consumption → inventory.
async function cookMeal(meal, button, status) {
  const uses = Array.isArray(meal.uses)
    ? meal.uses.filter((u) => u && String(u).trim())
    : [];
  if (!uses.length) return;

  button.disabled = true;
  button.textContent = "Cooking…";
  status.hidden = true;
  status.classList.remove("cook-error");

  try {
    const response = await fetch("/api/cook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uses, title: meal.title || "Recipe" }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      button.disabled = false;
      button.textContent = "🍳 Mark cooked";
      status.hidden = false;
      status.classList.add("cook-error");
      status.textContent = (payload && payload.error) || "Couldn't mark that cooked.";
      return;
    }

    button.textContent = "✓ Cooked";
    button.classList.add("btn-cook-done");
    status.hidden = false;
    const cleared = (payload && payload.cleared_perishables) || 0;
    status.textContent = cleared
      ? `Inventory updated · cleared ${cleared} at-risk item${cleared === 1 ? "" : "s"}.`
      : "Inventory updated.";

    // Reflect the decremented inventory and the new "used" events on other tabs.
    loadFridge();
    if (analyticsLoadedOnce) loadAnalytics();
  } catch (_err) {
    button.disabled = false;
    button.textContent = "🍳 Mark cooked";
    status.hidden = false;
    status.classList.add("cook-error");
    status.textContent = "Couldn't reach the server.";
  }
}

function buildPlanShoppingRow(s) {
  const row = document.createElement("div");
  row.className = "shopping-row";

  const item = document.createElement("span");
  item.className = "shopping-item";
  item.textContent = s.item;

  const forRecipes = document.createElement("span");
  forRecipes.className = "muted small";
  const titles = Array.isArray(s.for_recipes) ? s.for_recipes : [];
  forRecipes.textContent = titles.length ? `for ${titles.join(", ")}` : "";

  row.append(item, forRecipes);
  return row;
}

// --- Nutrition (Open Food Facts dashboard + barcode scanner) ----------------
async function loadNutrition() {
  els.nutriError.hidden = true;
  els.nutriBody.hidden = true;
  els.nutriLoading.hidden = false;

  try {
    const response = await fetch("/api/nutrition");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.nutriError.textContent =
        (payload && payload.error) || "Couldn't load nutrition right now.";
      els.nutriError.hidden = false;
      return;
    }
    renderNutrition(payload || {});
  } catch (_err) {
    els.nutriError.textContent = "Couldn't reach the server to load nutrition.";
    els.nutriError.hidden = false;
  } finally {
    els.nutriLoading.hidden = true;
  }
}

function renderNutrition(data) {
  const items = Array.isArray(data.items) ? data.items : [];
  const totals = data.totals || {};
  const coverage = data.coverage || {};
  const tracked = Number.isFinite(coverage.tracked) ? coverage.tracked : items.length;
  const withData = Number.isFinite(coverage.with_data) ? coverage.with_data : 0;

  // --- Metric tiles: coverage first, then summed macros ---
  els.nutriMetrics.innerHTML = "";
  els.nutriMetrics.appendChild(
    buildStat(`${withData}/${tracked}`, "Ingredients with data", "plan-stat-solver")
  );
  els.nutriMetrics.appendChild(buildStat(fmtNum(totals.kcal), "kcal"));
  els.nutriMetrics.appendChild(buildStat(`${fmtNum(totals.protein_g)} g`, "Protein"));
  els.nutriMetrics.appendChild(buildStat(`${fmtNum(totals.carbs_g)} g`, "Carbs"));
  els.nutriMetrics.appendChild(buildStat(`${fmtNum(totals.fat_g)} g`, "Fat"));

  // --- Per-ingredient table / empty state ---
  els.nutriTable.innerHTML = "";
  if (tracked === 0) {
    els.nutriEmpty.hidden = false;
    els.nutriTableSection.hidden = true;
  } else {
    els.nutriEmpty.hidden = true;
    els.nutriCount.textContent = String(items.length);
    // Header row, then one row per ingredient.
    els.nutriTable.appendChild(buildNutriRow(
      { name: "Ingredient", kcal: "kcal", protein_g: "Protein", carbs_g: "Carbs", fat_g: "Fat" },
      { header: true }
    ));
    for (const item of items) els.nutriTable.appendChild(buildNutriRow(item));
    els.nutriTableSection.hidden = false;
  }

  els.nutriBody.hidden = false;
}

// One row of the per-ingredient nutrition table. `header:true` renders the label row and
// prints the given strings verbatim; data rows show macros or an em-dash when unknown.
function buildNutriRow(item, opts = {}) {
  const row = document.createElement("div");
  row.className = `nutri-row${opts.header ? " nutri-row-head" : ""}`;

  const name = document.createElement("span");
  name.className = "nutri-cell nutri-cell-name";
  name.textContent = item.name || "—";
  row.appendChild(name);

  const macros = opts.header
    ? [item.kcal, item.protein_g, item.carbs_g, item.fat_g]
    : [
        fmtMacro(item.kcal),
        fmtMacro(item.protein_g, "g"),
        fmtMacro(item.carbs_g, "g"),
        fmtMacro(item.fat_g, "g"),
      ];
  for (const value of macros) {
    const cell = document.createElement("span");
    cell.className = "nutri-cell nutri-cell-num";
    cell.textContent = value;
    row.appendChild(cell);
  }
  return row;
}

async function refreshNutrition() {
  if (nutritionBusy) return;
  nutritionBusy = true;
  els.nutriRefresh.disabled = true;
  const original = els.nutriRefresh.textContent;
  els.nutriRefresh.textContent = "Refreshing…";
  els.nutriError.hidden = true;

  try {
    const response = await fetch("/api/nutrition/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.nutriError.textContent =
        (payload && payload.error) || "Couldn't refresh nutrition right now.";
      els.nutriError.hidden = false;
      return;
    }
    await loadNutrition();
    showNutriSummary(payload || {});
  } catch (_err) {
    els.nutriError.textContent = "Couldn't reach the server to refresh nutrition.";
    els.nutriError.hidden = false;
  } finally {
    nutritionBusy = false;
    els.nutriRefresh.disabled = false;
    els.nutriRefresh.textContent = original;
  }
}

// Report what a refresh actually did, inline in the (green-styled) error banner slot.
function showNutriSummary(summary) {
  const updated = (summary.updated || []).length;
  const cached = (summary.cached || []).length;
  const notFound = (summary.not_found || []).length;
  const failed = (summary.failed || []).length;
  const parts = [];
  if (updated) parts.push(`${updated} updated`);
  if (cached) parts.push(`${cached} already had data`);
  if (notFound) parts.push(`${notFound} not on Open Food Facts`);
  if (failed) parts.push(`${failed} couldn't be fetched`);
  els.nutriError.textContent = parts.length
    ? `Refreshed: ${parts.join(" · ")}.`
    : "Nothing to refresh — no ingredients on hand yet.";
  els.nutriError.className = "error-banner nutri-notice";
  els.nutriError.hidden = false;
}

// --- Barcode scanner --------------------------------------------------------
async function lookupBarcode(event) {
  if (event) event.preventDefault();
  const code = (els.barcodeInput.value || "").replace(/\D/g, "");
  hideBarcodeError();
  if (code.length < 8 || code.length > 14) {
    showBarcodeError("Enter a barcode of 8–14 digits.");
    return;
  }

  els.barcodeLookup.disabled = true;
  els.barcodeResult.hidden = true;

  try {
    const response = await fetch("/api/barcode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    const payload = await response.json().catch(() => null);
    if (response.status === 404) {
      showBarcodeError("No product found for that barcode on Open Food Facts.");
      return;
    }
    if (!response.ok) {
      showBarcodeError((payload && payload.error) || "Couldn't look up that barcode.");
      return;
    }
    renderBarcodeProduct(payload || {});
    // A cached product may have added a nutrition row; refresh the dashboard if it's loaded.
    if (nutritionLoadedOnce) loadNutrition();
  } catch (_err) {
    showBarcodeError("Couldn't reach the server to look up that barcode.");
  } finally {
    els.barcodeLookup.disabled = false;
  }
}

function renderBarcodeProduct(data) {
  const product = data.product || {};
  els.barcodeResult.innerHTML = "";

  const card = document.createElement("div");
  card.className = "card barcode-card";

  const head = document.createElement("div");
  head.className = "card-head";
  const name = document.createElement("h3");
  name.className = "card-name";
  name.textContent = product.name || "Product";
  head.appendChild(name);
  card.appendChild(head);

  const meta = document.createElement("div");
  meta.className = "card-meta";
  if (product.brands) {
    const brand = document.createElement("span");
    brand.className = "chip";
    brand.textContent = product.brands;
    meta.appendChild(brand);
  }
  const codeChip = document.createElement("span");
  codeChip.className = "chip";
  codeChip.textContent = `#${product.code || ""}`;
  meta.appendChild(codeChip);
  card.appendChild(meta);

  if (data.has_nutrition) {
    const strip = document.createElement("div");
    strip.className = "plan-nutrition";
    strip.appendChild(buildStat(fmtNum(product.kcal), "kcal"));
    strip.appendChild(buildStat(`${fmtNum(product.protein_g)} g`, "Protein"));
    strip.appendChild(buildStat(`${fmtNum(product.carbs_g)} g`, "Carbs"));
    strip.appendChild(buildStat(`${fmtNum(product.fat_g)} g`, "Fat"));
    card.appendChild(strip);
    const note = document.createElement("p");
    note.className = "muted small";
    note.textContent = data.cached
      ? "Per 100 g. Saved to your nutrition data."
      : "Per 100 g.";
    card.appendChild(note);
  } else {
    const note = document.createElement("p");
    note.className = "muted small";
    note.textContent = "Open Food Facts has this product but no nutrition facts for it.";
    card.appendChild(note);
  }

  els.barcodeResult.appendChild(card);
  els.barcodeResult.hidden = false;
}

async function startBarcodeScan() {
  hideBarcodeError();
  if (!("BarcodeDetector" in window)) {
    showBarcodeError("This browser can't scan barcodes — type the number instead.");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showBarcodeError("No camera available — type the barcode instead.");
    return;
  }

  let detector;
  try {
    detector = new window.BarcodeDetector({
      formats: ["ean_13", "ean_8", "upc_a", "upc_e", "code_128"],
    });
  } catch (_err) {
    showBarcodeError("Couldn't start the barcode scanner — type the number instead.");
    return;
  }

  try {
    barcodeStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
  } catch (err) {
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      showBarcodeError("Camera permission was denied — type the barcode instead.");
    } else if (err && err.name === "NotFoundError") {
      showBarcodeError("No camera was found — type the barcode instead.");
    } else {
      showBarcodeError("Couldn't start the camera — type the barcode instead.");
    }
    return;
  }

  els.barcodeVideo.srcObject = barcodeStream;
  els.barcodeCamera.hidden = false;
  els.barcodeScan.hidden = true;
  els.barcodeStop.hidden = false;
  barcodeScanning = true;

  const tick = async () => {
    if (!barcodeScanning) return;
    try {
      const codes = await detector.detect(els.barcodeVideo);
      if (codes && codes.length && codes[0].rawValue) {
        const value = codes[0].rawValue.replace(/\D/g, "");
        if (value.length >= 8) {
          els.barcodeInput.value = value;
          stopBarcodeScan();
          lookupBarcode();
          return;
        }
      }
    } catch (_err) {
      // A transient decode error is fine; keep scanning until the user stops.
    }
    barcodeRaf = requestAnimationFrame(tick);
  };
  barcodeRaf = requestAnimationFrame(tick);
}

function stopBarcodeScan() {
  barcodeScanning = false;
  if (barcodeRaf) {
    cancelAnimationFrame(barcodeRaf);
    barcodeRaf = null;
  }
  if (barcodeStream) {
    barcodeStream.getTracks().forEach((track) => track.stop());
    barcodeStream = null;
    els.barcodeVideo.srcObject = null;
  }
  els.barcodeCamera.hidden = true;
  els.barcodeStop.hidden = true;
  if ("BarcodeDetector" in window) els.barcodeScan.hidden = false;
}

function showBarcodeError(message) {
  els.barcodeError.textContent = message;
  els.barcodeError.hidden = false;
}

function hideBarcodeError() {
  els.barcodeError.hidden = true;
  els.barcodeError.textContent = "";
}

// Format an optional number for a metric tile: one decimal, or a dash when unknown.
function fmtNum(value) {
  if (value == null || value === "" || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

// Format an optional macro for a table cell, with an optional unit suffix.
function fmtMacro(value, unit) {
  if (value == null || value === "" || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  const text = Number.isInteger(n) ? String(n) : n.toFixed(1);
  return unit ? `${text} ${unit}` : text;
}

// --- Analytics (waste & spend dashboard) ------------------------------------
async function loadAnalytics() {
  els.analyticsError.hidden = true;
  els.analyticsBody.hidden = true;
  els.analyticsLoading.hidden = false;

  try {
    const response = await fetch("/api/analytics");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.analyticsError.textContent =
        (payload && payload.error) || "Couldn't load your analytics right now.";
      els.analyticsError.hidden = false;
      return;
    }
    renderAnalytics(payload || {});
  } catch (_err) {
    els.analyticsError.textContent = "Couldn't reach the server to load your analytics.";
    els.analyticsError.hidden = false;
  } finally {
    els.analyticsLoading.hidden = true;
  }
}

function renderAnalytics(data) {
  const totals = data.totals || {};
  const money = data.money || {};
  const topWasted = Array.isArray(data.top_wasted) ? data.top_wasted : [];
  const byWeek = Array.isArray(data.by_week) ? data.by_week : [];
  const hasData = !!data.has_data;

  els.analyticsHeadline.textContent = data.headline || "";

  // --- Metric tiles ---
  els.analyticsMetrics.innerHTML = "";
  els.analyticsMetrics.appendChild(
    buildStat(String(totals.events || 0), "Items logged", "plan-stat-solver")
  );
  els.analyticsMetrics.appendChild(buildStat(String(totals.used || 0), "Used"));
  els.analyticsMetrics.appendChild(
    buildStat(String(totals.wasted || 0), "Wasted", "plan-stat-waste")
  );
  els.analyticsMetrics.appendChild(
    buildStat(`${fmtNum(data.waste_rate)}%`, "Waste rate", "plan-stat-waste")
  );
  if (money.has_cost) {
    els.analyticsMetrics.appendChild(buildStat(fmtNum(money.wasted), "Value wasted", "plan-stat-waste"));
    els.analyticsMetrics.appendChild(buildStat(fmtNum(money.saved), "Value used"));
  }

  // --- Empty state vs charts ---
  els.analyticsEmpty.hidden = hasData;

  // Weekly trend.
  els.analyticsTrend.innerHTML = "";
  if (hasData) {
    els.analyticsTrend.appendChild(buildTrendChart(byWeek));
    els.analyticsTrendSection.hidden = false;
  } else {
    els.analyticsTrendSection.hidden = true;
  }

  // Most wasted.
  els.analyticsTop.innerHTML = "";
  if (topWasted.length) {
    els.analyticsTopCount.textContent = String(topWasted.length);
    for (const item of topWasted) {
      els.analyticsTop.appendChild(buildTopWastedRow(item, money.has_cost));
    }
    els.analyticsTopSection.hidden = false;
  } else {
    els.analyticsTopSection.hidden = true;
  }

  els.analyticsBody.hidden = false;
}

// A small stacked-bar chart: one column per week, used (green) below wasted (red), heights
// scaled to the busiest week. Legend-free — each bar carries a descriptive tooltip.
function buildTrendChart(byWeek) {
  const chart = document.createElement("div");
  chart.className = "trend-chart";
  const maxTotal = Math.max(1, ...byWeek.map((w) => (w.used || 0) + (w.wasted || 0)));

  for (const week of byWeek) {
    const used = week.used || 0;
    const wasted = week.wasted || 0;

    const col = document.createElement("div");
    col.className = "trend-col";

    const bar = document.createElement("div");
    bar.className = "trend-bar";
    bar.title = `Week of ${formatDate(week.week_start)}: ${used} used · ${wasted} wasted`;

    const usedSeg = document.createElement("div");
    usedSeg.className = "trend-seg trend-seg-used";
    usedSeg.style.height = `${(used / maxTotal) * 100}%`;
    const wastedSeg = document.createElement("div");
    wastedSeg.className = "trend-seg trend-seg-wasted";
    wastedSeg.style.height = `${(wasted / maxTotal) * 100}%`;
    // column-reverse stacking: used sits at the bottom, wasted on top.
    bar.append(usedSeg, wastedSeg);

    const label = document.createElement("span");
    label.className = "trend-label muted small";
    label.textContent = formatWeekLabel(week.week_start);

    col.append(bar, label);
    chart.appendChild(col);
  }
  return chart;
}

function buildTopWastedRow(item, hasCost) {
  const row = document.createElement("div");
  row.className = "top-wasted-row";

  const name = document.createElement("span");
  name.className = "top-wasted-name";
  name.textContent = item.name || item.token || "—";

  const count = document.createElement("span");
  count.className = "badge badge-spoiled";
  const n = item.wasted || 0;
  count.textContent = `${n}×`;

  row.append(name, count);

  if (hasCost && Number.isFinite(Number(item.cost)) && Number(item.cost) > 0) {
    const cost = document.createElement("span");
    cost.className = "muted small top-wasted-cost";
    cost.textContent = fmtNum(item.cost);
    row.appendChild(cost);
  }
  return row;
}

// Manually log a used/wasted item that wasn't a tracked perishable.
async function logWaste(event) {
  if (wasteBusy) return;
  hideWasteError();
  const name = els.wasteName.value.trim();
  const costRaw = els.wasteCost.value.trim();
  if (!name) {
    showWasteError("Enter what the item is.");
    return;
  }
  const body = { name, event };
  if (costRaw !== "") {
    const cost = Number(costRaw);
    if (!Number.isFinite(cost) || cost < 0) {
      showWasteError("Enter a cost as a non-negative number, or leave it blank.");
      return;
    }
    body.est_cost = cost;
  }

  wasteBusy = true;
  els.wasteUsed.disabled = true;
  els.wasteWasted.disabled = true;

  try {
    const response = await fetch("/api/waste", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showWasteError((payload && payload.error) || "Couldn't save that right now.");
      return;
    }
    els.wasteForm.reset();
    els.wasteName.focus();
    showWasteNotice(`Logged “${name}” as ${event}.`);
    loadAnalytics();
  } catch (_err) {
    showWasteError("Couldn't reach the server to save that.");
  } finally {
    wasteBusy = false;
    els.wasteUsed.disabled = false;
    els.wasteWasted.disabled = false;
  }
}

function showWasteNotice(message) {
  els.wasteNotice.textContent = message;
  els.wasteNotice.hidden = false;
}

function showWasteError(message) {
  els.wasteNotice.hidden = true;
  els.wasteError.textContent = message;
  els.wasteError.hidden = false;
}

function hideWasteError() {
  els.wasteError.hidden = true;
  els.wasteError.textContent = "";
  els.wasteNotice.hidden = true;
}

// A compact week label for the trend chart's x-axis, e.g. "Sep 15".
function formatWeekLabel(ymd) {
  if (!ymd) return "";
  const parts = String(ymd).split("-");
  if (parts.length !== 3) return ymd;
  const [y, m, d] = parts.map((n) => parseInt(n, 10));
  const date = new Date(y, m - 1, d);
  if (isNaN(date.getTime())) return ymd;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// --- Chef (tool-using agent) ------------------------------------------------
async function askChef(event) {
  event.preventDefault();
  const message = els.chefInput.value.trim();
  if (!message || chefBusy) return;

  els.chefError.hidden = true;
  if (els.chefIntro) els.chefIntro.hidden = true;

  appendChefBubble("user", message);
  els.chefInput.value = "";
  chefBusy = true;
  els.chefSend.disabled = true;
  els.chefInput.disabled = true;

  const pending = appendChefPending();
  const historyToSend = chefHistory.slice();

  try {
    const response = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: historyToSend }),
    });
    const payload = await response.json().catch(() => null);
    pending.remove();

    if (!response.ok) {
      const msg = (payload && payload.error) || `The Chef returned an error (HTTP ${response.status}).`;
      // 503 = not configured on the server; show it inline rather than as a hard failure.
      els.chefError.textContent = msg;
      els.chefError.hidden = false;
      return;
    }

    const answer = (payload && payload.answer) || "(no answer)";
    appendChefBubble("assistant", answer, payload && payload.steps, payload && payload.model);

    // Keep a bounded running transcript for follow-up context.
    chefHistory.push({ role: "user", content: message });
    chefHistory.push({ role: "assistant", content: answer });
    while (chefHistory.length > 10) chefHistory.shift();
  } catch (_err) {
    pending.remove();
    els.chefError.textContent = "Couldn't reach the server to ask the Chef.";
    els.chefError.hidden = false;
  } finally {
    chefBusy = false;
    els.chefSend.disabled = false;
    els.chefInput.disabled = false;
    els.chefInput.focus();
  }
}

function appendChefBubble(role, text, steps, model) {
  const bubble = document.createElement("div");
  bubble.className = `chef-msg chef-msg-${role}`;

  const body = document.createElement("p");
  body.className = "chef-msg-text";
  body.textContent = text;
  bubble.appendChild(body);

  // For the assistant, expose which tools it called (transparency), collapsed by default.
  if (role === "assistant" && Array.isArray(steps) && steps.length) {
    const details = document.createElement("details");
    details.className = "chef-steps";
    const summary = document.createElement("summary");
    const toolNames = steps.map((s) => s.tool).filter(Boolean);
    summary.textContent = `Used ${toolNames.length} tool${toolNames.length === 1 ? "" : "s"}${model ? " · " + model : ""}`;
    details.appendChild(summary);
    for (const step of steps) {
      const line = document.createElement("div");
      line.className = "chef-step muted small";
      line.textContent = `${step.tool}(${step.arguments ? JSON.stringify(step.arguments) : ""})`;
      details.appendChild(line);
    }
    bubble.appendChild(details);
  }

  els.chefLog.appendChild(bubble);
  els.chefLog.scrollTop = els.chefLog.scrollHeight;
  return bubble;
}

function appendChefPending() {
  const bubble = document.createElement("div");
  bubble.className = "chef-msg chef-msg-assistant chef-msg-pending";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const label = document.createElement("span");
  label.textContent = "The Chef is thinking…";
  bubble.append(spinner, label);
  els.chefLog.appendChild(bubble);
  els.chefLog.scrollTop = els.chefLog.scrollHeight;
  return bubble;
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

// Release any camera if the user navigates away.
window.addEventListener("pagehide", () => {
  stopCamera();
  stopBarcodeScan();
});

// --- Real-time updates (Socket.IO) ------------------------------------------
// When the fridge changes on the server, every open tab receives a "state_changed" push
// and live-refreshes the affected views. Degrades gracefully: if the Socket.IO client
// didn't load (offline, CDN blocked), the app keeps working with manual refresh as before.

// Map each server "scope" to the client refresh it triggers. A refresh runs only if that
// tab has been opened (tabsSeen) — except the fridge, the home view, which stays current so
// it's fresh whenever the user returns. The meal plan is intentionally excluded: re-solving
// on every change would fight the "rebuild explicitly" design.
const LIVE_SCOPES = {
  inventory: () => {
    loadFridge();
    if (tabsSeen.has("suggestions")) loadSuggestions();
  },
  perishables: () => {
    if (tabsSeen.has("suggestions")) loadPerishables();
  },
  analytics: () => {
    if (tabsSeen.has("analytics")) loadAnalytics();
  },
  nutrition: () => {
    if (tabsSeen.has("nutrition")) loadNutrition();
  },
};

const SCOPE_LABELS = {
  inventory: "Fridge",
  perishables: "Tracked items",
  analytics: "Analytics",
  nutrition: "Nutrition",
};

let liveToastTimer = null;
let liveToastHideTimer = null;

function setLiveStatus(state, label) {
  if (!els.liveStatus) return;
  els.liveStatus.hidden = false;
  els.liveStatus.dataset.state = state;
  if (els.liveLabel) els.liveLabel.textContent = label;
}

function showLiveToast(scopes) {
  if (!els.liveToast) return;
  const names = scopes.map((s) => SCOPE_LABELS[s]).filter(Boolean);
  els.liveToast.textContent = `🔄 Updated live: ${names.length ? names.join(" · ") : "Fridge"}`;
  els.liveToast.hidden = false;
  // Force a reflow so re-triggering the transition restarts the fade-in.
  void els.liveToast.offsetWidth;
  els.liveToast.classList.add("show");
  if (liveToastTimer) clearTimeout(liveToastTimer);
  if (liveToastHideTimer) clearTimeout(liveToastHideTimer);
  liveToastTimer = setTimeout(() => {
    els.liveToast.classList.remove("show");
    // Keep it out of the a11y tree once faded so it isn't re-announced.
    liveToastHideTimer = setTimeout(() => {
      els.liveToast.hidden = true;
    }, 300);
  }, 2600);
}

function handleStateChanged(payload) {
  const scopes = (payload && Array.isArray(payload.scopes) ? payload.scopes : []).filter(
    (s) => s in LIVE_SCOPES
  );
  if (!scopes.length) return;
  for (const scope of scopes) {
    try {
      LIVE_SCOPES[scope]();
    } catch (err) {
      console.error("Live refresh failed for scope", scope, err);
    }
  }
  showLiveToast(scopes);
}

function initRealtime() {
  // No Socket.IO client on the page (CDN blocked / offline) → stay on manual refresh.
  if (typeof io !== "function") return;

  let socket;
  try {
    socket = io(); // same-origin; polling with automatic upgrade to WebSocket
  } catch (err) {
    console.warn("Real-time updates unavailable:", err);
    return;
  }

  setLiveStatus("connecting", "Connecting…");
  socket.on("connect", () => setLiveStatus("live", "Live"));
  socket.on("disconnect", () => setLiveStatus("offline", "Offline"));
  socket.on("connect_error", () => setLiveStatus("offline", "Offline"));
  socket.on("state_changed", handleStateChanged);
}

// The Fridge tab is the home view: load the current inventory on startup.
activateTab("fridge");

// Subscribe to live updates once the initial view is set up.
initRealtime();
