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
  // Frost shell: the seven top-level destinations, the nav surfaces that switch
  // between them (frosted sidebar on laptop, glass tab bar + "more" sheet on phone),
  // and the segmented sub-navs inside Scan / Plan / Insights.
  dests: {
    fridge: document.getElementById("dest-fridge"),
    recipes: document.getElementById("dest-recipes"),
    scan: document.getElementById("dest-scan"),
    chef: document.getElementById("dest-chef"),
    cart: document.getElementById("dest-cart"),
    plan: document.getElementById("dest-plan"),
    insights: document.getElementById("dest-insights"),
  },
  // Every [data-dest] button across the sidebar, bottom tab bar and more-sheet.
  destItems: Array.from(document.querySelectorAll("[data-dest]")),

  // Plan (meal plan | nutrition) and Insights (analytics | history) sub-panels,
  // toggled by their [data-sub] segmented controls.
  subItems: Array.from(document.querySelectorAll("[data-sub]")),
  panelPlan: document.getElementById("panel-plan"),
  panelNutrition: document.getElementById("panel-nutrition"),
  panelAnalytics: document.getElementById("panel-analytics"),
  panelHistory: document.getElementById("panel-history"),

  // Scan source (Upload | Webcam) merged into one destination via a segmented control.
  scanModeItems: Array.from(document.querySelectorAll("#dest-scan [data-mode]")),
  panelUpload: document.getElementById("panel-upload"),
  panelWebcam: document.getElementById("panel-webcam"),

  // Theme toggles (laptop + phone), the phone "more" sheet, and the frost canvas.
  themeToggle: document.getElementById("theme-toggle"),
  themeToggleM: document.getElementById("theme-toggle-m"),
  menuBtn: document.getElementById("menu-btn"),
  moreSheet: document.getElementById("more-sheet"),
  sheetOverlay: document.getElementById("sheet-overlay"),
  frostCanvas: document.getElementById("frost-canvas"),

  // Cart count badges on the Cart nav item (sidebar + bottom bar).
  navCartCount: document.getElementById("nav-cart-count"),
  tabbarCartCount: document.getElementById("tabbar-cart-count"),

  // Real-time status indicator + transient "just updated" toast.
  liveStatus: document.getElementById("live-status"),
  liveLabel: document.getElementById("live-label"),
  liveToast: document.getElementById("live-toast"),

  // Wrapper (display:contents) around the whole capture flow (preview/loading/results).
  // It lives inside the Scan destination, so it's naturally scoped to that view.
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
  planBudget: document.getElementById("plan-budget"),
  planBudgetCur: document.getElementById("plan-budget-cur"),
  planBuild: document.getElementById("plan-build"),
  planWeek: document.getElementById("plan-week"),
  planLoading: document.getElementById("plan-loading"),
  planError: document.getElementById("plan-error"),
  planBody: document.getElementById("plan-body"),
  planMetrics: document.getElementById("plan-metrics"),
  planNotes: document.getElementById("plan-notes"),
  planEmpty: document.getElementById("plan-empty"),
  planBydaySection: document.getElementById("plan-byday-section"),
  planByday: document.getElementById("plan-byday"),
  planMealsList: document.getElementById("plan-meals-list"),
  planShoppingSection: document.getElementById("plan-shopping-section"),
  planShoppingCount: document.getElementById("plan-shopping-count"),
  planShoppingAddall: document.getElementById("plan-shopping-addall"),
  planShoppingAdded: document.getElementById("plan-shopping-added"),
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
  chefMode: document.getElementById("chef-mode"),
  chefWaiting: document.getElementById("chef-waiting"),
  chefWaitingList: document.getElementById("chef-waiting-list"),

  // Chef's memory + shopping list (Chef tab side cards).
  memDiet: document.getElementById("mem-diet"),
  memAllergies: document.getElementById("mem-allergies"),
  memDislikes: document.getElementById("mem-dislikes"),
  memAllergyForm: document.getElementById("mem-allergy-form"),
  memDislikeForm: document.getElementById("mem-dislike-form"),
  memOther: document.getElementById("mem-other"),
  memClear: document.getElementById("mem-clear"),
  memError: document.getElementById("mem-error"),
  shopForm: document.getElementById("shop-form"),
  shopInput: document.getElementById("shop-input"),
  shopMsg: document.getElementById("shop-msg"),
  shopList: document.getElementById("shop-list"),
  shopEmpty: document.getElementById("shop-empty"),
  shopCount: document.getElementById("shop-count"),
  shopClear: document.getElementById("shop-clear"),

  // Smart restock card (Cart tab): staples predicted low from usage history.
  restockCard: document.getElementById("restock-card"),
  restockCount: document.getElementById("restock-count"),
  restockAddall: document.getElementById("restock-addall"),
  restockHeadline: document.getElementById("restock-headline"),
  restockAdded: document.getElementById("restock-added"),
  restockList: document.getElementById("restock-list"),

  // Receipt import (Scan tab): paste receipt text → preview groceries → add to fridge.
  receiptForm: document.getElementById("receipt-form"),
  receiptText: document.getElementById("receipt-text"),
  receiptPreview: document.getElementById("receipt-preview"),
  receiptClear: document.getElementById("receipt-clear"),
  receiptError: document.getElementById("receipt-error"),
  receiptAdded: document.getElementById("receipt-added"),
  receiptResult: document.getElementById("receipt-result"),
  receiptCount: document.getElementById("receipt-count"),
  receiptSummary: document.getElementById("receipt-summary"),
  receiptList: document.getElementById("receipt-list"),
  receiptImport: document.getElementById("receipt-import"),
  receiptImportHint: document.getElementById("receipt-import-hint"),

  // Chef's proactive briefing (My Fridge tab).
  briefing: document.getElementById("briefing"),
  briefingWhen: document.getElementById("briefing-when"),
  briefingHeadline: document.getElementById("briefing-headline"),
  briefingLines: document.getElementById("briefing-lines"),
  briefingActions: document.getElementById("briefing-actions"),
  briefingDismiss: document.getElementById("briefing-dismiss"),
  briefingCheck: document.getElementById("briefing-check"),
  briefingError: document.getElementById("briefing-error"),

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
// The Chef tab's memory + shopping cards load on first open, then follow live updates.
let chefLoadedOnce = false;
// The live Socket.IO connection (null when real-time is unavailable). The Chef streams its
// working steps to this tab over it while an answer is being prepared.
let liveSocket = null;
// The Chef run currently streaming steps into its "thinking" bubble.
let activeChefRun = null;

// Which destinations the user has opened at least once. A real-time "state_changed"
// push only re-fetches a view if it has actually been loaded, so a background update
// never eagerly populates a view the user has never visited (the lazy-load model).
const destsSeen = new Set(["fridge"]);

// The current sub-view within Plan (meal plan | nutrition) and Insights (analytics |
// history), and the current Scan source (upload | webcam). These persist across
// destination switches so returning to Plan/Insights/Scan restores the last sub-view.
const activeSub = { plan: "plan", insights: "analytics" };
let scanMode = "upload";
// Handle for the frost-particle animation loop (so we can pause it when the tab hides).
let frostRaf = null;

// --- Destination switching --------------------------------------------------
// One entry point drives every nav surface. It shows the chosen destination, mirrors
// the selection onto the sidebar / bottom bar / more-sheet, scopes the cameras to
// where they live, and lazily loads each view's data the same way the old tabs did.
function activateDest(which) {
  if (!els.dests[which]) which = "fridge";
  destsSeen.add(which);

  for (const [name, node] of Object.entries(els.dests)) {
    if (node) node.hidden = name !== which;
  }
  for (const item of els.destItems) {
    item.classList.toggle("active", item.dataset.dest === which);
  }
  closeMoreSheet();

  // Cameras are scoped to their home view: the webcam to Scan→Webcam, the barcode
  // scanner to Plan→Nutrition. Leaving either releases the stream.
  if (!(which === "scan" && scanMode === "webcam")) stopCamera();
  if (!(which === "plan" && activeSub.plan === "nutrition")) stopBarcodeScan();

  switch (which) {
    case "fridge":
      loadFridge();
      loadBriefing();
      break;
    case "recipes":
      loadPerishables();
      loadSuggestions();
      break;
    case "scan":
      applyScanMode();
      break;
    case "cart":
      loadShopping();
      loadRestock();
      break;
    case "chef":
      if (!chefLoadedOnce) {
        chefLoadedOnce = true;
        loadMemory();
      }
      syncActions();
      els.chefInput.focus();
      break;
    case "plan":
      applySub("plan");
      break;
    case "insights":
      applySub("insights");
      break;
  }
}

// Sub-navigation inside Plan and Insights. `sub` (optional) switches the active view;
// with no argument it just (re)applies the current one — used when entering the dest.
function applySub(group, sub) {
  if (sub) activeSub[group] = sub;
  const current = activeSub[group];

  if (group === "plan") {
    els.panelPlan.hidden = current !== "plan";
    els.panelNutrition.hidden = current !== "nutrition";
    if (current !== "nutrition") stopBarcodeScan(); // scanner lives in Nutrition
    if (current === "plan" && !planLoadedOnce) {
      // Solve automatically on first open; afterwards the user rebuilds explicitly.
      planLoadedOnce = true;
      loadPlan();
    } else if (current === "nutrition" && !nutritionLoadedOnce) {
      nutritionLoadedOnce = true;
      loadNutrition();
    }
  } else if (group === "insights") {
    els.panelAnalytics.hidden = current !== "analytics";
    els.panelHistory.hidden = current !== "history";
    if (current === "analytics") {
      if (!analyticsLoadedOnce) {
        analyticsLoadedOnce = true;
        loadAnalytics();
      }
    } else if (current === "history") {
      loadHistory();
    }
  }

  for (const item of els.subItems) {
    if (item.dataset.destSub !== group) continue;
    const on = item.dataset.sub === current;
    item.classList.toggle("active", on);
    item.setAttribute("aria-selected", String(on));
  }
}

// Scan source (Upload | Webcam) — merged into one destination via a segmented control.
function setScanMode(mode) {
  scanMode = mode === "webcam" ? "webcam" : "upload";
  applyScanMode();
}

function applyScanMode() {
  els.panelUpload.hidden = scanMode !== "upload";
  els.panelWebcam.hidden = scanMode !== "webcam";
  if (scanMode !== "webcam") stopCamera();
  for (const item of els.scanModeItems) {
    const on = item.dataset.mode === scanMode;
    item.classList.toggle("active", on);
    item.setAttribute("aria-selected", String(on));
  }
  els.captureExtras.hidden = false;
}

// --- Phone "more" sheet (Plan / Insights live here on small screens) --------
function openMoreSheet() {
  if (els.moreSheet) els.moreSheet.hidden = false;
  if (els.sheetOverlay) els.sheetOverlay.hidden = false;
  if (els.menuBtn) els.menuBtn.setAttribute("aria-expanded", "true");
}
function closeMoreSheet() {
  if (els.moreSheet) els.moreSheet.hidden = true;
  if (els.sheetOverlay) els.sheetOverlay.hidden = true;
  if (els.menuBtn) els.menuBtn.setAttribute("aria-expanded", "false");
}

// --- Theme (manual light/dark toggle, persisted; light by default) ----------
function updateThemeColor() {
  const meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) return;
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  meta.setAttribute("content", dark ? "#071523" : "#eaf6ff");
}
function initTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem("frost-theme");
  } catch (_e) {
    /* storage blocked (private mode) — fall back to the light default */
  }
  if (saved === "dark" || saved === "light") {
    document.documentElement.setAttribute("data-theme", saved);
  }
  const toggle = () => {
    const next =
      document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("frost-theme", next);
    } catch (_e) {
      /* ignore — the toggle still works for this session */
    }
    updateThemeColor();
  };
  if (els.themeToggle) els.themeToggle.addEventListener("click", toggle);
  if (els.themeToggleM) els.themeToggleM.addEventListener("click", toggle);
  updateThemeColor();
}

// --- Frost particles (ambient snow drifting behind the glass) ---------------
function initFrost() {
  const canvas = els.frostCanvas;
  if (!canvas || !canvas.getContext) return;
  const reduce =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) return; // honour the OS "reduce motion" setting — no particles at all
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let w = 0;
  let h = 0;
  let flakes = [];
  const COUNT = 70;

  function resize() {
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function make() {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 2.2 + 0.6,
      vy: Math.random() * 0.5 + 0.15,
      vx: (Math.random() - 0.5) * 0.35,
      a: Math.random() * 0.5 + 0.25,
      drift: Math.random() * Math.PI * 2,
    };
  }
  function tick() {
    ctx.clearRect(0, 0, w, h);
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    ctx.fillStyle = dark ? "#bcdcff" : "#7fb8ff";
    for (const f of flakes) {
      f.drift += 0.01;
      f.y += f.vy;
      f.x += f.vx + Math.sin(f.drift) * 0.25;
      if (f.y > h + 5) {
        f.y = -5;
        f.x = Math.random() * w;
      }
      if (f.x > w + 5) f.x = -5;
      else if (f.x < -5) f.x = w + 5;
      ctx.globalAlpha = f.a;
      ctx.beginPath();
      ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    frostRaf = requestAnimationFrame(tick);
  }

  resize();
  flakes = Array.from({ length: COUNT }, make);
  window.addEventListener("resize", resize);
  // Pause the loop while the tab is hidden, resume when it returns.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (frostRaf) cancelAnimationFrame(frostRaf);
      frostRaf = null;
    } else if (!frostRaf) {
      frostRaf = requestAnimationFrame(tick);
    }
  });
  frostRaf = requestAnimationFrame(tick);
}

// Keep the Cart nav badges (sidebar + bottom bar) in sync with the open-item count.
function updateCartBadges(count) {
  for (const badge of [els.navCartCount, els.tabbarCartCount]) {
    if (!badge) continue;
    badge.hidden = count <= 0;
    badge.textContent = String(count);
  }
}

// Wire every nav surface with delegated handlers (they all share [data-*] attributes).
for (const item of els.destItems) {
  item.addEventListener("click", () => activateDest(item.dataset.dest));
}
for (const item of els.subItems) {
  item.addEventListener("click", () => applySub(item.dataset.destSub, item.dataset.sub));
}
for (const item of els.scanModeItems) {
  item.addEventListener("click", () => setScanMode(item.dataset.mode));
}
if (els.menuBtn) els.menuBtn.addEventListener("click", openMoreSheet);
if (els.sheetOverlay) els.sheetOverlay.addEventListener("click", closeMoreSheet);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeMoreSheet();
});

els.refreshFridge.addEventListener("click", loadFridge);
els.refreshHistory.addEventListener("click", loadHistory);
els.refreshSuggestions.addEventListener("click", loadSuggestions);
els.perishForm.addEventListener("submit", addPerishable);
els.recipeSearchForm.addEventListener("submit", searchRecipes);
els.fridgeScanCta.addEventListener("click", () => activateDest("scan"));
els.fridgeToggle.addEventListener("click", () => setFridgeItemsOpen(!fridgeItemsOpen));
els.planBuild.addEventListener("click", () => loadPlan());
if (els.planWeek) els.planWeek.addEventListener("click", () => loadPlan({ autopilot: true }));
if (els.planShoppingAddall)
  els.planShoppingAddall.addEventListener("click", addPlanShoppingToList);
els.chefForm.addEventListener("submit", askChef);
els.memDiet.addEventListener("change", () => saveMemory("diet", els.memDiet.value));
els.memAllergyForm.addEventListener("submit", addMemoryFromForm);
els.memDislikeForm.addEventListener("submit", addMemoryFromForm);
els.memClear.addEventListener("click", clearMemory);
els.shopForm.addEventListener("submit", addShoppingItem);
els.shopClear.addEventListener("click", clearBoughtShopping);
if (els.restockAddall) els.restockAddall.addEventListener("click", addAllRestock);
if (els.receiptForm) {
  els.receiptForm.addEventListener("submit", previewReceipt);
  els.receiptClear.addEventListener("click", clearReceipt);
  els.receiptImport.addEventListener("click", importReceipt);
}
els.briefingCheck.addEventListener("click", checkBriefingNow);
els.briefingDismiss.addEventListener("click", dismissBriefing);
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

  // "Cook it" → opens Cook Mode (guided steps, timers, voice, swap ideas).
  if (recipe.id) {
    const actions = document.createElement("div");
    actions.className = "recipe-card-actions";
    const cookBtn = document.createElement("button");
    cookBtn.type = "button";
    cookBtn.className = "btn-cook";
    cookBtn.textContent = "👨‍🍳 Cook it";
    cookBtn.addEventListener("click", () => openCookMode(recipe.id));
    actions.appendChild(cookBtn);
    card.appendChild(actions);
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
  if (r.id) {
    const cookBtn = document.createElement("button");
    cookBtn.type = "button";
    cookBtn.className = "btn-cook";
    cookBtn.textContent = "👨‍🍳 Cook it";
    cookBtn.addEventListener("click", () => openCookMode(r.id));
    actions.appendChild(cookBtn);
  }
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
// Remember the last plan so "Add all to my list" can post its shopping list.
let lastPlanShopping = [];

async function loadPlan(options) {
  const opts = options || {};
  // Autopilot week: a fixed 7-day, 3-meal horizon regardless of the controls.
  const autopilot = opts.autopilot === true;
  const days = autopilot ? 7 : parseInt(els.planDays.value, 10) || 3;
  const mealsPerDay = autopilot ? 3 : parseInt(els.planMeals.value, 10) || 2;

  // Reflect the horizon in the controls so the UI stays truthful in autopilot mode.
  if (autopilot) {
    if (els.planDays) els.planDays.value = "7";
    if (els.planMeals) els.planMeals.value = "3";
  }

  const budget = parseBudgetInput();
  const body = { days, meals_per_day: mealsPerDay };
  if (budget != null) body.budget = budget;

  els.planError.hidden = true;
  els.planBody.hidden = true;
  els.planLoading.hidden = false;
  if (els.planBuild) els.planBuild.disabled = true;
  if (els.planWeek) els.planWeek.disabled = true;

  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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
    if (els.planBuild) els.planBuild.disabled = false;
    if (els.planWeek) els.planWeek.disabled = false;
  }
}

// Read the optional budget cap. Blank / non-numeric → null (no cap); negative clamps to 0.
function parseBudgetInput() {
  if (!els.planBudget) return null;
  const raw = String(els.planBudget.value || "").trim();
  if (!raw) return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, n);
}

// Format a money amount with the plan's currency symbol (defaults to ₹).
function formatMoney(amount, currency) {
  const sym = currency || "₹";
  const n = Number(amount);
  if (!Number.isFinite(n)) return "";
  // Whole numbers read cleaner without trailing ".0"; keep cents otherwise.
  const body = Number.isInteger(n) ? String(n) : n.toFixed(2);
  return `${sym}${body}`;
}

function renderPlan(data) {
  const plan = Array.isArray(data.plan) ? data.plan : [];
  const metrics = data.metrics || {};
  const notes = Array.isArray(data.notes) ? data.notes : [];
  const shopping = Array.isArray(data.shopping_list) ? data.shopping_list : [];
  const atRisk = Array.isArray(data.at_risk_remaining) ? data.at_risk_remaining : [];
  const byDay = Array.isArray(data.by_day) ? data.by_day : [];
  const currency = metrics.currency || "₹";
  const priced = Number.isFinite(metrics.shopping_cost);

  // Keep the budget field's currency hint in step with the server's currency.
  if (els.planBudgetCur) els.planBudgetCur.textContent = `(${currency}, optional)`;

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

  // --- Cost tiles (only when the plan is priced) ---
  if (priced) {
    // Shopping cost — highlighted green within budget, amber when over.
    let costClass = "plan-stat-cost";
    let costLabel = "Est. shopping cost";
    if (Number.isFinite(metrics.budget)) {
      const within = metrics.within_budget !== false;
      costClass += within ? " plan-stat-ok" : " plan-stat-over";
      costLabel = within
        ? `Within ${formatMoney(metrics.budget, currency)} budget`
        : `Over ${formatMoney(metrics.budget, currency)} budget`;
    }
    els.planMetrics.appendChild(
      buildStat(formatMoney(metrics.shopping_cost, currency), costLabel, costClass)
    );
    if (Number.isFinite(metrics.pantry_value_used) && metrics.pantry_value_used > 0) {
      els.planMetrics.appendChild(
        buildStat(
          formatMoney(metrics.pantry_value_used, currency),
          "Pantry value used",
          "plan-stat-pantry"
        )
      );
    }
  }

  // --- Notes ---
  els.planNotes.innerHTML = "";
  for (const note of notes) {
    const p = document.createElement("p");
    p.className = "plan-note";
    p.textContent = note;
    els.planNotes.appendChild(p);
  }

  // --- Day-by-day breakdown (Autopilot week) ---
  els.planByday.innerHTML = "";
  const showByDay = byDay.length > 1;
  els.planBydaySection.hidden = !showByDay;
  if (showByDay) {
    for (const day of byDay) {
      els.planByday.appendChild(buildPlanDayCard(day, currency));
    }
  }

  // --- Meal cards / empty state ---
  els.planMealsList.innerHTML = "";
  els.planEmpty.hidden = plan.length > 0;
  for (const meal of plan) {
    els.planMealsList.appendChild(buildPlanMealCard(meal, currency));
  }

  // --- Shopping list ---
  lastPlanShopping = shopping;
  els.planShopping.innerHTML = "";
  els.planShoppingSection.hidden = shopping.length === 0;
  els.planShoppingCount.textContent = String(shopping.length);
  if (els.planShoppingAdded) els.planShoppingAdded.hidden = true;
  if (els.planShoppingAddall) {
    els.planShoppingAddall.disabled = false;
    els.planShoppingAddall.textContent = "Add all to my list";
  }
  for (const s of shopping) {
    els.planShopping.appendChild(buildPlanShoppingRow(s, currency));
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

function buildPlanMealCard(meal, currency) {
  const card = document.createElement("div");
  card.className = "card plan-meal-card";

  const head = document.createElement("div");
  head.className = "card-head";
  const title = document.createElement("h3");
  title.className = "card-name";
  const slot = Number.isFinite(meal.slot) ? meal.slot : "";
  title.textContent = slot ? `Meal ${slot}: ${meal.title || "Recipe"}` : (meal.title || "Recipe");
  head.appendChild(title);
  const meta = document.createElement("div");
  meta.className = "card-head-meta";
  if (Number.isFinite(meal.time_min)) {
    const time = document.createElement("span");
    time.className = "recipe-time muted small";
    time.textContent = `⏱ ${meal.time_min} min`;
    meta.appendChild(time);
  }
  // Per-meal top-up cost (0 when everything is already on hand).
  if (Number.isFinite(meal.est_cost)) {
    const cost = document.createElement("span");
    cost.className = meal.est_cost > 0 ? "meal-cost" : "meal-cost meal-cost-free";
    cost.textContent = meal.est_cost > 0 ? `+${formatMoney(meal.est_cost, currency)}` : "on hand";
    meta.appendChild(cost);
  }
  if (meta.childElementCount) head.appendChild(meta);
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

function buildPlanShoppingRow(s, currency) {
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

  // Priced plans show the per-item estimate on the right.
  if (Number.isFinite(s.est_cost)) {
    const price = document.createElement("span");
    price.className = "shopping-price";
    price.textContent = formatMoney(s.est_cost, currency);
    row.appendChild(price);
  }
  return row;
}

// One day's card in the Autopilot week breakdown: its meals + the day's top-up cost.
function buildPlanDayCard(day, currency) {
  const card = document.createElement("div");
  card.className = "card plan-day-card";

  const head = document.createElement("div");
  head.className = "plan-day-head";
  const label = document.createElement("h4");
  label.className = "plan-day-label";
  label.textContent = `Day ${day.day}`;
  head.appendChild(label);
  if (Number.isFinite(day.est_cost)) {
    const cost = document.createElement("span");
    cost.className = day.est_cost > 0 ? "plan-day-cost" : "plan-day-cost meal-cost-free";
    cost.textContent = day.est_cost > 0 ? `+${formatMoney(day.est_cost, currency)}` : "on hand";
    head.appendChild(cost);
  }
  card.appendChild(head);

  const meals = Array.isArray(day.meals) ? day.meals : [];
  const list = document.createElement("ul");
  list.className = "plan-day-meals";
  for (const meal of meals) {
    const li = document.createElement("li");
    li.className = "plan-day-meal";
    const name = document.createElement("span");
    name.className = "plan-day-meal-name";
    name.textContent = meal.title || "Recipe";
    li.appendChild(name);
    if (Number.isFinite(meal.est_cost) && meal.est_cost > 0) {
      const mc = document.createElement("span");
      mc.className = "plan-day-meal-cost muted small";
      mc.textContent = `+${formatMoney(meal.est_cost, currency)}`;
      li.appendChild(mc);
    }
    list.appendChild(li);
  }
  card.appendChild(list);
  return card;
}

// "Add all to my list" — post every planned shopping item to the shared list in one call.
async function addPlanShoppingToList() {
  const items = (lastPlanShopping || [])
    .map((s) => (s && s.item ? String(s.item).trim() : ""))
    .filter(Boolean);
  if (!items.length) return;

  const btn = els.planShoppingAddall;
  const note = els.planShoppingAdded;
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Adding…";
  }
  if (note) note.hidden = true;

  try {
    const response = await fetch("/api/shopping", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: items.map((name) => ({ name })) }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      if (note) {
        note.hidden = false;
        note.classList.add("cook-error");
        note.textContent = (payload && payload.error) || "Couldn't add those items.";
      }
      if (btn) {
        btn.disabled = false;
        btn.textContent = "Add all to my list";
      }
      return;
    }
    const added = Array.isArray(payload && payload.added) ? payload.added.length : items.length;
    const skipped = Number.isFinite(payload && payload.skipped) ? payload.skipped : 0;
    if (btn) btn.textContent = "✓ Added";
    if (note) {
      note.hidden = false;
      note.classList.remove("cook-error");
      note.textContent = skipped
        ? `Added ${added} item${added === 1 ? "" : "s"} · ${skipped} already on your list.`
        : `Added ${added} item${added === 1 ? "" : "s"} to your shopping list.`;
    }
    // Keep the Cart tab's list in sync with the server.
    loadShopping();
  } catch (_err) {
    if (note) {
      note.hidden = false;
      note.classList.add("cook-error");
      note.textContent = "Couldn't reach the server.";
    }
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Add all to my list";
    }
  }
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
// Chef works in a loop: look things up with tools, decide, act. Reversible things
// (remembering a preference, adding to the shopping list) it just does; anything that
// changes the fridge comes back as an action card that waits for the user's Confirm.

const ACTION_ICONS = {
  cook_meal: "🍳",
  discard_items: "🗑️",
  track_expiry: "📅",
  add_shopping: "🛒",
};

const ACTION_STATUS_TEXT = {
  pending: "Waiting for your OK",
  running: "Working…",
  done: "✓ Done",
  cancelled: "Cancelled",
  expired: "Out of date — ask Chef again",
  failed: "Didn't work",
};

// Action ids already shown as cards in the chat, so "Waiting for your OK" doesn't repeat them.
const inlineActionIds = new Set();
// Chat suggestions still waiting for an OK (from the server), for the "Waiting" box.
let waitingActions = [];

function newRunId() {
  const rand = Math.random().toString(36).slice(2, 10);
  return `run-${Date.now().toString(36)}-${rand}`;
}

function isLive() {
  return Boolean(liveSocket && liveSocket.connected);
}

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
  const body = { message, history: chefHistory.slice() };
  // Over a live connection, the server streams Chef's steps to this tab while it works.
  if (isLive() && liveSocket.id) {
    body.run_id = newRunId();
    body.sid = liveSocket.id;
    activeChefRun = { id: body.run_id, bubble: pending, open: {} };
  }

  try {
    const response = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    activeChefRun = null;
    pending.remove();

    if (!response.ok) {
      const msg = (payload && payload.error) || `The Chef returned an error (HTTP ${response.status}).`;
      els.chefError.textContent = msg;
      els.chefError.hidden = false;
      return;
    }

    const answer = (payload && payload.answer) || "(no answer)";
    appendChefBubble("assistant", answer, payload);
    setChefMode(payload && payload.mode);

    // Without a live connection nobody tells this tab what Chef changed — refresh by hand.
    if (!isLive() && payload) {
      refreshScopes([...(payload.changed || []), "agent"]);
    }

    // Keep a bounded running transcript for follow-up context.
    chefHistory.push({ role: "user", content: message });
    chefHistory.push({ role: "assistant", content: answer });
    while (chefHistory.length > 10) chefHistory.shift();
  } catch (_err) {
    activeChefRun = null;
    pending.remove();
    els.chefError.textContent = "Couldn't reach the server to ask the Chef.";
    els.chefError.hidden = false;
  } finally {
    chefBusy = false;
    els.chefSend.disabled = false;
    els.chefInput.disabled = false;
    els.chefInput.focus();
    renderWaiting();
  }
}

// Chat models like **bold**; show it as bold instead of raw asterisks. Built from text nodes
// (never innerHTML), so nothing the model writes can inject markup.
function appendRichText(el, text) {
  for (const part of String(text || "").split(/(\*\*[^*\n]+\*\*)/)) {
    if (!part) continue;
    if (/^\*\*[^*\n]+\*\*$/.test(part)) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      el.appendChild(strong);
    } else {
      el.appendChild(document.createTextNode(part));
    }
  }
}

function appendChefBubble(role, text, payload) {
  const bubble = document.createElement("div");
  bubble.className = `chef-msg chef-msg-${role}`;

  // Chef says so when it had to fall back to its simpler, rules-based brain.
  if (role === "assistant" && payload && payload.note) {
    const note = document.createElement("p");
    note.className = "chef-note small";
    note.textContent = `⚠️ ${payload.note}`;
    bubble.appendChild(note);
  }

  const body = document.createElement("p");
  body.className = "chef-msg-text";
  if (role === "assistant") appendRichText(body, text);
  else body.textContent = text;
  bubble.appendChild(body);

  if (role === "assistant" && payload) {
    // Anything that would change the fridge waits here for the user's OK.
    const actions = Array.isArray(payload.actions) ? payload.actions : [];
    if (actions.length) {
      const list = document.createElement("div");
      list.className = "action-list chef-actions";
      for (const action of actions) {
        inlineActionIds.add(action.id);
        list.appendChild(renderActionCard(action));
      }
      bubble.appendChild(list);
    }

    // Which tools it used (transparency), collapsed by default.
    const steps = Array.isArray(payload.steps) ? payload.steps : [];
    if (steps.length) {
      const details = document.createElement("details");
      details.className = "chef-steps";
      const summary = document.createElement("summary");
      const via = payload.mode === "offline" ? "simple mode" : payload.model;
      summary.textContent = `How Chef worked it out · ${steps.length} step${steps.length === 1 ? "" : "s"}${via ? " · " + via : ""}`;
      details.appendChild(summary);
      for (const step of steps) {
        const failed = step.result && typeof step.result === "object" && step.result.error;
        const line = document.createElement("div");
        line.className = "chef-step muted small";
        line.textContent = `${failed ? "⚠" : "✓"} ${step.tool}(${step.arguments ? JSON.stringify(step.arguments) : ""})`;
        details.appendChild(line);
      }
      bubble.appendChild(details);
    }
  }

  els.chefLog.appendChild(bubble);
  els.chefLog.scrollTop = els.chefLog.scrollHeight;
  return bubble;
}

function appendChefPending() {
  const bubble = document.createElement("div");
  bubble.className = "chef-msg chef-msg-assistant chef-msg-pending";
  const head = document.createElement("div");
  head.className = "chef-pending-head";
  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.setAttribute("aria-hidden", "true");
  const label = document.createElement("span");
  label.className = "chef-pending-label";
  label.textContent = "The Chef is thinking…";
  head.append(spinner, label);
  // Filled live from "agent_step" events: one line per tool Chef runs.
  const steps = document.createElement("ul");
  steps.className = "chef-live-steps";
  steps.hidden = true;
  bubble.append(head, steps);
  els.chefLog.appendChild(bubble);
  els.chefLog.scrollTop = els.chefLog.scrollHeight;
  return bubble;
}

// One live progress event from the server while Chef works on this tab's question.
function handleAgentStep(event) {
  const run = activeChefRun;
  if (!run || !event || event.run_id !== run.id) return;
  const label = run.bubble.querySelector(".chef-pending-label");

  if (event.phase === "thinking") {
    if (!label) return;
    if (event.round === "final") label.textContent = "Writing the answer…";
    else if (event.round > 1) label.textContent = "Thinking about what it found…";
    else label.textContent = "Reading your question…";
    return;
  }
  if (event.phase !== "tool") return;

  const list = run.bubble.querySelector(".chef-live-steps");
  if (!list) return;
  const text = event.label || event.tool || "Working";
  if (event.status === "running") {
    const item = document.createElement("li");
    item.className = "chef-live-step";
    item.dataset.state = "running";
    item.textContent = `${text}…`;
    list.appendChild(item);
    list.hidden = false;
    run.open[event.tool] = item;
    if (label) label.textContent = "Working on it…";
  } else {
    const item = run.open[event.tool] || list.appendChild(document.createElement("li"));
    delete run.open[event.tool];
    item.className = "chef-live-step";
    item.dataset.state = event.status === "error" ? "error" : "done";
    item.textContent = text;
    list.hidden = false;
  }
  els.chefLog.scrollTop = els.chefLog.scrollHeight;
}

function setChefMode(mode) {
  if (!els.chefMode || !mode) return;
  const offline = mode === "offline";
  els.chefMode.hidden = false;
  els.chefMode.dataset.mode = offline ? "offline" : "ai";
  els.chefMode.textContent = offline ? "Simple mode" : "AI mode";
  els.chefMode.title = offline
    ? "No AI model is reachable, so Chef is using built-in rules — still with your real fridge data."
    : "Chef is using an AI model with your real fridge data.";
}

// --- Chef's suggested changes (confirm-before-change) -------------------------
function actionIcon(action) {
  if (action.tool === "resolve_perishable") {
    return action.arguments && action.arguments.event === "wasted" ? "🗑️" : "✅";
  }
  return ACTION_ICONS[action.tool] || "🧑‍🍳";
}

function renderActionCard(action) {
  const card = document.createElement("div");
  card.className = "action-card";
  card.dataset.actionId = String(action.id);

  const icon = document.createElement("span");
  icon.className = "action-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = actionIcon(action);

  const main = document.createElement("div");
  main.className = "action-main";
  const summary = document.createElement("p");
  summary.className = "action-summary";
  summary.textContent = action.summary || action.tool;
  const status = document.createElement("p");
  status.className = "action-status small";
  main.append(summary, status);

  const buttons = document.createElement("div");
  buttons.className = "action-buttons";
  const confirmBtn = document.createElement("button");
  confirmBtn.type = "button";
  confirmBtn.className = "btn btn-primary btn-sm";
  confirmBtn.textContent = "Confirm";
  confirmBtn.addEventListener("click", () => confirmAction(action.id));
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "btn btn-ghost btn-sm";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", () => cancelAction(action.id));
  buttons.append(confirmBtn, cancelBtn);

  card.append(icon, main, buttons);
  applyActionStatus(card, action.status || "pending", actionError(action));
  return card;
}

function actionError(action) {
  const result = action && action.result;
  return action && action.status === "failed" && result && result.error ? String(result.error) : "";
}

function applyActionStatus(card, status, detail) {
  card.dataset.status = status;
  const text = card.querySelector(".action-status");
  if (text) {
    const base = ACTION_STATUS_TEXT[status] || status;
    text.textContent = detail ? (status === "failed" ? `${base}: ${detail}` : detail) : base;
  }
  const buttons = card.querySelector(".action-buttons");
  if (buttons) {
    buttons.hidden = status !== "pending";
    for (const btn of buttons.querySelectorAll("button")) btn.disabled = status !== "pending";
  }
}

// The same suggestion can be on screen twice (chat + waiting box) — update every copy.
function setActionStatus(id, status, detail) {
  for (const card of document.querySelectorAll(`[data-action-id="${Number(id)}"]`)) {
    applyActionStatus(card, status, detail);
  }
}

async function confirmAction(id) {
  setActionStatus(id, "running");
  try {
    const response = await fetch(`/api/agent/actions/${Number(id)}/confirm`, { method: "POST" });
    const payload = await response.json().catch(() => null);
    if (response.ok) {
      setActionStatus(id, "done");
      if (!isLive()) refreshScopes(["inventory", "perishables", "analytics", "shopping", "agent", "briefing"]);
      return;
    }
    const msg = (payload && payload.error) || `Couldn't do that (HTTP ${response.status}).`;
    if (response.status === 409) {
      // Already handled elsewhere (another tab / double click) or out of date.
      const current = payload && payload.action;
      if (current && current.status) setActionStatus(id, current.status, actionError(current));
      else setActionStatus(id, "expired", msg);
    } else {
      setActionStatus(id, "failed", msg);
    }
  } catch (_err) {
    setActionStatus(id, "pending", "Couldn't reach the server — try again.");
  }
}

async function cancelAction(id) {
  setActionStatus(id, "running", "Cancelling…");
  try {
    const response = await fetch(`/api/agent/actions/${Number(id)}/cancel`, { method: "POST" });
    if (response.ok) {
      setActionStatus(id, "cancelled");
      if (!isLive()) syncActions();
      return;
    }
    const payload = await response.json().catch(() => null);
    setActionStatus(id, "pending", (payload && payload.error) || "Couldn't cancel that.");
    syncActions(); // it may have been handled elsewhere — show its real state
  } catch (_err) {
    setActionStatus(id, "pending", "Couldn't reach the server — try again.");
  }
}

// Re-read Chef's recent suggestions: refresh every card's status and the "Waiting" box.
async function syncActions() {
  let actions;
  try {
    const response = await fetch("/api/agent/actions?status=all");
    if (!response.ok) return;
    const payload = await response.json();
    actions = Array.isArray(payload.actions) ? payload.actions : [];
  } catch (_err) {
    return;
  }
  for (const action of actions) {
    for (const card of document.querySelectorAll(`[data-action-id="${Number(action.id)}"]`)) {
      // Don't flip a card back to "pending" while this tab's own click is in flight.
      if (card.dataset.status === "running" && action.status === "pending") continue;
      applyActionStatus(card, action.status, actionError(action));
    }
  }
  waitingActions = actions.filter((a) => a.status === "pending" && a.source !== "briefing");
  renderWaiting();
}

function renderWaiting() {
  if (!els.chefWaiting || chefBusy) return; // an answer in flight will show its own cards
  const list = waitingActions.filter((a) => !inlineActionIds.has(a.id));
  els.chefWaitingList.innerHTML = "";
  for (const action of list) els.chefWaitingList.appendChild(renderActionCard(action));
  els.chefWaiting.hidden = list.length === 0;
}

// --- Chef's memory (household preferences) ------------------------------------
const MEMORY_LABELS = {
  household_size: "Household size",
  favorite_cuisines: "Favourite cuisines",
  notes: "Notes",
  spice_level: "Spice level",
  goal: "Goal",
};
let memoryDiets = [];
let memClearArmed = null;

async function loadMemory() {
  try {
    const response = await fetch("/api/agent/memory");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showMemError((payload && payload.error) || "Couldn't load your preferences.");
      return;
    }
    renderMemory(payload);
  } catch (_err) {
    showMemError("Couldn't reach the server to load your preferences.");
  }
}

function renderMemory(payload) {
  const memory = (payload && payload.memory) || {};
  if (payload && Array.isArray(payload.diets)) memoryDiets = payload.diets;

  els.memDiet.innerHTML = "";
  const choices = [["", "Not set"], ...memoryDiets.map((d) => [d, d.charAt(0).toUpperCase() + d.slice(1)])];
  for (const [value, text] of choices) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    els.memDiet.appendChild(option);
  }
  els.memDiet.value = typeof memory.diet === "string" ? memory.diet : "";

  renderChips(els.memAllergies, "allergies", memory.allergies);
  renderChips(els.memDislikes, "dislikes", memory.dislikes);

  // Anything else Chef picked up in conversation (household size, spice level, ...).
  els.memOther.innerHTML = "";
  for (const [key, value] of Object.entries(memory)) {
    if (key === "diet" || key === "allergies" || key === "dislikes") continue;
    const row = document.createElement("div");
    row.className = "mem-other-row";
    const name = document.createElement("span");
    name.className = "mem-label";
    name.textContent = MEMORY_LABELS[key] || key;
    const val = document.createElement("span");
    val.className = "mem-value";
    val.textContent = Array.isArray(value) ? value.join(", ") : String(value);
    const forget = document.createElement("button");
    forget.type = "button";
    forget.className = "chip-x";
    forget.textContent = "×";
    forget.setAttribute("aria-label", `Forget ${MEMORY_LABELS[key] || key}`);
    forget.addEventListener("click", () => forgetMemory(key));
    row.append(name, val, forget);
    els.memOther.appendChild(row);
  }

  els.memClear.hidden = Object.keys(memory).length === 0;
  disarmMemClear();
  hideMemError();
}

function renderChips(container, key, values) {
  container.innerHTML = "";
  const list = Array.isArray(values) ? values : [];
  if (!list.length) {
    const none = document.createElement("span");
    none.className = "muted small";
    none.textContent = "None yet";
    container.appendChild(none);
    return;
  }
  for (const value of list) {
    const chip = document.createElement("span");
    chip.className = key === "allergies" ? "chip chip-allergy" : "chip";
    const text = document.createElement("span");
    text.textContent = value;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-x";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${value}`);
    remove.addEventListener("click", () => forgetMemory(key, value));
    chip.append(text, remove);
    container.appendChild(chip);
  }
}

function saveMemory(key, value) {
  if (key === "diet" && !value) return forgetMemory("diet");
  return memoryRequest(
    fetch("/api/agent/memory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, value }),
    })
  );
}

function forgetMemory(key, value) {
  const params = new URLSearchParams();
  if (key) params.set("key", key);
  if (key && value !== undefined) params.set("value", value);
  const query = params.toString();
  return memoryRequest(fetch(`/api/agent/memory${query ? "?" + query : ""}`, { method: "DELETE" }));
}

async function memoryRequest(request) {
  hideMemError();
  try {
    const response = await request;
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showMemError((payload && payload.error) || "Couldn't update your preferences.");
      loadMemory(); // e.g. put the diet dropdown back to what's actually saved
      return false;
    }
    if (payload && payload.memory) renderMemory({ memory: payload.memory });
    else loadMemory();
    return true;
  } catch (_err) {
    showMemError("Couldn't reach the server to update your preferences.");
    return false;
  }
}

async function addMemoryFromForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const input = form.querySelector("input");
  const value = input.value.trim();
  if (!value) return;
  if (await saveMemory(form.dataset.key, value)) input.value = "";
}

// "Forget all" asks twice (no blocking dialog): the first tap arms it for a few seconds.
function clearMemory() {
  if (!memClearArmed) {
    els.memClear.textContent = "Tap again to forget all";
    els.memClear.classList.add("armed");
    memClearArmed = setTimeout(disarmMemClear, 4000);
    return;
  }
  disarmMemClear();
  forgetMemory();
}

function disarmMemClear() {
  if (memClearArmed) clearTimeout(memClearArmed);
  memClearArmed = null;
  els.memClear.textContent = "Forget all";
  els.memClear.classList.remove("armed");
}

function showMemError(message) {
  els.memError.textContent = message;
  els.memError.hidden = false;
}

function hideMemError() {
  els.memError.hidden = true;
  els.memError.textContent = "";
}

// --- Shopping list --------------------------------------------------------------
async function loadShopping() {
  try {
    const response = await fetch("/api/shopping");
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showShopMsg((payload && payload.error) || "Couldn't load your shopping list.", true);
      return;
    }
    renderShopping((payload && payload.items) || []);
  } catch (_err) {
    showShopMsg("Couldn't reach the server to load your shopping list.", true);
  }
}

function renderShopping(items) {
  els.shopList.innerHTML = "";
  for (const item of items) {
    const row = document.createElement("li");
    row.className = item.done ? "shop-item done" : "shop-item";

    const label = document.createElement("label");
    label.className = "shop-check";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = Boolean(item.done);
    box.addEventListener("change", () => toggleShopping(item.id, box.checked));
    const text = document.createElement("span");
    text.className = "shop-text";
    const name = document.createElement("span");
    name.className = "shop-name";
    name.textContent = item.qty ? `${item.name} (${item.qty})` : item.name;
    text.appendChild(name);
    if (item.reason) {
      const why = document.createElement("span");
      why.className = "shop-reason muted small";
      why.textContent = item.reason;
      text.appendChild(why);
    }
    label.append(box, text);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-x";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${item.name}`);
    remove.addEventListener("click", () => deleteShopping(item.id));

    row.append(label, remove);
    els.shopList.appendChild(row);
  }
  const open = items.filter((i) => !i.done).length;
  els.shopEmpty.hidden = items.length > 0;
  els.shopCount.hidden = open === 0;
  els.shopCount.textContent = String(open);
  els.shopClear.hidden = !items.some((i) => i.done);
  updateCartBadges(open); // keep the sidebar + bottom-bar Cart badges in sync
}

async function addShoppingItem(event) {
  event.preventDefault();
  const name = els.shopInput.value.trim();
  if (!name) return;
  try {
    const response = await fetch("/api/shopping", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showShopMsg((payload && payload.error) || "Couldn't add that.", true);
      return;
    }
    els.shopInput.value = "";
    const refused = (payload && payload.refused_for_allergy) || [];
    if (refused.length) showShopMsg(`Not added — household allergy: ${refused.join(", ")}.`, true);
    else if (payload && !(payload.added || []).length) showShopMsg(`${name} is already on the list.`);
    else els.shopMsg.hidden = true;
    loadShopping();
  } catch (_err) {
    showShopMsg("Couldn't reach the server to update your list.", true);
  }
}

async function toggleShopping(id, done) {
  await shoppingRequest(
    fetch(`/api/shopping/${Number(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ done }),
    })
  );
}

async function deleteShopping(id) {
  await shoppingRequest(fetch(`/api/shopping/${Number(id)}`, { method: "DELETE" }));
}

async function clearBoughtShopping() {
  await shoppingRequest(fetch("/api/shopping?done=1", { method: "DELETE" }));
}

async function shoppingRequest(request) {
  try {
    const response = await request;
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      showShopMsg((payload && payload.error) || "Couldn't update your list.", true);
    } else {
      els.shopMsg.hidden = true;
    }
  } catch (_err) {
    showShopMsg("Couldn't reach the server to update your list.", true);
  }
  loadShopping(); // always re-read: the list on screen should match the server
}

function showShopMsg(message, isError) {
  els.shopMsg.textContent = message;
  els.shopMsg.classList.toggle("inline-error", Boolean(isError));
  els.shopMsg.hidden = false;
}

// --- Smart restock (Cart tab) ---------------------------------------------------
// Predicts which staples are running low from real usage history (the waste log) and lets
// the user drop them straight onto the shopping list below.
let lastRestockItems = [];

async function loadRestock() {
  if (!els.restockCard) return;
  try {
    const response = await fetch("/api/restock");
    if (!response.ok) {
      els.restockCard.hidden = true; // the card is a bonus — hide quietly if it can't load
      return;
    }
    const payload = await response.json().catch(() => null);
    renderRestock(payload || {});
  } catch (_err) {
    els.restockCard.hidden = true;
  }
}

function renderRestock(data) {
  const items = Array.isArray(data.items) ? data.items : [];
  const currency = data.currency || "₹";
  lastRestockItems = items;

  // Nothing worth nudging and no history yet → keep the card out of the way entirely,
  // and clear any leftover rows so a re-open doesn't flash a stale "running low" list.
  if (!items.length) {
    els.restockCard.hidden = true;
    els.restockAdded.hidden = true;
    els.restockList.innerHTML = "";
    els.restockCount.hidden = true;
    els.restockCount.textContent = "";
    els.restockAddall.hidden = true;
    els.restockHeadline.textContent = "";
    return;
  }
  els.restockCard.hidden = false;
  els.restockAdded.hidden = true;

  els.restockHeadline.textContent = data.headline || "";
  els.restockCount.hidden = false;
  els.restockCount.textContent = String(items.length);
  els.restockAddall.hidden = items.length < 2;

  els.restockList.innerHTML = "";
  for (const item of items) {
    els.restockList.appendChild(buildRestockRow(item, currency));
  }
}

function buildRestockRow(item, currency) {
  const row = document.createElement("div");
  row.className = "restock-row";

  const main = document.createElement("div");
  main.className = "restock-main";

  const head = document.createElement("div");
  head.className = "restock-head";
  const name = document.createElement("span");
  name.className = "restock-name";
  name.textContent = item.name || item.token || "";
  const badge = document.createElement("span");
  const status = item.status === "out" ? "out" : "low";
  badge.className = `restock-badge restock-${status}`;
  badge.textContent = status === "out" ? "Out" : "Low";
  head.append(name, badge);

  const reason = document.createElement("p");
  reason.className = "restock-reason muted small";
  reason.textContent = item.reason || "";
  main.append(head, reason);

  const side = document.createElement("div");
  side.className = "restock-side";
  const price = Number(item.est_cost);
  if (Number.isFinite(price) && price > 0) {
    const cost = document.createElement("span");
    cost.className = "restock-cost";
    cost.textContent = `~${formatMoney(price, currency)}`;
    side.appendChild(cost);
  }
  const add = document.createElement("button");
  add.type = "button";
  add.className = "btn btn-secondary btn-sm restock-add";
  add.textContent = "Add";
  add.setAttribute("aria-label", `Add ${item.name} to shopping list`);
  add.addEventListener("click", () => addRestockToList([item.name], add));
  side.appendChild(add);

  row.append(main, side);
  return row;
}

// POST one or more restock suggestions onto the shopping list, then refresh both the list
// and the nudge (the added items drop off it). Mirrors addPlanShoppingToList.
async function addRestockToList(names, triggerBtn) {
  const items = (names || []).filter(Boolean).map((name) => ({ name }));
  if (!items.length) return;
  if (triggerBtn) triggerBtn.disabled = true;
  try {
    const response = await fetch("/api/shopping", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showRestockAdded((payload && payload.error) || "Couldn't add that to your list.", true);
      return;
    }
    const added = (payload && payload.added) || [];
    const refused = (payload && payload.refused_for_allergy) || [];
    if (refused.length) {
      showRestockAdded(`Not added — household allergy: ${refused.join(", ")}.`, true);
    } else if (!added.length) {
      showRestockAdded("Already on your list.");
    } else {
      const label = added.length === 1 ? added[0] : `${added.length} items`;
      showRestockAdded(`Added ${label} to your shopping list.`);
    }
    loadShopping();
    loadRestock();
  } catch (_err) {
    showRestockAdded("Couldn't reach the server to update your list.", true);
  } finally {
    if (triggerBtn) triggerBtn.disabled = false;
  }
}

function addAllRestock() {
  addRestockToList(lastRestockItems.map((i) => i.name), els.restockAddall);
}

function showRestockAdded(message, isError) {
  els.restockAdded.textContent = message;
  els.restockAdded.classList.toggle("cook-error", Boolean(isError));
  els.restockAdded.hidden = false;
}

// --- Receipt import (Scan tab) --------------------------------------------------
// Paste receipt text → the server parses it offline into grocery line-items → the user
// reviews (ticking off what to keep) → we merge the checked items into the fridge as a
// fresh scan. Nothing is written until the user taps "Add to fridge".
let lastReceipt = { items: [], currency: "₹" };

async function previewReceipt(event) {
  if (event) event.preventDefault();
  if (!els.receiptText) return;
  const text = els.receiptText.value.trim();
  els.receiptAdded.hidden = true;
  if (!text) {
    showReceiptError("Paste the receipt text first.");
    return;
  }
  els.receiptError.hidden = true;
  els.receiptPreview.disabled = true;
  els.receiptPreview.textContent = "Reading…";
  try {
    const response = await fetch("/api/receipt/parse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showReceiptError((payload && payload.error) || "Couldn't read that receipt.");
      return;
    }
    renderReceipt(payload || {});
  } catch (_err) {
    showReceiptError("Couldn't reach the server to read that receipt.");
  } finally {
    els.receiptPreview.disabled = false;
    els.receiptPreview.textContent = "Preview items";
  }
}

function renderReceipt(data) {
  const items = Array.isArray(data.items) ? data.items : [];
  const currency = data.currency || "₹";
  lastReceipt = { items, currency };
  els.receiptClear.hidden = false;

  if (!items.length) {
    els.receiptResult.hidden = true;
    els.receiptList.innerHTML = "";
    showReceiptError("No grocery items found in that text — try pasting the item lines.");
    return;
  }

  els.receiptError.hidden = true;
  els.receiptResult.hidden = false;
  els.receiptCount.textContent = String(items.length);

  const known = Number(data.known_count) || items.filter((i) => i.known).length;
  const totalTxt =
    Number.isFinite(Number(data.total)) && Number(data.total) > 0
      ? ` · about ${formatMoney(data.total, currency)}`
      : "";
  els.receiptSummary.textContent = `${known} of ${items.length} recognised${totalTxt}.`;

  els.receiptList.innerHTML = "";
  items.forEach((item, idx) => {
    els.receiptList.appendChild(buildReceiptRow(item, idx, currency));
  });
  updateReceiptImportHint();
}

function buildReceiptRow(item, idx, currency) {
  const row = document.createElement("label");
  row.className = "receipt-row";
  row.htmlFor = `receipt-item-${idx}`;

  const check = document.createElement("input");
  check.type = "checkbox";
  check.className = "receipt-check";
  check.id = `receipt-item-${idx}`;
  check.checked = true;
  check.dataset.idx = String(idx);
  check.addEventListener("change", updateReceiptImportHint);

  const main = document.createElement("div");
  main.className = "receipt-main";

  const head = document.createElement("div");
  head.className = "receipt-head";
  const name = document.createElement("span");
  name.className = "receipt-name";
  name.textContent = item.name || item.token || "Item";
  head.appendChild(name);
  const qty = Number(item.qty);
  if (Number.isFinite(qty) && qty > 1) {
    const q = document.createElement("span");
    q.className = "receipt-qty";
    q.textContent = `×${qty}`;
    head.appendChild(q);
  }
  if (!item.known) {
    const tag = document.createElement("span");
    tag.className = "receipt-unsure-tag";
    tag.textContent = "unsure";
    head.appendChild(tag);
  }
  main.appendChild(head);
  if (item.raw && item.raw !== item.name) {
    const raw = document.createElement("p");
    raw.className = "receipt-raw muted small";
    raw.textContent = item.raw;
    main.appendChild(raw);
  }

  const side = document.createElement("div");
  side.className = "receipt-side";
  const price = Number(item.price);
  if (Number.isFinite(price) && price > 0) {
    const cost = document.createElement("span");
    cost.className = "receipt-cost";
    cost.textContent = formatMoney(price, currency);
    side.appendChild(cost);
  }

  row.append(check, main, side);
  return row;
}

function checkedReceiptItems() {
  const boxes = els.receiptList.querySelectorAll(".receipt-check");
  const picked = [];
  boxes.forEach((box) => {
    if (box.checked) {
      const item = lastReceipt.items[Number(box.dataset.idx)];
      if (item) picked.push(item);
    }
  });
  return picked;
}

function updateReceiptImportHint() {
  const picked = checkedReceiptItems();
  els.receiptImport.disabled = picked.length === 0;
  const units = picked.reduce((sum, i) => sum + (Number(i.qty) > 0 ? Number(i.qty) : 1), 0);
  els.receiptImportHint.textContent = picked.length
    ? `${picked.length} item${picked.length === 1 ? "" : "s"} · ${units} unit${units === 1 ? "" : "s"}`
    : "Tick at least one item.";
}

async function importReceipt() {
  const picked = checkedReceiptItems();
  if (!picked.length) return;
  const items = picked.map((i) => ({ name: i.name, qty: Number(i.qty) > 0 ? Number(i.qty) : 1 }));
  els.receiptImport.disabled = true;
  els.receiptImport.textContent = "Adding…";
  try {
    const response = await fetch("/api/receipt/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      showReceiptError((payload && payload.error) || "Couldn't add those items.");
      return;
    }
    const added = Number(payload && payload.added) || items.length;
    const units = Number(payload && payload.units) || 0;
    els.receiptAdded.textContent = `✓ Added ${added} item${added === 1 ? "" : "s"}${
      units ? ` (${units} unit${units === 1 ? "" : "s"})` : ""
    } to your fridge.`;
    els.receiptAdded.classList.remove("cook-error");
    els.receiptAdded.hidden = false;
    // Reset the form; the server broadcast refreshes the fridge, but refresh locally too.
    clearReceipt();
    refreshScopes(["inventory", "nutrition"]);
  } catch (_err) {
    showReceiptError("Couldn't reach the server to add those items.");
  } finally {
    els.receiptImport.disabled = false;
    els.receiptImport.textContent = "Add to fridge";
  }
}

function clearReceipt() {
  els.receiptText.value = "";
  els.receiptResult.hidden = true;
  els.receiptList.innerHTML = "";
  els.receiptError.hidden = true;
  els.receiptClear.hidden = true;
  lastReceipt = { items: [], currency: "₹" };
}

function showReceiptError(message) {
  els.receiptError.textContent = message;
  els.receiptError.hidden = false;
}

// --- Chef's briefing (proactive, on the My Fridge tab) --------------------------
async function loadBriefing() {
  try {
    const response = await fetch("/api/agent/briefing");
    if (!response.ok) return; // the banner is a bonus — stay quiet if it can't load
    const payload = await response.json().catch(() => null);
    renderBriefing(payload && payload.briefing);
  } catch (_err) {
    // ignore: same reason
  }
}

function renderBriefing(briefing) {
  if (!briefing) {
    els.briefing.hidden = true;
    delete els.briefing.dataset.id;
    return;
  }
  const body = briefing.body || {};
  const facts = body.facts || {};
  const urgent = (facts.discard || []).length || (facts.soon || []).length || facts.cook;
  const actions = Array.isArray(briefing.actions) ? briefing.actions : [];

  els.briefing.dataset.id = String(briefing.id);
  els.briefing.classList.toggle("calm", !urgent);
  els.briefingWhen.textContent = timeAgo(briefing.created_at);
  els.briefingHeadline.textContent = briefing.headline || "";
  els.briefingLines.innerHTML = "";
  for (const line of body.lines || []) {
    if (line === briefing.headline) continue;
    const li = document.createElement("li");
    li.textContent = line;
    els.briefingLines.appendChild(li);
  }
  els.briefingLines.hidden = !els.briefingLines.children.length;
  els.briefingActions.innerHTML = "";
  for (const action of actions) els.briefingActions.appendChild(renderActionCard(action));
  els.briefingActions.hidden = actions.length === 0;
  els.briefing.hidden = false;
}

async function checkBriefingNow() {
  els.briefingError.hidden = true;
  els.briefingCheck.disabled = true;
  const original = els.briefingCheck.textContent;
  els.briefingCheck.textContent = "Checking…";
  try {
    const response = await fetch("/api/agent/briefing", { method: "POST" });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      els.briefingError.textContent = (payload && payload.error) || "Chef couldn't check the fridge.";
      els.briefingError.hidden = false;
      return;
    }
    renderBriefing(payload && payload.briefing);
  } catch (_err) {
    els.briefingError.textContent = "Couldn't reach the server.";
    els.briefingError.hidden = false;
  } finally {
    els.briefingCheck.disabled = false;
    els.briefingCheck.textContent = original;
  }
}

async function dismissBriefing() {
  const id = els.briefing.dataset.id;
  els.briefing.hidden = true;
  if (!id) return;
  try {
    await fetch(`/api/agent/briefing/${Number(id)}/dismiss`, { method: "POST" });
  } catch (_err) {
    // best-effort: it's already hidden, and the next check re-syncs
  }
}

// "just now" / "5 min ago" / "3 h ago", else a full local date-time.
function timeAgo(iso) {
  const date = new Date(iso);
  if (!iso || isNaN(date.getTime())) return "";
  const minutes = Math.round((Date.now() - date.getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 24 * 60) return `${Math.round(minutes / 60)} h ago`;
  return formatTimestamp(iso);
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

// Map each server "scope" to the client refresh it triggers. A refresh runs only if the
// relevant view has been opened (destsSeen) — except the fridge, the home view, which stays
// current so it's fresh whenever the user returns, and shopping, which drives the always-
// visible Cart badges. The meal plan is intentionally excluded: re-solving on every change
// would fight the "rebuild explicitly" design.
const LIVE_SCOPES = {
  inventory: () => {
    loadFridge();
    if (destsSeen.has("recipes")) loadSuggestions();
    if (destsSeen.has("cart")) loadRestock(); // what's on hand changed
  },
  perishables: () => {
    if (destsSeen.has("recipes")) loadPerishables();
    if (destsSeen.has("cart")) loadRestock();
  },
  analytics: () => {
    if (analyticsLoadedOnce) loadAnalytics();
    if (destsSeen.has("cart")) loadRestock(); // waste log = the restock usage history
  },
  nutrition: () => {
    if (nutritionLoadedOnce) loadNutrition();
  },
  // Chef's memory; the shopping list (always refreshed so the Cart badges stay live even
  // when the Cart view is closed); suggested changes; and the fridge briefing.
  memory: () => {
    if (chefLoadedOnce) loadMemory();
  },
  shopping: () => {
    loadShopping();
    if (destsSeen.has("cart")) loadRestock(); // an item queued to buy drops off the nudge
  },
  agent: () => syncActions(), // cheap; keeps Confirm cards on every view (incl. briefing) honest
  briefing: () => loadBriefing(),
};

// Only these scopes pop the "Updated live" toast; Chef's bookkeeping (agent, briefing)
// updates quietly.
const SCOPE_LABELS = {
  inventory: "Fridge",
  perishables: "Tracked items",
  analytics: "Analytics",
  nutrition: "Nutrition",
  memory: "Chef's memory",
  shopping: "Shopping list",
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
  refreshScopes(scopes);
  if (scopes.some((s) => s in SCOPE_LABELS)) showLiveToast(scopes);
}

// Run the client refresh for each scope (also used by hand when there's no live link).
function refreshScopes(scopes) {
  for (const scope of new Set(scopes)) {
    if (!(scope in LIVE_SCOPES)) continue;
    try {
      LIVE_SCOPES[scope]();
    } catch (err) {
      console.error("Live refresh failed for scope", scope, err);
    }
  }
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
  liveSocket = socket;

  setLiveStatus("connecting", "Connecting…");
  socket.on("connect", () => setLiveStatus("live", "Live"));
  socket.on("disconnect", () => setLiveStatus("offline", "Offline"));
  socket.on("connect_error", () => setLiveStatus("offline", "Offline"));
  socket.on("state_changed", handleStateChanged);
  socket.on("agent_step", handleAgentStep);
}

// --- Cook Mode --------------------------------------------------------------
// A full-screen guided cooking overlay: synthesised steps, a per-step countdown timer,
// optional spoken steps (Web Speech), ingredient swap ideas from the trained embeddings,
// and an "I cooked this" button that decrements the fridge through the normal cook path.
// The heavy lifting is server-side (GET /api/cook/<id> builds the session); this is just
// presentation + a couple of small browser toys (timer chime, text-to-speech).

const cookEls = {}; // populated in initCookMode()
const cookState = {
  session: null,
  index: 0,
  voice: false,
  voiceSupported: typeof window !== "undefined" && "speechSynthesis" in window,
  saved: false,        // "I cooked this" already recorded (guards double submits)
  timer: { remaining: 0, total: 0, running: false, handle: null, done: false },
  lastFocus: null,     // element to restore focus to on close
};

function initCookMode() {
  const ids = [
    "overlay", "sheet", "title", "method", "time", "have", "voice", "close",
    "ingredient-list", "note", "dots", "step-count", "step-n", "step-title",
    "step-text", "step-uses", "timer", "timer-display", "timer-toggle",
    "timer-reset", "prev", "next", "done", "status",
    "leftovers", "leftovers-form", "leftovers-name", "leftovers-days",
    "leftovers-save", "leftovers-msg",
  ];
  for (const id of ids) cookEls[id] = document.getElementById(`cook-${id}`);
  if (!cookEls.overlay) return; // markup absent → feature disabled

  cookEls.close.addEventListener("click", closeCookMode);
  if (cookEls["leftovers-form"]) {
    cookEls["leftovers-form"].addEventListener("submit", cookSaveLeftovers);
  }
  cookEls.overlay.addEventListener("click", (e) => {
    if (e.target === cookEls.overlay) closeCookMode(); // click the dimmed backdrop
  });
  cookEls.prev.addEventListener("click", () => cookGoTo(cookState.index - 1));
  cookEls.next.addEventListener("click", () => cookGoTo(cookState.index + 1));
  cookEls.done.addEventListener("click", cookMarkCooked);
  cookEls.voice.addEventListener("click", cookToggleVoice);
  cookEls["timer-toggle"].addEventListener("click", cookTimerToggle);
  cookEls["timer-reset"].addEventListener("click", cookTimerReset);

  if (!cookState.voiceSupported) {
    cookEls.voice.hidden = true; // no Web Speech in this browser
  }

  document.addEventListener("keydown", (e) => {
    if (cookEls.overlay.hidden) return;
    if (e.key === "Escape") { closeCookMode(); return; }
    if (e.key === "ArrowRight") { cookGoTo(cookState.index + 1); }
    else if (e.key === "ArrowLeft") { cookGoTo(cookState.index - 1); }
  });
}

async function openCookMode(recipeId) {
  if (!recipeId || !cookEls.overlay) return;
  cookState.lastFocus = document.activeElement;
  cookState.saved = false;
  // Show the shell immediately with a loading title so the tap feels instant.
  cookEls.title.textContent = "Loading…";
  cookEls.method.textContent = "";
  cookEls.time.textContent = "";
  cookEls.have.textContent = "";
  cookEls["ingredient-list"].innerHTML = "";
  cookEls.note.textContent = "";
  cookEls.status.hidden = true;
  if (cookEls.leftovers) cookEls.leftovers.hidden = true; // reset the leftovers offer
  if (cookEls["leftovers-msg"]) cookEls["leftovers-msg"].hidden = true;
  cookEls.overlay.hidden = false;
  document.body.style.overflow = "hidden"; // lock background scroll
  cookEls.close.focus();

  try {
    const response = await fetch(`/api/cook/${encodeURIComponent(recipeId)}`);
    const payload = await response.json().catch(() => null);
    if (!response.ok || !payload) {
      cookEls.title.textContent = "Couldn't open Cook Mode";
      cookEls.note.textContent =
        (payload && payload.error) || "Please try again in a moment.";
      return;
    }
    cookState.session = payload;
    cookState.index = 0;
    renderCookHeader(payload);
    renderCookIngredients(payload);
    renderCookStep(0);
  } catch (_err) {
    cookEls.title.textContent = "Couldn't open Cook Mode";
    cookEls.note.textContent = "Couldn't reach the server. Check your connection.";
  }
}

function closeCookMode() {
  if (!cookEls.overlay) return;
  cookTimerStop();
  cookStopSpeech();
  cookEls.overlay.hidden = true;
  document.body.style.overflow = "";
  if (cookState.lastFocus && typeof cookState.lastFocus.focus === "function") {
    cookState.lastFocus.focus();
  }
}

function renderCookHeader(session) {
  cookEls.title.textContent = session.title || "Cook Mode";
  cookEls.method.textContent = (session.method || "recipe").replace(/_/g, " ");
  cookEls.time.textContent = Number.isFinite(session.time_min)
    ? `⏱ ${session.time_min} min`
    : "";
  const have = Number.isFinite(session.have_count) ? session.have_count : 0;
  const total = Number.isFinite(session.total_count) ? session.total_count : 0;
  cookEls.have.textContent = total ? `🧺 ${have}/${total} on hand` : "";
}

function renderCookIngredients(session) {
  const list = cookEls["ingredient-list"];
  list.innerHTML = "";
  const ingredients = Array.isArray(session.ingredients) ? session.ingredients : [];
  for (const ing of ingredients) {
    const item = document.createElement("div");
    const stateClass = ing.have ? "have" : ing.pantry ? "pantry" : "missing";
    item.className = `cook-ing ${stateClass}`;

    const head = document.createElement("div");
    head.className = "cook-ing-head";
    const check = document.createElement("span");
    check.className = "cook-ing-check";
    check.textContent = ing.have ? "✓" : ing.pantry ? "•" : "+";
    check.setAttribute("aria-hidden", "true");
    const name = document.createElement("span");
    name.className = "cook-ing-name";
    name.textContent = ing.name || ing.token || "";
    const tag = document.createElement("span");
    tag.className = "cook-ing-tag";
    tag.textContent = ing.have ? "Have" : ing.pantry ? "Pantry" : "Need";
    head.append(check, name, tag);
    item.appendChild(head);

    // Swap ideas: only for things you don't have (and aren't basic pantry staples).
    const subs = Array.isArray(ing.substitutes) ? ing.substitutes : [];
    if (!ing.have && !ing.pantry && subs.length) {
      const swaps = document.createElement("div");
      swaps.className = "cook-swaps";
      const label = document.createElement("span");
      label.className = "cook-swaps-label";
      label.textContent = "Swap:";
      swaps.appendChild(label);
      for (const s of subs) {
        const chip = document.createElement("span");
        chip.className = "cook-swap-chip";
        chip.textContent = s.name || s.token || "";
        swaps.appendChild(chip);
      }
      item.appendChild(swaps);
    }
    list.appendChild(item);
  }

  cookEls.note.textContent = session.note || "";
  cookEls.note.hidden = !session.note;
}

function renderCookStep(index) {
  const session = cookState.session;
  if (!session || !Array.isArray(session.steps) || !session.steps.length) return;
  const steps = session.steps;
  const i = Math.max(0, Math.min(index, steps.length - 1));
  cookState.index = i;
  const step = steps[i];

  cookEls["step-n"].textContent = String(step.n || i + 1);
  cookEls["step-title"].textContent = step.title || `Step ${i + 1}`;
  cookEls["step-text"].textContent = step.text || "";
  // Re-trigger the slide-in animation on each step change.
  const card = cookEls["step-title"].closest(".cook-step-card");
  if (card) { card.style.animation = "none"; void card.offsetWidth; card.style.animation = ""; }

  // "Uses" chips for this step.
  const uses = cookEls["step-uses"];
  uses.innerHTML = "";
  const stepUses = Array.isArray(step.uses) ? step.uses : [];
  for (const u of stepUses) {
    const chip = document.createElement("span");
    chip.className = "cook-use-chip";
    chip.textContent = u;
    uses.appendChild(chip);
  }

  // Progress dots + counter.
  renderCookDots(i, steps.length);
  cookEls["step-count"].textContent = `Step ${i + 1} of ${steps.length}`;

  // Timer for this step (only if it has a duration).
  cookTimerStop();
  const secs = Number.isFinite(step.seconds) ? step.seconds : 0;
  if (secs > 0) {
    cookEls.timer.hidden = false;
    cookTimerSet(secs);
  } else {
    cookEls.timer.hidden = true;
  }

  // Nav buttons: Back disabled on first; Next vs "I cooked this" on last.
  cookEls.prev.disabled = i === 0;
  const onLast = i === steps.length - 1;
  cookEls.next.hidden = onLast;
  cookEls.done.hidden = !onLast || cookState.saved;

  // Read the step aloud if voice is on.
  cookSpeak(`${step.title}. ${step.text}`);
}

function renderCookDots(active, total) {
  const dots = cookEls.dots;
  dots.innerHTML = "";
  for (let i = 0; i < total; i++) {
    const dot = document.createElement("span");
    dot.className = "cook-dot" + (i === active ? " active" : i < active ? " done" : "");
    dots.appendChild(dot);
  }
}

function cookGoTo(index) {
  const session = cookState.session;
  if (!session || !Array.isArray(session.steps)) return;
  if (index < 0 || index >= session.steps.length) return;
  renderCookStep(index);
}

// --- Per-step countdown timer ------------------------------------------------

function cookFormatTime(total) {
  const s = Math.max(0, Math.round(total));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function cookTimerSet(seconds) {
  cookState.timer.total = seconds;
  cookState.timer.remaining = seconds;
  cookState.timer.running = false;
  cookState.timer.done = false;
  cookEls["timer-display"].textContent = cookFormatTime(seconds);
  cookEls["timer-display"].classList.remove("running", "done");
  cookEls["timer-toggle"].textContent = "Start timer";
  cookEls["timer-toggle"].disabled = false;
}

function cookTimerToggle() {
  if (cookState.timer.done) { cookTimerReset(); return; }
  if (cookState.timer.running) {
    cookTimerStop();
    cookEls["timer-toggle"].textContent = "Resume";
  } else {
    cookTimerStart();
  }
}

function cookTimerStart() {
  const t = cookState.timer;
  if (t.remaining <= 0) return;
  t.running = true;
  cookEls["timer-toggle"].textContent = "Pause";
  cookEls["timer-display"].classList.add("running");
  const started = Date.now();
  let base = t.remaining;
  t.handle = setInterval(() => {
    const elapsed = (Date.now() - started) / 1000;
    t.remaining = Math.max(0, base - elapsed);
    cookEls["timer-display"].textContent = cookFormatTime(t.remaining);
    if (t.remaining <= 0) {
      cookTimerStop();
      t.done = true;
      cookEls["timer-display"].classList.remove("running");
      cookEls["timer-display"].classList.add("done");
      cookEls["timer-toggle"].textContent = "Restart";
      cookChime();
      cookSpeak("Time's up.");
    }
  }, 250);
}

function cookTimerStop() {
  const t = cookState.timer;
  if (t.handle) { clearInterval(t.handle); t.handle = null; }
  t.running = false;
  if (cookEls["timer-display"]) cookEls["timer-display"].classList.remove("running");
}

function cookTimerReset() {
  cookTimerStop();
  cookTimerSet(cookState.timer.total);
}

// A short two-note chime via Web Audio (no asset file needed). Silently no-ops if the
// browser blocks audio (e.g. no prior user gesture).
function cookChime() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const now = ctx.currentTime;
    [880, 1174.66].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      const t = now + i * 0.18;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.28, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.4);
    });
    setTimeout(() => { ctx.close().catch(() => {}); }, 1200);
  } catch (_e) { /* audio unavailable */ }
}

// --- Voice (Web Speech) ------------------------------------------------------

function cookToggleVoice() {
  if (!cookState.voiceSupported) return;
  cookState.voice = !cookState.voice;
  cookEls.voice.setAttribute("aria-pressed", String(cookState.voice));
  cookEls.voice.textContent = cookState.voice ? "🔊 Voice on" : "🔊 Voice off";
  if (cookState.voice) {
    const step = cookState.session && cookState.session.steps[cookState.index];
    if (step) cookSpeak(`${step.title}. ${step.text}`);
  } else {
    cookStopSpeech();
  }
}

function cookSpeak(text) {
  if (!cookState.voice || !cookState.voiceSupported || !text) return;
  try {
    const synth = window.speechSynthesis;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(String(text));
    u.rate = 0.98;
    u.pitch = 1.0;
    synth.speak(u);
  } catch (_e) { /* speech unavailable */ }
}

function cookStopSpeech() {
  try {
    if (cookState.voiceSupported) window.speechSynthesis.cancel();
  } catch (_e) { /* ignore */ }
}

// --- "I cooked this" → decrement the fridge through the normal cook path ------

async function cookMarkCooked() {
  const session = cookState.session;
  if (!session || cookState.saved) return;
  // Use exactly what you actually have on hand (skip pantry staples and missing items),
  // by their pretty names — the server normalises them back to canonical tokens.
  const uses = (session.ingredients || [])
    .filter((i) => i.have && !i.pantry)
    .map((i) => i.name)
    .filter(Boolean);

  cookEls.done.disabled = true;
  cookEls.status.hidden = false;
  cookEls.status.textContent = "Saving…";
  try {
    const response = await fetch("/api/cook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uses, title: session.title || "" }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      cookEls.status.textContent =
        (payload && payload.error) || "Couldn't record that meal.";
      cookEls.done.disabled = false;
      return;
    }
    cookState.saved = true;
    cookEls.done.hidden = true;
    const n = uses.length;
    cookEls.status.textContent = n
      ? `✓ Enjoy! Removed ${n} ingredient${n === 1 ? "" : "s"} from your fridge.`
      : "✓ Enjoy your meal!";
    // Refresh the fridge/analytics/perishables locally too (the server also broadcasts).
    refreshScopes(["inventory", "analytics", "perishables"]);
    // Offer to track leftovers instead of closing immediately — a real kitchen makes extra.
    showCookLeftovers(session.title || "");
  } catch (_err) {
    cookEls.status.textContent = "Couldn't reach the server to save that.";
    cookEls.done.disabled = false;
  }
}

// --- Leftovers: track what you made extra of so it doesn't get forgotten -------

function showCookLeftovers(title) {
  if (!cookEls.leftovers) return;
  const name = String(title || "").trim();
  cookEls["leftovers-name"].value = name ? `Leftover ${name}` : "";
  cookEls["leftovers-days"].value = "3";
  cookEls["leftovers-msg"].hidden = true;
  cookEls["leftovers-save"].disabled = false;
  cookEls["leftovers-save"].textContent = "Save leftovers";
  cookEls.leftovers.hidden = false;
}

async function cookSaveLeftovers(event) {
  if (event) event.preventDefault();
  const name = cookEls["leftovers-name"].value.trim();
  const days = Number(cookEls["leftovers-days"].value) || 3;
  if (!name) {
    cookEls["leftovers-msg"].textContent = "Give the leftovers a name first.";
    cookEls["leftovers-msg"].classList.add("cook-error");
    cookEls["leftovers-msg"].hidden = false;
    return;
  }
  cookEls["leftovers-save"].disabled = true;
  cookEls["leftovers-save"].textContent = "Saving…";
  try {
    const response = await fetch("/api/leftovers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, days }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      cookEls["leftovers-msg"].textContent =
        (payload && payload.error) || "Couldn't save those leftovers.";
      cookEls["leftovers-msg"].classList.add("cook-error");
      cookEls["leftovers-msg"].hidden = false;
      cookEls["leftovers-save"].disabled = false;
      cookEls["leftovers-save"].textContent = "Save leftovers";
      return;
    }
    const useBy = (payload && payload.perishable && payload.perishable.use_by) || "";
    cookEls["leftovers-msg"].classList.remove("cook-error");
    cookEls["leftovers-msg"].textContent = useBy
      ? `✓ Tracked — use by ${useBy}. It'll show up under Tracked items.`
      : "✓ Leftovers tracked.";
    cookEls["leftovers-msg"].hidden = false;
    cookEls["leftovers-save"].textContent = "Saved ✓";
    refreshScopes(["perishables"]);
  } catch (_err) {
    cookEls["leftovers-msg"].textContent = "Couldn't reach the server to save those.";
    cookEls["leftovers-msg"].classList.add("cook-error");
    cookEls["leftovers-msg"].hidden = false;
    cookEls["leftovers-save"].disabled = false;
    cookEls["leftovers-save"].textContent = "Save leftovers";
  }
}

// --- Startup ----------------------------------------------------------------
// Restore the saved theme, start the ambient frost, and prime the shopping list so the
// Cart badges are live from the first paint. Fridge is the home view: it loads the current
// inventory (and Chef's briefing). Then subscribe to live updates.
initTheme();
initFrost();
initCookMode();
loadShopping();
activateDest("fridge");
initRealtime();
