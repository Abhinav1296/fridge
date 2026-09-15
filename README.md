# SmartFridge Vision MVP

**Phase 1 of a smart-fridge system.** It does exactly one thing well: you give it a
photo of your fridge or box contents — by **file upload** or a **live webcam
snapshot** — and it returns a structured inventory with **per-item freshness**,
powered by a vision-language model.

That's the whole scope for now. See [Out of scope / coming next](#out-of-scope--coming-next).

---

## What you get

- A **My Fridge** home view showing your **current inventory** at a glance — a total,
  a freshness breakdown, a "use these first" nudge for anything spoiled or about to go,
  and items grouped by category. See [What's in my fridge now](#whats-in-my-fridge-now).
- Two input paths for scanning: drag-and-drop / file-picker upload, and a live webcam
  capture.
- Each detected item rendered as a card: **name, count, category**, a colour-coded
  **freshness badge** (green = fresh, teal = ripe, amber = use soon, red = spoiled,
  grey = unknown), a **confidence** chip, and free-text notes.
- Anything the model can't identify (opaque bags, heavy occlusion) is listed
  separately under **"Needs your input"** instead of being guessed.
- A **History** tab that records every scan — what was found and **when** — so you can
  look back at previous inventories. See [Inventory history](#inventory-history).

---

## Requirements

- Python 3.11+
- An API key for any **OpenAI-compatible vision** chat-completions endpoint. The
  defaults target [OpenRouter](https://openrouter.ai)'s free vision tier (`:free` models).

---

## Setup

All commands below are for **PowerShell on Windows** (this project's environment).
On macOS/Linux, activate the venv with `source .venv/bin/activate` instead.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once for this session.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Add your API key

Copy the example config and open `.env` in an editor:

```powershell
Copy-Item .env.example .env
```

Then set `VISION_API_KEY` to your key. Create a free OpenRouter key at
<https://openrouter.ai/keys>. The `.env` file is git-ignored — **never commit it**.

```dotenv
VISION_API_KEY=your_key_here
VISION_BASE_URL=https://openrouter.ai/api/v1
VISION_MODEL=inclusionai/ling-3.0-flash-vl:free
```

> **Free-tier note:** OpenRouter's `:free` models have rate limits (and some require a
> small one-time credit top-up to raise the daily cap). If you hit a limit, the app
> shows a friendly "rate limited" message — wait a moment, or switch to another `:free`
> model (see below). Better yet, configure a few [fallbacks](#fallbacks) so the app
> automatically tries another model or key when one is rate limited.

### 4. Run

```powershell
flask run
```

Then open <http://127.0.0.1:5000>. (You can also run `python app.py`, which starts the
same server with debug reload enabled.)

Upload a produce photo or capture one from your webcam, press **Analyze image**, and
within a few seconds you'll see the inventory cards.

---

## Configuration & swapping providers

Everything is controlled by three environment variables in `.env` — no code changes
needed to switch providers:

| Variable          | What it is                                             |
| ----------------- | ------------------------------------------------------ |
| `VISION_API_KEY`  | Your provider API key.                                 |
| `VISION_BASE_URL` | The OpenAI-compatible base URL (ends in `/v1`).        |
| `VISION_MODEL`    | The vision model id to call.                           |

The `_2` / `_3` / `_4` suffixed versions of these configure fallbacks — see
[Fallbacks](#fallbacks) below.

Because the app speaks the OpenAI chat-completions format, you can point it at any
compatible **vision** provider:

- **OpenRouter (default)** — aggregates many providers; `:free` models cost nothing.
  `VISION_BASE_URL=https://openrouter.ai/api/v1`
  `VISION_MODEL=inclusionai/ling-3.0-flash-vl:free`
  Browse current free vision models at <https://openrouter.ai/models> (filter by
  **input: image** and **price: free**). Other free vision ids seen recently:
  `nex-agi/nex-n2.5-mini:free`, `nex-agi/nex-n2.5-pro:free`.

- **Groq**
  `VISION_BASE_URL=https://api.groq.com/openai/v1`
  `VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`

- **Google Gemini** (OpenAI-compatible endpoint)
  `VISION_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`
  `VISION_MODEL=gemini-2.0-flash` *(or another current Gemini vision model)*

- **Any other OpenAI-compatible vision provider** — set the base URL and model id they
  document.

> ⚠️ **Model ids change over time — free ones rotate especially fast.** The default is a
> starting point, not a guarantee. Before relying on it, **verify the current vision
> model id from your provider's model list** (for OpenRouter, the models page above). The
> model you pick **must support image input** — text-only models will fail to read the
> photo.

---

## Fallbacks

Free vision tiers rate-limit aggressively and rotate model ids, so a single hard-coded
model is fragile. The app supports a **primary plus up to three fallbacks**, tried in
order until one succeeds. If an attempt fails — rate limited (429), rejected key
(401/403), unreachable endpoint, or an unreadable/garbled response — the app quietly
moves on to the next fallback, and only surfaces an error if **every** attempt fails.
Each attempt (and which one finally succeeded) is logged **server-side only**.

Configure fallbacks with the `_2`, `_3`, `_4` suffixes on any of the three variables:

| Slot        | Variables                                              |
| ----------- | ------------------------------------------------------ |
| Primary     | `VISION_API_KEY`  / `VISION_BASE_URL`  / `VISION_MODEL`  |
| Fallback 1  | `VISION_API_KEY_2` / `VISION_BASE_URL_2` / `VISION_MODEL_2` |
| Fallback 2  | `VISION_API_KEY_3` / `VISION_BASE_URL_3` / `VISION_MODEL_3` |
| Fallback 3  | `VISION_API_KEY_4` / `VISION_BASE_URL_4` / `VISION_MODEL_4` |

**Anything you leave blank in a fallback is inherited from the slot before it.** A
fallback is ignored entirely unless you fill in at least one of its variables, and exact
duplicate attempts are skipped so no call is wasted.

- **Just try different models** (same key + provider) — set only the model lines:

  ```dotenv
  VISION_API_KEY=your_openrouter_key
  VISION_BASE_URL=https://openrouter.ai/api/v1
  VISION_MODEL=inclusionai/ling-3.0-flash-vl:free
  VISION_MODEL_2=nex-agi/nex-n2.5-pro:free
  VISION_MODEL_3=nex-agi/nex-n2.5-mini:free
  ```

- **Fall back to a different key** (e.g. when one hits its daily cap) — set the key:

  ```dotenv
  VISION_API_KEY_2=your_second_openrouter_key
  ```

- **Fall back to a different provider entirely** — set all three for that slot:

  ```dotenv
  VISION_API_KEY_4=your_groq_key
  VISION_BASE_URL_4=https://api.groq.com/openai/v1
  VISION_MODEL_4=meta-llama/llama-4-scout-17b-16e-instruct
  ```

> Want more than three fallbacks? Bump `_MAX_ATTEMPTS` in `vision_service.py` and add the
> matching `_5`, `_6`, … variables.

---

## Project structure

```
capstone/
├── app.py               # Flask web layer: routing, image guards, error mapping
├── vision_service.py    # Vision integration (framework-agnostic) + robust JSON parsing
├── storage.py           # SQLite inventory-history persistence (framework-agnostic)
├── templates/
│   └── index.html       # Single-page UI (My Fridge / Upload / Webcam / History tabs)
├── static/
│   ├── style.css        # Responsive, mobile-friendly styles + freshness badges
│   └── app.js           # Upload, webcam capture, fetch, and result rendering
├── requirements.txt
├── .env.example         # Config template (copy to .env)
├── smartfridge.db       # Local history database (auto-created, git-ignored)
└── README.md
```

**Design note:** all vision logic lives in `vision_service.analyze_image(image_bytes)
-> dict`, and all persistence in `storage.py` — **neither has a Flask dependency**. A
future agent layer can call them directly without touching the web code; that's the seam
that keeps this modular.

### How an analysis flows

1. The browser sends the image to `POST /analyze` — as multipart form data (upload) or
   a base64 JSON body (webcam).
2. `app.py` validates it (rejects missing images and anything over 5 MB), downscales
   images whose longest side exceeds ~1600 px, and re-encodes to JPEG.
3. `vision_service.analyze_image` base64-encodes the image into a `data:` URL, then tries
   the configured provider attempts in order — the primary first, then each
   [fallback](#fallbacks) — until one returns a usable response. For each attempt it calls
   the vision endpoint and parses the response, stripping any ```` ```json ```` fences and
   normalizing the shape.
4. The parsed inventory JSON comes back to the page and renders as cards.
5. The scan is recorded to the local [history database](#inventory-history) with a
   UTC timestamp (best-effort — if the database is unavailable the analysis still
   returns normally). The newest scan is what the [My Fridge](#whats-in-my-fridge-now)
   view reads back as your current inventory.

Errors (missing/oversized image, network/API failure, HTTP 429 rate limits, and
unparseable model output) are turned into clean, friendly messages. Raw model text and
provider details are logged **server-side only** — never shown to the user.

---

## What's in my fridge now

The **My Fridge** tab (the home view) answers the everyday question — *what's in my
fridge right now?* — without you scrolling through history. Because each scan captures
the whole fridge at one moment, the **most recent scan is treated as the current
inventory**. The view shows:

- an **"as of" timestamp** (in your local time) and whether it came from an upload or the
  webcam;
- a **total item count** and a **freshness breakdown** (how many fresh / ripe / use soon /
  spoiled / unknown);
- a **"use these first" callout** whenever something is spoiled or about to go;
- your items **grouped by category** (fruit, vegetable, dairy, …), each as a full card —
  with a **"Show all items"** toggle to collapse the list down to just the summary and
  callout when you only want the glance.

If you haven't scanned anything yet, it shows an empty state with a shortcut to the
Upload tab. It's served by `GET /api/current`, which returns the latest scan (or
`{"scan": null}` when history is empty) via `storage.get_latest_scan()`.

> This is a **snapshot** view — it reflects your last scan, not a running tally
> reconciled across scans. A smarter rollup (and scan-to-scan diffing) is a planned next
> step; because callers go through `storage.get_latest_scan()`, that can be swapped in
> without changing the API.

---

## Inventory history

Every successful scan is saved to a local **SQLite** database, so you build up a
timestamped record of what was in your fridge and **when**. Open the **History** tab in
the UI to see past scans newest-first — each entry shows the date/time (in your local
timezone), whether it came from an upload or the webcam, and a quick row of the items
found with their freshness. Hit **"Show all items"** on any entry to expand a full
breakdown of *everything detected in that particular image* — grouped by category
(vegetables, fruit, dairy, …) as full cards, with anything the model couldn't identify
listed under "Needs your input". It's read from `GET /api/history` (metadata only — no
photos are stored).

Storage uses **libSQL** (a SQLite fork) through the `libsql-client` package, so the same
code runs against either a local file in development or a hosted **Turso** database in the
cloud. It's created automatically on first run. Which target is used is decided from the
environment at call time:

| Variable             | Default          | What it is                                                        |
| -------------------- | ---------------- | ----------------------------------------------------------------- |
| `DATABASE_PATH`      | `smartfridge.db` | Local SQLite file for history — used when no Turso URL is set.     |
| `TURSO_DATABASE_URL` | *(unset)*        | If set, connect to this Turso/libSQL database instead of the file. |
| `TURSO_AUTH_TOKEN`   | *(unset)*        | Auth token for the Turso database above.                          |

Locally you don't need Turso at all — leave the Turso vars unset and history is written to
the `DATABASE_PATH` file. The file (`*.db`) is git-ignored — it's your data, not code.
Delete it to clear all history; the app recreates an empty one on the next run. History is
best-effort by design: if the database can't be opened or written, it's logged server-side
and the core analysis keeps working — nothing about scanning depends on it.

Persistence lives in `storage.py`, which — like `vision_service.py` — has **no Flask
dependency**, so a future agent layer can read and write history directly.

---

## Deploy for free (Vercel + Turso)

The app runs on **Vercel's** free tier, with history stored in a free **Turso** database.
Turso is needed because a serverless platform's filesystem is ephemeral — a plain SQLite
file would be wiped between requests, so history wouldn't persist. Turso is a hosted,
SQLite-compatible database that solves exactly that. Config for the deploy lives in
[`vercel.json`](vercel.json); `app.py` is used directly as the serverless entry point.

1. **Create a Turso database** at [turso.tech](https://turso.tech) (sign in with GitHub).
   Create a database, then copy two values from its dashboard: the **database URL**
   (looks like `libsql://<name>-<you>.turso.io`) and a freshly-generated **auth token**.
2. **Import the repo into Vercel** at [vercel.com/new](https://vercel.com/new) — pick this
   GitHub repo. Vercel auto-detects the Python config; you don't need to change build
   settings.
3. **Add environment variables** in Vercel (Project → Settings → Environment Variables),
   matching your `.env`:
   - `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` — from step 1.
   - `VISION_API_KEY` — your vision provider key.
   - `VISION_BASE_URL` and `VISION_MODEL` — and any `*_2` / `*_3` / `*_4` fallbacks you use.
4. **Deploy.** Vercel builds and gives you a public URL. History and My Fridge now persist
   in Turso across restarts and redeploys.

> The Turso auth token is a secret — set it only in Vercel's dashboard (and your local
> `.env`), never in committed files.

---

## Out of scope / coming next

This project is intentionally narrow and grows one layer at a time. **Not built yet** —
planned for later phases:

- **Reconciled inventory** — a running tally that carries items across scans and lets you
  tick things off, instead of the current snapshot ([My Fridge](#whats-in-my-fridge-now)
  today shows your latest scan).
- **Scan diffing** — what was added/removed/used up between two scans.
- **Shopping list** — auto-generated from what's low or spoiled.
- **Notifications** — expiry / low-stock alerts.
- **Agents** — reasoning/automation on top of the inventory.
- **Hardware / IoT** — in-fridge cameras and sensors.

**Recently added:** a persistent **inventory history** database and a **My Fridge**
current-inventory view (see above) — the first steps beyond the vision MVP. The code is
structured so the rest can be added without a rewrite.
