"""
HF Hub Scraper
==============
Fetches live data from the public Hugging Face Hub REST API.
No authentication required — all endpoints serve public data.

Strategy
--------
• /api/models?sort=downloads&limit=1000  → top 1000 models by monthly downloads
  - Aggregates by pipeline_tag → task distribution (weighted by downloads)
  - Extracts license tags      → license distribution
  - Surfaces top 10 as table rows

• /api/spaces?sort=likes&limit=1000 → top 1000 spaces by likes
  - Aggregates by .sdk         → framework distribution

Output: data/hub_stats.json
"""

import json, datetime, time, sys, os
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ─────────────────────────────────────────────────────────
# CONFIG — tweak without touching rendering logic
# ─────────────────────────────────────────────────────────

BASE          = "https://huggingface.co/api"
TIMEOUT       = 30
MODEL_SAMPLE  = 1000   # must be ≤ HF max (1000)
SPACE_SAMPLE  = 1000
TOP_MODELS_N  = 10
MAX_TASKS     = 12
MAX_LICENSES  = 6
RETRY         = 3
RETRY_DELAY   = 4

TASK_COLORS = {
    "text-generation":              "#00d4ff",
    "text-classification":          "#00d4ff",
    "token-classification":         "#00d4ff",
    "fill-mask":                    "#00d4ff",
    "translation":                  "#00d4ff",
    "question-answering":           "#6366f1",
    "summarization":                "#6366f1",
    "sentence-similarity":          "#6366f1",
    "feature-extraction":           "#6366f1",
    "text-ranking":                 "#6366f1",
    "zero-shot-classification":     "#818cf8",
    "text-to-image":                "#a855f7",
    "image-to-text":                "#22d3a5",
    "image-text-to-text":           "#22d3a5",
    "image-classification":         "#22d3a5",
    "object-detection":             "#22d3a5",
    "depth-estimation":             "#22d3a5",
    "image-segmentation":           "#22d3a5",
    "zero-shot-image-classification":"#22d3a5",
    "automatic-speech-recognition": "#f97316",
    "text-to-speech":               "#f97316",
    "audio-classification":         "#f97316",
    "audio-to-audio":               "#f97316",
    "time-series-forecasting":      "#fbbf24",
    "tabular-classification":       "#fbbf24",
    "reinforcement-learning":       "#fbbf24",
    "video-classification":         "#22d3a5",
}

LICENSE_COLORS = {
    "apache-2.0":            "#00d4ff",
    "mit":                   "#22d3a5",
    "other":                 "#4a5270",
    "cc-by-4.0":             "#818cf8",
    "cc-by-nc-4.0":          "#a855f7",
    "cc-by-nc-sa-4.0":       "#a855f7",
    "openrail":              "#6366f1",
    "openrail++":            "#6366f1",
    "gemma":                 "#f97316",
    "llama2":                "#f97316",
    "llama3":                "#f97316",
    "llama3.1":              "#f97316",
    "llama3.2":              "#f97316",
    "llama3.3":              "#f97316",
    "creativeml-openrail-m": "#a855f7",
    "gpl-3.0":               "#818cf8",
}

SDK_COLORS = {
    "gradio":    "#f97316",
    "streamlit": "#6366f1",
    "docker":    "#00d4ff",
    "static":    "#22d3a5",
}


# ─────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────

def get(url, params=None, label=""):
    hdrs = {"Accept": "application/json", "User-Agent": "hf-hub-scraper/1.0"}
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, params=params, headers=hdrs, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  [{label}] attempt {attempt}/{RETRY}: {e}", flush=True)
            if attempt < RETRY:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"All retries failed: {label}")

def fmt(n):
    if   n >= 1_000_000_000: return f"{n/1e9:.1f}B"
    elif n >= 1_000_000:      return f"{n/1e6:.1f}M"
    elif n >= 1_000:          return f"{round(n/1000)}K"
    return str(n)


# ─────────────────────────────────────────────────────────
# SCRAPERS
# ─────────────────────────────────────────────────────────

def fetch_models(limit=MODEL_SAMPLE):
    print(f"Fetching top {limit} models by monthly downloads…", flush=True)
    r = get(f"{BASE}/models",
            params={"sort":"downloads","direction":"-1","limit":limit,"full":"False"},
            label="models")
    return r.json()


def fetch_spaces(limit=SPACE_SAMPLE):
    print(f"Fetching top {limit} Spaces by likes…", flush=True)
    r = get(f"{BASE}/spaces",
            params={"sort":"likes","direction":"-1","limit":limit,"full":"False"},
            label="spaces")
    return r.json()


# ─────────────────────────────────────────────────────────
# AGGREGATORS
# ─────────────────────────────────────────────────────────

def agg_tasks(models):
    """
    Aggregate model task distribution from sample.
    Weighted by 30-day download count (more meaningful than raw model count).
    """
    tasks = defaultdict(lambda: {"count": 0, "downloads": 0})
    for m in models:
        tag = m.get("pipeline_tag") or "no-tag"
        dl  = m.get("downloads", 0) or 0
        tasks[tag]["count"]     += 1
        tasks[tag]["downloads"] += dl

    # Sort by downloads (descending)
    ranked = sorted(tasks.items(), key=lambda x: -x[1]["downloads"])

    result = []
    for tag, v in ranked[:MAX_TASKS]:
        label = tag.replace("-", " ").title()
        result.append({
            "task":      label,
            "count":     v["count"],
            "downloads": v["downloads"],
            "color":     TASK_COLORS.get(tag, "#6366f1"),
        })
    return result


def agg_licenses(models):
    """
    Extract license distribution from model tags.
    Tags like 'license:apache-2.0' → 'apache-2.0'.
    """
    lic = defaultdict(int)
    for m in models:
        for tag in m.get("tags", []):
            if tag.startswith("license:"):
                lic[tag[8:]] += 1
                break   # one licence per model

    ranked = sorted(lic.items(), key=lambda x: -x[1])
    total  = sum(lic.values()) or 1

    return [
        {
            "name":  k,
            "count": v,
            "pct":   round(v / total * 100, 1),
            "color": LICENSE_COLORS.get(k, "#4a5270"),
        }
        for k, v in ranked[:MAX_LICENSES]
    ]


def agg_frameworks(spaces):
    """Compute Space framework share by SDK field."""
    sdk = defaultdict(int)
    for s in spaces:
        k = (s.get("sdk") or "other").lower()
        sdk[k] += 1

    total  = sum(sdk.values()) or 1
    ranked = sorted(sdk.items(), key=lambda x: -x[1])

    return [
        {
            "name":  k.title(),
            "count": v,
            "pct":   round(v / total * 100, 1),
            "color": SDK_COLORS.get(k, "#4a5270"),
        }
        for k, v in ranked
    ]


def build_top_models(models, n=TOP_MODELS_N):
    result = []
    for i, m in enumerate(models[:n]):
        mid    = m.get("id", "")
        parts  = mid.split("/")
        creator= parts[0] if len(parts) > 1 else "—"
        name   = parts[-1]
        task   = m.get("pipeline_tag") or "unknown"
        label  = task.replace("-", " ").title()

        # licence from tags
        lic = "—"
        for tag in m.get("tags", []):
            if tag.startswith("license:"):
                lic = tag[8:]
                break

        result.append({
            "rank":      i + 1,
            "name":      name,
            "creator":   creator,
            "task":      label,
            "taskColor": TASK_COLORS.get(task, "#6366f1"),
            "dl":        fmt(m.get("downloads", 0) or 0),
            "likes":     fmt(m.get("likes",     0) or 0),
            "lic":       lic,
            "licColor":  LICENSE_COLORS.get(lic, "#4a5270"),
        })
    return result


def build_summary(models, spaces, tasks):
    total_dl   = sum(m.get("downloads", 0) or 0 for m in models)
    unique_tasks = len(tasks)

    return [
        {"label": "Models Sampled",    "value": fmt(len(models)),  "raw": len(models),  "trend": "top by downloads", "period": "scrape"},
        {"label": "30-Day Downloads",  "value": fmt(total_dl),     "raw": total_dl,     "trend": "from sample",      "period": "scrape"},
        {"label": "Tasks Identified",  "value": str(unique_tasks), "raw": unique_tasks, "trend": "unique categories", "period": "scrape"},
        {"label": "Spaces Sampled",    "value": fmt(len(spaces)),  "raw": len(spaces),  "trend": "top by likes",     "period": "scrape"},
        {"label": "SDKs / Frameworks", "value": str(len(set((s.get("sdk") or "other").lower() for s in spaces))),
                                               "raw": 4,          "trend": "distinct",  "period": "scrape"},
    ]


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    print("=" * 56, flush=True)
    print("  HF Hub Scraper — " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), flush=True)
    print("=" * 56, flush=True)

    models = fetch_models()
    spaces = fetch_spaces()
    tasks  = agg_tasks(models)

    elapsed = round(time.time() - t0, 1)

    output = {
        "meta": {
            "scraped":     datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source":      "huggingface.co/api",
            "sampleSize":  len(models),
            "totalRecords": len(models) + len(spaces),
            "runDuration": f"{elapsed}s",
            "nextRun":     (
                datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=6)
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "summary":    build_summary(models, spaces, tasks),
        "modelTasks": tasks,
        "licenses":   agg_licenses(models),
        "frameworks": agg_frameworks(spaces),
        "topModels":  build_top_models(models),
        # spaceCategories / growth / opportunities: populated by
        # dashboard fallback data — dedicated endpoints TBD
    }

    os.makedirs("data", exist_ok=True)
    path = "data/hub_stats.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size = os.path.getsize(path)
    print(f"\n✓ Wrote {path} ({size:,} bytes)", flush=True)
    print(f"  Tasks: {len(output['modelTasks'])} | "
          f"Top models: {len(output['topModels'])} | "
          f"Frameworks: {len(output['frameworks'])} | "
          f"Licenses: {len(output['licenses'])}", flush=True)
    print(f"  Run time: {elapsed}s", flush=True)


if __name__ == "__main__":
    run()
