import json
import os
import re
from typing import Any

import uvicorn
from dotenv import load_dotenv

import config  # noqa: F401 — loads .env on import

load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from routers.cookies import router as cookies_router
from routers.graph import router as graph_router
from routers.pipeline import router as pipeline_router
from routers.topics import router as topics_router

app = FastAPI()

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ["ALLOWED_ORIGINS"].split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(cookies_router)
app.include_router(pipeline_router)
app.include_router(graph_router)
app.include_router(topics_router)

# ---------------------------------------------------------------------------
# Topic data — loaded at startup
# ---------------------------------------------------------------------------


TOPIC_GROUPS: dict[str, Any] = {}
HIERARCHY: dict[str, Any] = {}

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GROUPS_PATH = os.path.join(DATA_DIR, "topic_groups.json")
HIERARCHY_PATH = os.path.join(DATA_DIR, "topic_hierarchy.json")

FRONTEND_URL = os.environ["FRONTEND_URL"]


def _load_topic_data() -> None:
    """Load cached topic data files if they exist."""
    global TOPIC_GROUPS, HIERARCHY
    if os.path.exists(GROUPS_PATH):
        with open(GROUPS_PATH) as f:
            TOPIC_GROUPS = json.load(f)
        print(f"Loaded {len(TOPIC_GROUPS)} topic groups.")
    if os.path.exists(HIERARCHY_PATH):
        with open(HIERARCHY_PATH) as f:
            HIERARCHY = json.load(f)
        print(f"Loaded {len(HIERARCHY)} root categories.")
    if TOPIC_GROUPS and not HIERARCHY:
        print("No hierarchy found. Generating via Mistral...")
        from services.pipeline.pipeline import build_hierarchy, get_client

        client = get_client()
        HIERARCHY = build_hierarchy(client, TOPIC_GROUPS)
        with open(HIERARCHY_PATH, "w") as f:
            json.dump(HIERARCHY, f, indent=2)
        print(f"Saved topic_hierarchy.json ({len(HIERARCHY)} root categories).")


# ---------------------------------------------------------------------------
# Sensitive-topic filter
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Health & medical
        r"\b(illness|disease|disorder|diagnosis|symptom|chronic|cancer|tumor|diabetes)\b",
        r"\b(medication|prescription|therapy|therapist|psychiatr|psycholog)\b",
        r"\b(surgery|hospital|clinic|medical|patient)\b",
        # Mental health
        r"\b(depress|anxiety|bipolar|adhd|autism|ptsd|ocd|schizophren)\b",
        r"\b(mental\s*health|suicid|self[- ]?harm|eating\s*disorder|addiction)\b",
        r"\b(trauma|panic\s*attack|insomnia|burnout)\b",
        # Relationships & family
        r"\b(divorce|breakup|break[- ]?up|infidelity|cheating|affair)\b",
        r"\b(marriage\s*problem|relationship\s*(issue|problem|trouble))\b",
        r"\b(domestic\s*(violence|abuse)|child\s*custody|alimony)\b",
        r"\b(family\s*(conflict|issue|problem|trouble|drama))\b",
        # Sexuality & gender
        r"\b(sexual\s*(orientation|identity|preference|health))\b",
        r"\b(gender\s*(identity|dysphoria|transition))\b",
        r"\b(fertility|pregnan|abortion|miscarriage|contracepti)\b",
        # Personal identity / sensitive demographics
        r"\b(religion|religious\s*belief|political\s*affiliation)\b",
        r"\b(criminal\s*record|arrest|conviction|incarcerat)\b",
        r"\b(debt|bankruptcy|financial\s*(trouble|hardship|distress))\b",
        # Substance use
        r"\b(drug\s*(use|abuse|addict)|alcohol|substance\s*abuse|rehab)\b",
    ]
]


def _is_sensitive(label: str, keywords: list[str]) -> bool:
    """Return True if a topic label or any of its keywords match sensitive patterns."""
    texts = [label, *keywords]
    for text in texts:
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                return True
    return False


# ---------------------------------------------------------------------------
# Redirects — all UI lives in the frontend (localhost:3000)
# ---------------------------------------------------------------------------


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse(f"{FRONTEND_URL}/dashboard")


@app.get("/graph")
async def graph_page() -> RedirectResponse:
    return RedirectResponse(f"{FRONTEND_URL}/dashboard")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    _load_topic_data()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
