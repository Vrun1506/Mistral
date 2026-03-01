import json
import os
from typing import Any

import config  # noqa: F401 — loads .env on import
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException

from routers.cookies import router as cookies_router

app = FastAPI()
app.include_router(cookies_router)

# ---------------------------------------------------------------------------
# Topic data — loaded at startup
# ---------------------------------------------------------------------------


TOPIC_GROUPS: dict[str, Any] = {}
HIERARCHY: dict[str, Any] = {}

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GROUPS_PATH = os.path.join(DATA_DIR, "topic_groups.json")
HIERARCHY_PATH = os.path.join(DATA_DIR, "topic_hierarchy.json")

templates = Jinja2Templates(directory=os.path.join(DATA_DIR, "templates"))


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
# Tree helpers
# ---------------------------------------------------------------------------


def _build_tree() -> list[dict[str, Any]]:
    """Build enriched tree with segment counts at every level."""
    tree: list[dict[str, Any]] = []
    for root_name, subcats in HIERARCHY.items():
        root: dict[str, Any] = {"name": root_name, "subcategories": [], "segment_count": 0}
        for sub_name, labels in subcats.items():
            sub: dict[str, Any] = {"name": sub_name, "topics": [], "segment_count": 0}
            for label in labels:
                info = TOPIC_GROUPS.get(label, {})
                count = len(info.get("segments", []))
                sub["topics"].append(
                    {
                        "label": label,
                        "keywords": info.get("keywords", [])[:5],
                        "segment_count": count,
                    }
                )
                sub["segment_count"] += count
            root["subcategories"].append(sub)
            root["segment_count"] += sub["segment_count"]
        tree.append(root)
    return tree


def _lookup_breadcrumb(label: str) -> tuple[str | None, str | None]:
    """Find the root category and subcategory for a given topic label."""
    for root_name, subcats in HIERARCHY.items():
        for sub_name, labels in subcats.items():
            if label in labels:
                return root_name, sub_name
    return None, None


# ---------------------------------------------------------------------------
# Topic browsing endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Home page: expandable topic tree."""
    tree = _build_tree()
    total_topics = sum(len(t) for sub in HIERARCHY.values() for t in sub.values())
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "tree": tree,
            "total_topics": total_topics,
        },
    )


@app.get("/topic/{label}", response_class=HTMLResponse)
async def topic_detail(request: Request, label: str) -> HTMLResponse:
    """Show all segments in a topic group."""
    info = TOPIC_GROUPS.get(label)
    if not info:
        raise HTTPException(status_code=404, detail="Topic not found")
    root_cat, sub_cat = _lookup_breadcrumb(label)
    return templates.TemplateResponse(
        request,
        "topic.html",
        {
            "label": label,
            "keywords": info["keywords"],
            "segments": info["segments"],
            "root_cat": root_cat,
            "sub_cat": sub_cat,
        },
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    _load_topic_data()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
