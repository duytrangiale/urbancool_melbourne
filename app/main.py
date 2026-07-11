"""UrbanCool Melbourne — FastAPI backend for the console dashboard.

Serves the pre-built static console (app/static/index.html, built by
app/build_static.py) plus the two genuinely dynamic endpoints: live What-If
re-prediction and per-suburb SHAP waterfall rendering.

Run with:

    uvicorn app.main:app --reload          # dev
    uvicorn app.main:app --host 0.0.0.0 --port 7860   # production / HF Spaces
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found in any parent)")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core import all_suburbs, complete_suburbs, compute_whatif, get_snapshot, render_shap_waterfall_png

STATIC_DIR = PROJECT_ROOT / "app" / "static"

app = FastAPI(title="UrbanCool Melbourne")


class WhatIfRequest(BaseModel):
    suburb: str
    extra_tree: float = 0.0
    extra_veg: float = 0.0
    green_conversion: float = 0.0


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Static console not built yet — run `python -m app.build_static` first.",
        )
    return FileResponse(index_path)


@app.get("/api/suburbs")
def api_suburbs():
    return {"suburbs": all_suburbs(), "complete_suburbs": complete_suburbs()}


@app.get("/api/snapshot/{suburb}")
def api_snapshot(suburb: str):
    try:
        return get_snapshot(suburb)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown suburb: {suburb!r}")


@app.post("/api/whatif")
def api_whatif(req: WhatIfRequest):
    try:
        return compute_whatif(req.suburb, req.extra_tree, req.extra_veg, req.green_conversion)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown suburb: {req.suburb!r}")


@app.get("/api/shap/{suburb}")
def api_shap(suburb: str):
    try:
        png_bytes = render_shap_waterfall_png(suburb)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown suburb: {suburb!r}")
    return Response(content=png_bytes, media_type="image/png")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Mount last so it doesn't shadow the API routes above.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
