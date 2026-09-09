"""HTTP API — a thin wrapper over :mod:`finrec.service`.

Deliberately contains no business logic. Every endpoint resolves to a function
in the service layer, so the eventual web frontend and the current Streamlit UI
can never drift apart in what they compute.

Run with::

    uvicorn finrec.api:app --host 127.0.0.1 --reload

FastAPI is an optional dependency; importing this module without it raises a
clear message rather than an opaque ImportError deep in a traceback.
"""

from __future__ import annotations

from typing import Any
import contextvars
import hmac
import ipaddress
import os
import sqlite3

try:
    from fastapi import Body, FastAPI, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The HTTP API needs FastAPI. Install it with:  pip install 'fastapi[standard]'"
    ) from exc

from . import __version__, db, service, storage
from .profile import Profile

app = FastAPI(
    title="Finance Planner API",
    version=__version__,
    description=(
        "Every analysis takes the minimum inputs and fills in the rest from your location. "
        "Responses are shaped {simple, advanced, assumptions, meta} — render `simple` by "
        "default and `advanced` behind a disclosure."
    ),
)

# The browser frontend will be served from a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get(
        "FINREC_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
        if origin.strip() and origin.strip() != "*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

_identity = contextvars.ContextVar("verified_finrec_user", default=None)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def deployment_gate(request: Request, call_next):
    """Local socket + Host checks, or one explicitly configured token account.

    Forwarded headers and caller-supplied user identifiers are never identity.
    This is not a general hosted multi-user authentication implementation.
    """
    credential = os.environ.get("FINREC_API_TOKEN", "")
    if credential:
        account = os.environ.get("FINREC_API_USER", "")
        if not account:
            return JSONResponse({"detail": "FINREC_API_USER is required with FINREC_API_TOKEN"}, status_code=503)
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth.encode(), f"Bearer {credential}".encode()):
            return JSONResponse({"detail": "Valid bearer authentication required"}, status_code=401)
    else:
        if not request.client or not _loopback(request.client.host) or not _loopback(request.url.hostname or ""):
            return JSONResponse({"detail": "API is local-only; configure authenticated deployment explicitly"}, status_code=403)
        origin = request.headers.get("origin")
        allowed = os.environ.get("FINREC_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
        if origin and origin not in [s.strip() for s in allowed]:
            return JSONResponse({"detail": "Untrusted browser origin"}, status_code=403)
        account = os.environ.get("FINREC_USER") or db.DEFAULT_USER_EMAIL
    try:
        user_id = db.ensure_user(account)
    except sqlite3.Error:
        return JSONResponse({"detail": "Storage unavailable; retry later"}, status_code=503)
    token = _identity.set(user_id)
    try:
        return await call_next(request)
    finally:
        _identity.reset(token)


@app.exception_handler(storage.ConflictError)
async def conflict_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(ValueError)
async def validation_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(sqlite3.Error)
async def storage_error(request, exc):
    return JSONResponse({"detail": "Storage unavailable; retry later"}, status_code=503)


@app.get("/health")
def health() -> dict:
    """Liveness plus the state of the store, so a deploy can be verified."""
    store = db.health()
    return {
        "status": "ok" if store.get("ok") else "degraded",
        "version": __version__,
        "store": {k: store.get(k) for k in
                  ("ok", "integrity", "schema_version", "users", "plans", "versions")},
    }


@app.get("/api/defaults")
def defaults() -> dict:
    """The blank-slate profile, so a client never has to hardcode field names."""
    return {"profile": Profile().to_dict(), "services": sorted(service.SERVICES)}


@app.get("/api/location")
def location(q: str = Query(..., description="City, state, or ZIP"),
             live: bool = Query(False)) -> dict:
    """Everything we can infer about a place. Powers onboarding auto-fill."""
    return service.location_preview(q, live=live)


@app.post("/api/profile")
def create_profile(payload: dict[str, Any] = Body(...)) -> dict:
    """Build a complete profile from salary + location."""
    if not payload.get("salary"):
        raise HTTPException(status_code=422, detail="salary is required")
    if not payload.get("location"):
        raise HTTPException(status_code=422, detail="location is required")
    return service.create_profile(payload)


# --------------------------------------------------------------------------
# Saved plans
# --------------------------------------------------------------------------
# Declared before the catch-all below, because FastAPI matches in registration
# order and ``/api/{analysis:path}`` would otherwise swallow every one of these.
#
# ``user`` is retained as a compatibility parameter but never grants identity.
# Only deployment_gate resolves the local account or verified token account.


def _user_id(user: str | None) -> int:
    verified = _identity.get()
    if verified is None:
        raise HTTPException(status_code=401, detail="Verified identity required")
    return verified


@app.get("/api/plans")
def list_plans(user: str | None = Query(None)) -> dict:
    return service.list_plans_service({"user_id": _user_id(user)})


@app.get("/api/plans/{slug}")
def get_plan(slug: str, user: str | None = Query(None),
             version: int | None = Query(None)) -> dict:
    result = service.load_plan_service(
        {"slug": slug, "version": version, "user_id": _user_id(user)})
    if not result["simple"]["found"]:
        raise HTTPException(status_code=404, detail=f"No plan '{slug}'")
    return result


@app.get("/api/plans/{slug}/history")
def plan_history(slug: str, user: str | None = Query(None),
                 limit: int = Query(50, ge=1, le=500)) -> dict:
    return service.plan_history_service(
        {"slug": slug, "limit": limit, "user_id": _user_id(user)})


@app.post("/api/plans")
def save_plan(payload: dict[str, Any] = Body(...), user: str | None = Query(None)) -> dict:
    if not payload.get("profile"):
        raise HTTPException(status_code=422, detail="profile is required")
    return service.save_plan_service({**payload, "user_id": _user_id(user)})


@app.post("/api/plans/{slug}/restore/{version}")
def restore_plan(slug: str, version: int, user: str | None = Query(None),
                 expected_version: int | None = Query(None, ge=1)) -> dict:
    result = service.restore_plan_service(
        {"slug": slug, "version": version, "user_id": _user_id(user),
         "expected_version": expected_version})
    if not result["simple"]["restored"]:
        raise HTTPException(status_code=404, detail=f"No version {version} of '{slug}'")
    return result


@app.delete("/api/plans/{slug}")
def delete_plan(slug: str, user: str | None = Query(None),
                purge: bool = Query(False), expected_version: int | None = Query(None, ge=1)) -> dict:
    result = service.delete_plan_service(
        {"slug": slug, "purge": purge, "user_id": _user_id(user), "expected_version": expected_version})
    if not result["simple"]["deleted"]:
        raise HTTPException(status_code=404, detail=f"No plan '{slug}'")
    return result


@app.get("/api/export")
def export_everything(user: str | None = Query(None)) -> dict:
    """Every plan and its history. Users must be able to take their data."""
    return service.export_service({"user_id": _user_id(user)})


@app.post("/api/import")
def import_everything(payload: dict[str, Any] = Body(...)) -> dict:
    return service.import_service({**payload, "user_id": _user_id(None)})


@app.post("/api/{analysis:path}")
def run_analysis(analysis: str, payload: dict[str, Any] = Body(default={})) -> dict:
    """Dispatch to any service by name. See ``/api/defaults`` for the list."""
    handler = service.SERVICES.get(analysis)
    if handler is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown analysis '{analysis}'. Available: {sorted(service.SERVICES)}",
        )
    try:
        clean = {k: v for k, v in payload.items() if k not in {"email", "user", "user_id"}}
        clean["user_id"] = _user_id(None)
        return handler(clean)
    except storage.ConflictError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
