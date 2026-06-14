"""
Shared feedback router — creates Forgejo issues from in-app beta feedback.
Products include this with make_feedback_router(repo=..., product=...).
"""
from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_LABEL_COLORS: dict[str, str] = {
    "beta-feedback": "#0075ca",
    "needs-triage": "#e4e669",
    "bug": "#d73a4a",
    "feature-request": "#a2eeef",
    "question": "#d876e3",
}

_TYPE_LABEL_MAP: dict[str, str] = {"bug": "bug", "feature": "feature-request"}
_TYPE_DISPLAY: dict[str, str] = {
    "bug": "🐛 Bug",
    "feature": "✨ Feature Request",
    "other": "💬 Other",
}


class FeedbackRequest(BaseModel):
    title: str
    description: str
    type: Literal["bug", "feature", "other"] = "other"
    repro: str = ""
    tab: str = "unknown"
    submitter: str = ""


class FeedbackResponse(BaseModel):
    issue_number: int
    issue_url: str


def _forgejo_token() -> str:
    """Return the bot token when set; fall back to the personal API token."""
    return os.environ.get("FORGEJO_BOT_TOKEN") or os.environ.get("FORGEJO_API_TOKEN", "")


def _forgejo_headers() -> dict[str, str]:
    return {"Authorization": f"token {_forgejo_token()}", "Content-Type": "application/json"}


def _ensure_labels(label_names: list[str], base: str, repo: str) -> list[int]:
    headers = _forgejo_headers()
    resp = requests.get(f"{base}/repos/{repo}/labels", headers=headers, timeout=10)
    existing = {lb["name"]: lb["id"] for lb in resp.json()} if resp.ok else {}
    ids: list[int] = []
    for name in label_names:
        if name in existing:
            ids.append(existing[name])
        else:
            r = requests.post(
                f"{base}/repos/{repo}/labels",
                headers=headers,
                json={"name": name, "color": _LABEL_COLORS.get(name, "#ededed")},
                timeout=10,
            )
            if r.ok:
                ids.append(r.json()["id"])
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to create label '{name}': {r.text[:200]}",
                )
    return ids


def _collect_context(tab: str, product: str) -> dict[str, str]:
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=Path.cwd(),
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        version = "dev"
    return {
        "product": product,
        "tab": tab,
        "version": version,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _build_issue_body(payload: FeedbackRequest, context: dict[str, str]) -> str:
    lines: list[str] = [
        f"## {_TYPE_DISPLAY.get(payload.type, '💬 Other')}",
        "",
        payload.description,
        "",
    ]
    if payload.type == "bug" and payload.repro:
        lines += ["### Reproduction Steps", "", payload.repro, ""]
    lines += ["### Context", ""]
    for k, v in context.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    if payload.submitter:
        lines += ["---", f"*Submitted by: {payload.submitter}*"]
    return "\n".join(lines)


def make_feedback_router(
    repo: str,
    product: str,
    demo_mode_fn: Callable[[], bool] | None = None,
) -> APIRouter:
    """Return a configured feedback APIRouter for the given Forgejo repo and product.

    Args:
        repo: Forgejo repo slug, e.g. "Circuit-Forge/kiwi".
        product: Product name injected into issue context, e.g. "kiwi".
        demo_mode_fn: Optional callable returning True when in demo mode.
            If None, reads the DEMO_MODE environment variable.
    """

    def _is_demo() -> bool:
        if demo_mode_fn is not None:
            return demo_mode_fn()
        return os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes")

    router = APIRouter()

    @router.get("/status")
    def feedback_status() -> dict:
        """Return whether feedback submission is configured on this instance."""
        return {"enabled": bool(_forgejo_token()) and not _is_demo()}

    @router.post("", response_model=FeedbackResponse)
    def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
        """File a Forgejo issue from in-app feedback."""
        token = _forgejo_token()
        if not token:
            raise HTTPException(
                status_code=503,
                detail="Feedback disabled: FORGEJO_BOT_TOKEN (or FORGEJO_API_TOKEN) not configured.",
            )
        if _is_demo():
            raise HTTPException(status_code=403, detail="Feedback disabled in demo mode.")

        base = os.environ.get(
            "FORGEJO_API_URL", "https://git.opensourcesolarpunk.com/api/v1"
        )
        context = _collect_context(payload.tab, product)
        body = _build_issue_body(payload, context)
        labels = [
            "beta-feedback",
            "needs-triage",
            _TYPE_LABEL_MAP.get(payload.type, "question"),
        ]
        label_ids = _ensure_labels(labels, base, repo)

        resp = requests.post(
            f"{base}/repos/{repo}/issues",
            headers=_forgejo_headers(),
            json={"title": payload.title, "body": body, "labels": label_ids},
            timeout=15,
        )
        if not resp.ok:
            raise HTTPException(
                status_code=502, detail=f"Forgejo error: {resp.text[:200]}"
            )
        data = resp.json()
        return FeedbackResponse(issue_number=data["number"], issue_url=data["html_url"])

    return router
