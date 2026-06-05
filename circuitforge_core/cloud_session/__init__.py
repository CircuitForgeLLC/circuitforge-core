"""
circuitforge_core.cloud_session — shared cloud session resolution for all CF products.

Usage (FastAPI product):

    from circuitforge_core.cloud_session import CloudSessionFactory
    from pathlib import Path

    _sessions = CloudSessionFactory(
        product="avocet",
        local_db=Path("data/avocet.db"),
    )
    get_session = _sessions.dependency()
    require_tier = _sessions.require_tier

    @router.get("/api/imitate")
    def imitate(session: CloudUser = Depends(get_session)):
        # session.user_id is the Directus UUID for cloud users, "local" for self-hosted
        ...

Environment variables (set per-product via .env / compose):
    CLOUD_MODE              1/true/yes to enable cloud auth (default: off)
    CLOUD_DATA_ROOT         Root directory for per-user data (default: /devl/<product>-cloud-data)
    DIRECTUS_JWT_SECRET     HS256 secret used to sign cf_session JWTs (required in cloud mode)
    HEIMDALL_URL            License server base URL (default: https://license.circuitforge.tech)
    HEIMDALL_ADMIN_TOKEN    Heimdall admin bearer token (required for tier resolution)
    CF_SERVER_SECRET        Server-side secret for deriving per-user encryption keys
    CLOUD_AUTH_BYPASS_IPS   Comma-separated IPs/CIDRs to skip JWT auth (dev LAN only)
"""
from __future__ import annotations

import ipaddress
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from starlette.requests import Request as _Request
    from starlette.responses import Response as _Response
except ImportError:  # pragma: no cover — starlette may be absent in non-web envs
    _Request = Any  # type: ignore[assignment,misc]
    _Response = Any  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)

TIERS: list[str] = ["free", "paid", "premium", "ultra"]

# ── CloudUser ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CloudUser:
    """Resolved user identity for one HTTP request.

    user_id:  Directus UUID for authenticated cloud users.
              "local"          for self-hosted / CLOUD_MODE=false.
              "local-dev"      for dev-bypass-IP sessions.
              "anon-<uuid>"    for unauthenticated guest visitors.
    tier:     free | paid | premium | ultra | local
    product:  Which CF product this session belongs to (e.g. "avocet").
    meta:     Product-specific extras (e.g. household_id for Kiwi).
              Access via session.meta.get("household_id").
    """
    user_id: str
    tier: str
    product: str
    has_byok: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_bypass_nets(raw: str) -> tuple[list[ipaddress.IPv4Network | ipaddress.IPv6Network], frozenset[str]]:
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    ips: set[str] = set()
    for entry in (e.strip() for e in raw.split(",") if e.strip()):
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            ips.add(entry)
    return nets, frozenset(ips)


def _is_bypass_ip(
    ip: str,
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    ips: frozenset[str],
) -> bool:
    if not ip or (not nets and not ips):
        return False
    if ip in ips:
        return True
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in nets)
    except ValueError:
        return False


def _extract_session_token(header_value: str) -> str:
    """Pull cf_session value out of a raw Cookie header or return the value as-is."""
    m = re.search(r'(?:^|;)\s*cf_session=([^;]+)', header_value)
    return m.group(1).strip() if m else header_value.strip()


# ── CloudSessionFactory ───────────────────────────────────────────────────────


class CloudSessionFactory:
    """Per-product session factory. Instantiate once at module level.

    Args:
        product:          Product code string (e.g. "avocet", "kiwi").
        extra_meta:       Optional async-or-sync callable that receives
                          (user_id: str, tier: str) and returns a dict merged
                          into CloudUser.meta.  Use for product-specific fields
                          like household_id.
        byok_detector:    Callable() → bool.  Override to detect BYOK for this
                          product's config path.  Default: always False.
    """

    def __init__(
        self,
        product: str,
        extra_meta: Callable[[str, str], dict[str, Any]] | None = None,
        byok_detector: Callable[[], bool] | None = None,
    ) -> None:
        self.product = product
        self._extra_meta = extra_meta
        self._byok_detector = byok_detector or (lambda: False)

        # Config — read from environment at construction time so tests can patch env
        self._cloud_mode: bool = os.environ.get("CLOUD_MODE", "").lower() in ("1", "true", "yes")
        self._directus_secret: str = os.environ.get("DIRECTUS_JWT_SECRET", "")
        self._heimdall_url: str = os.environ.get("HEIMDALL_URL", "https://license.circuitforge.tech")
        self._heimdall_token: str = os.environ.get("HEIMDALL_ADMIN_TOKEN", "")
        self._cloud_data_root: Path = Path(
            os.environ.get("CLOUD_DATA_ROOT", f"/devl/{product}-cloud-data")
        )

        _bypass_raw = os.environ.get("CLOUD_AUTH_BYPASS_IPS", "")
        self._bypass_nets, self._bypass_ips = _parse_bypass_nets(_bypass_raw)

        # Tier resolution cache: {user_id: (result_dict, timestamp)}
        self._tier_cache: dict[str, tuple[dict, float]] = {}
        self._tier_cache_ttl: float = 300.0  # 5 minutes

    # ── JWT ───────────────────────────────────────────────────────────────────

    def validate_jwt(self, token: str) -> str:
        """Validate a cf_session JWT and return the Directus user_id. Raises HTTPException on failure."""
        try:
            import jwt as pyjwt  # lazy — not needed in local mode
            from fastapi import HTTPException
            payload = pyjwt.decode(
                token,
                self._directus_secret,
                algorithms=["HS256"],
                options={"require": ["id", "exp"]},
            )
            return payload["id"]
        except Exception as exc:
            log.debug("JWT validation failed: %s", exc)
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail="Session invalid or expired")

    # ── Heimdall ──────────────────────────────────────────────────────────────

    def _ensure_provisioned(self, user_id: str) -> None:
        if not self._heimdall_token:
            return
        try:
            import requests
            requests.post(
                f"{self._heimdall_url}/admin/provision",
                json={"directus_user_id": user_id, "product": self.product, "tier": "free"},
                headers={"Authorization": f"Bearer {self._heimdall_token}"},
                timeout=5,
            )
        except Exception as exc:
            log.warning("Heimdall provision failed for user %s: %s", user_id, exc)

    def _resolve_tier(self, user_id: str) -> dict[str, Any]:
        """Returns dict with keys: tier, license_key (and any product extras)."""
        now = time.monotonic()
        cached = self._tier_cache.get(user_id)
        if cached and (now - cached[1]) < self._tier_cache_ttl:
            return cached[0]

        result: dict[str, Any] = {"tier": "free", "license_key": None}
        if self._heimdall_token:
            try:
                import requests
                resp = requests.post(
                    f"{self._heimdall_url}/admin/cloud/resolve",
                    json={"directus_user_id": user_id, "product": self.product},
                    headers={"Authorization": f"Bearer {self._heimdall_token}"},
                    timeout=5,
                )
                if resp.ok:
                    data = resp.json()
                    result["tier"] = data.get("tier", "free")
                    result["license_key"] = data.get("key_display")
                    # Forward any extra fields Heimdall returns (household_id etc.)
                    result.update({k: v for k, v in data.items() if k not in result})
            except Exception as exc:
                log.warning("Heimdall tier resolve failed for %s: %s", user_id, exc)
        else:
            log.debug("HEIMDALL_ADMIN_TOKEN not set — defaulting tier to free")

        self._tier_cache[user_id] = (result, now)
        return result

    # ── Guest sessions ────────────────────────────────────────────────────────

    _GUEST_COOKIE = "cf_guest_id"
    _GUEST_COOKIE_MAX_AGE = 60 * 60 * 24 * 90  # 90 days

    def _resolve_guest(self, request: Any, response: Any) -> CloudUser:
        guest_id = (request.cookies.get(self._GUEST_COOKIE) or "").strip()
        if not guest_id:
            guest_id = str(uuid.uuid4())
        is_https = request.headers.get("x-forwarded-proto", "http").lower() == "https"
        response.set_cookie(
            key=self._GUEST_COOKIE,
            value=guest_id,
            max_age=self._GUEST_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=is_https,
        )
        return CloudUser(
            user_id=f"anon-{guest_id}",
            tier="free",
            product=self.product,
            has_byok=self._byok_detector(),
        )

    # ── Core resolver ─────────────────────────────────────────────────────────

    def resolve(self, request: Any, response: Any) -> CloudUser:
        """Resolve the CloudUser for a FastAPI request. Suitable as a Depends() target."""
        has_byok = self._byok_detector()

        if not self._cloud_mode:
            return CloudUser(user_id="local", tier="local", product=self.product, has_byok=has_byok)

        client_ip = (
            request.headers.get("x-real-ip", "")
            or (request.client.host if request.client else "")
        )
        is_bypass = _is_bypass_ip(client_ip, self._bypass_nets, self._bypass_ips)

        raw_session = (
            request.headers.get("x-cf-session", "").strip()
            or request.cookies.get("cf_session", "").strip()
        )

        # Bypass IPs skip the JWT *requirement* but not JWT *validation*.
        # If a token is present (dev is logged in), honour it so they land on
        # their own account DB rather than the shared local-dev DB.
        if not raw_session:
            if is_bypass:
                log.debug("Bypass IP %s, no token — returning local-dev session for product %s", client_ip, self.product)
                return CloudUser(user_id="local-dev", tier="local", product=self.product, has_byok=has_byok)
            return self._resolve_guest(request, response)

        token = _extract_session_token(raw_session)
        if not token:
            return self._resolve_guest(request, response)

        # Soft-fail on invalid/expired JWT: downgrade to guest rather than
        # hard-erroring with 401.  Public endpoints (e.g. community blocklist)
        # should remain accessible even when the browser has a stale cookie.
        # Routes that genuinely require an authenticated identity should gate
        # themselves with require_tier() — that's where the 401/403 belongs.
        try:
            user_id = self.validate_jwt(token)
        except Exception:
            log.warning(
                "JWT validation failed for product %s (expired or tampered) — falling back to guest",
                self.product,
            )
            return self._resolve_guest(request, response)

        self._ensure_provisioned(user_id)
        tier_data = self._resolve_tier(user_id)
        tier = tier_data.get("tier", "free")

        meta: dict[str, Any] = {}
        if self._extra_meta:
            meta = self._extra_meta(user_id, tier) or {}
        # Merge any extra fields from Heimdall response (e.g. household_id)
        meta.update({k: v for k, v in tier_data.items() if k not in ("tier", "license_key")})
        meta["license_key"] = tier_data.get("license_key")

        return CloudUser(
            user_id=user_id,
            tier=tier,
            product=self.product,
            has_byok=has_byok,
            meta=meta,
        )

    def dependency(self) -> Callable[["_Request", "_Response"], CloudUser]:
        """Return a FastAPI-compatible dependency function (use with Depends())."""
        factory = self

        def _get_session(request: _Request, response: _Response) -> CloudUser:
            return factory.resolve(request, response)

        return _get_session

    def require_tier(self, min_tier: str) -> Callable:
        """Dependency factory — raises 403 if the session tier is below min_tier."""
        from fastapi import Depends, HTTPException
        min_idx = TIERS.index(min_tier)
        get_session = self.dependency()

        def _check(session: CloudUser = Depends(get_session)) -> CloudUser:
            if session.tier in ("local", "local-dev"):
                return session
            try:
                if TIERS.index(session.tier) < min_idx:
                    raise HTTPException(
                        status_code=403,
                        detail=f"This feature requires {min_tier} tier or above.",
                    )
            except ValueError:
                raise HTTPException(status_code=403, detail="Unknown tier.")
            return session

        return _check


# ── BYOK detection ────────────────────────────────────────────────────────────

def detect_byok(config_path: Path | None = None) -> bool:
    """Return True if at least one enabled non-vision LLM backend is configured.

    Reads the shared llm.yaml that LLMRouter uses. Local (Ollama, vLLM) and
    API-key backends both count — the policy is "user is supplying compute",
    regardless of where that compute lives.

    Args:
        config_path: Override the default config location. Useful in tests.
    """
    import yaml
    if config_path is None:
        config_path = Path.home() / ".config" / "circuitforge" / "llm.yaml"
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return any(
            b.get("enabled", True) and b.get("type") != "vision_service"
            for b in cfg.get("backends", {}).values()
        )
    except Exception:
        return False
