"""Plan-tier labels: which subscription tier an account is on, and whether it
has a per-model (Fable) weekly window.

Two independent facts, kept separate on purpose:

- **Tier** comes from ``GET /api/oauth/profile`` (``organization.
  rate_limit_tier`` / ``organization.organization_type`` / ``account.
  has_claude_max`` / ``account.has_claude_pro``). It changes rarely, so it
  is fetched at most once per :data:`TIER_TTL_S` per account and cached in
  the account's ``sequence.json`` record next to ``alias`` /
  ``organizationName``.
- **Fable access** is read from the usage payload cswap already polls: a
  seat with per-model access carries a ``limits[]`` entry scoped to that
  model (surfaced as ``usage["scoped"]``), a seat without it carries only
  the unscoped 5h/7d windows. It is re-derived on every successful usage
  fetch, so a seat that silently loses the model window shows it within one
  poll.

Why both: two Team seats in one pool moved from a Fable-capable premium
tier to a standard tier overnight and nothing in cswap said so — the only
symptom was a missing usage line, indistinguishable from an incomplete
snapshot. With the tier labelled everywhere accounts are displayed, "seat
lost Fable access" reads as ``[Team standard · no Fable]`` instead of a
blank.

Everything here is pure (no I/O, no clock reads): callers pass ``now``.
Unknown tier strings are shown verbatim rather than hidden behind a
generic label, so a new tier the mapping table has never seen is still
visible to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

#: Re-fetch the profile at most this often per account (24 h). The tier
#: changes on a billing event, not per request; the usage poll is the
#: high-cadence signal and this rides on its successes.
TIER_TTL_S = 24 * 60 * 60

#: After a FAILED profile read, do not try again sooner than this (1 h) —
#: the read rides on every successful usage poll, and a seat whose profile
#: keeps failing must not cost one extra request per poll.
TIER_RETRY_S = 60 * 60

#: Model whose per-model weekly window decides the "Fable" / "no Fable" flag.
FABLE_MODEL_NAME = "Fable"

#: Human labels for the tiers observed so far (``rate_limit_tier`` values
#: under each ``organization_type``). Anything absent here renders its raw
#: ``rate_limit_tier`` string.
_TEAM_STANDARD_TIERS = frozenset({"default_raven"})

#: Compact forms for the menu bar (space is at a premium there).
_COMPACT = {
    "Team standard": "Team std",
    "Team premium": "Team prem",
    "Enterprise": "Ent",
}

#: Sequence-record keys this module owns. Kept in one place so a persist
#: that compares "did anything change?" and a JSON projection agree.
RECORD_KEYS = (
    "planTier",
    "rateLimitTier",
    "organizationType",
    "hasClaudeMax",
    "hasClaudePro",
    "tierFetchedAt",
    "tierAttemptedAt",
    "tierError",
    "fableAccess",
)

#: Shown in place of a tier when the profile endpoint answered 401 and no
#: earlier tier is cached — the access token is dead, not the tier unknown.
RELOGIN_TIER_TEXT = "tier: re-login needed"


@dataclass(frozen=True)
class PlanTier:
    """The tier facts read from one profile response."""

    label: str
    rate_limit_tier: str | None
    organization_type: str | None
    has_claude_max: bool | None
    has_claude_pro: bool | None


def _str_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def tier_label(
    rate_limit_tier: str | None,
    organization_type: str | None,
    has_claude_max: bool | None = None,
    has_claude_pro: bool | None = None,
) -> str | None:
    """Human tier label, or None when nothing usable was supplied.

    Observed mapping (18 seats, 2026-09-10)::

        default_claude_max_20x / claude_max   -> "Max 20x"
        default_claude_max_5x  / claude_max   -> "Max 5x"
        default_claude_max_5x  / claude_team  -> "Team premium"
        default_raven          / claude_team  -> "Team standard"

    Plus: ``claude_enterprise`` -> "Enterprise"; ``has_claude_pro`` (or a
    tier / org type naming ``pro``) -> "Pro". Enterprise and Team are
    decided by the org type first because a Team seat's tier string reuses
    the Max vocabulary (``default_claude_max_5x``), and Max is decided
    before Pro because a Max account may also report ``has_claude_pro``.
    Anything unrecognised returns the raw ``rate_limit_tier`` (or, failing
    that, the raw ``organization_type``) so nothing is hidden.
    """
    tier = (rate_limit_tier or "").strip().lower()
    org_type = (organization_type or "").strip().lower()

    if org_type == "claude_enterprise":
        return "Enterprise"
    if org_type == "claude_team":
        if tier in _TEAM_STANDARD_TIERS:
            return "Team standard"
        if "max" in tier:
            return "Team premium"
        return rate_limit_tier.strip() if rate_limit_tier and rate_limit_tier.strip() else "Team"
    if org_type == "claude_max" or has_claude_max or tier.startswith("default_claude_max"):
        if tier.endswith("20x"):
            return "Max 20x"
        if tier.endswith("5x"):
            return "Max 5x"
        if tier:
            return rate_limit_tier.strip()  # type: ignore[union-attr]
        return "Max"
    if has_claude_pro or org_type == "claude_pro" or "pro" in tier:
        return "Pro"
    if tier:
        return rate_limit_tier.strip()  # type: ignore[union-attr]
    if org_type:
        return organization_type.strip()  # type: ignore[union-attr]
    return None


def tier_from_profile(data: object) -> PlanTier | None:
    """Parse a raw ``/api/oauth/profile`` body into a :class:`PlanTier`.

    None when the body carries nothing tier-shaped (no ``organization`` and
    no ``account`` flags) — a schema change or an unexpected envelope must
    read as "unknown", never as a wrong label.
    """
    if not isinstance(data, dict):
        return None
    org = data.get("organization")
    acct = data.get("account")
    org = org if isinstance(org, dict) else {}
    acct = acct if isinstance(acct, dict) else {}
    rate_limit_tier = _str_or_none(org.get("rate_limit_tier"))
    organization_type = _str_or_none(org.get("organization_type"))
    has_max = _bool_or_none(acct.get("has_claude_max"))
    has_pro = _bool_or_none(acct.get("has_claude_pro"))
    label = tier_label(rate_limit_tier, organization_type, has_max, has_pro)
    if label is None:
        return None
    return PlanTier(
        label=label,
        rate_limit_tier=rate_limit_tier,
        organization_type=organization_type,
        has_claude_max=has_max,
        has_claude_pro=has_pro,
    )


def fable_access(usage: object, model: str = FABLE_MODEL_NAME) -> bool | None:
    """Whether a normalized usage dict carries the per-model window.

    True when ``usage["scoped"]`` names ``model`` (case-insensitive); False
    when the payload is a usage dict with window data but no such entry;
    None when there is no usage dict to judge from (fetch failed, sentinel,
    API-key account). The distinction matters: False is evidence the seat
    lacks the model, None is the absence of evidence.
    """
    if not isinstance(usage, dict) or not usage:
        return None
    wanted = model.lower()
    scoped = usage.get("scoped")
    if isinstance(scoped, list):
        for entry in scoped:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                if entry["name"].strip().lower() == wanted:
                    return True
    if any(isinstance(usage.get(k), dict) for k in ("five_hour", "seven_day")):
        return False
    return None


def _iso_z(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _parse_iso(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def tier_record_fields(tier: PlanTier, now: float) -> dict:
    """Sequence-record fields for a freshly fetched tier (camelCase, like the
    rest of the record). Clears ``tierError`` — a successful read supersedes
    any earlier failure."""
    return {
        "planTier": tier.label,
        "rateLimitTier": tier.rate_limit_tier,
        "organizationType": tier.organization_type,
        "hasClaudeMax": tier.has_claude_max,
        "hasClaudePro": tier.has_claude_pro,
        "tierFetchedAt": _iso_z(now),
        "tierAttemptedAt": _iso_z(now),
        "tierError": None,
    }


def tier_error_fields(error: str, now: float) -> dict:
    """Record fields for a failed profile read: the error kind and the
    attempt time (which paces the retry). The cached tier is not touched,
    so a transient failure never blanks a known label."""
    return {"tierError": error, "tierAttemptedAt": _iso_z(now)}


def fable_record_fields(access: bool | None) -> dict:
    """Record field for the Fable flag; ``None`` (no evidence) is not written
    over a known value by :func:`merge_record_fields`."""
    return {} if access is None else {"fableAccess": access}


def merge_record_fields(record: dict, fields: dict) -> bool:
    """Apply ``fields`` onto an account record in place; True if it changed."""
    changed = False
    for key, value in fields.items():
        if record.get(key) != value or key not in record:
            record[key] = value
            changed = True
    return changed


def tier_fetched_at(record: dict | None) -> float | None:
    """Epoch seconds of the cached tier's fetch, if any."""
    return _parse_iso((record or {}).get("tierFetchedAt"))


def tier_due(
    record: dict | None,
    now: float,
    ttl: float = TIER_TTL_S,
    retry: float = TIER_RETRY_S,
) -> bool:
    """Whether the profile should be (re)fetched for this account now.

    Due when never successfully fetched or when the cached tier is older
    than ``ttl`` — unless an attempt (success or failure) was made within
    ``retry``, which paces a persistently failing read to once an hour
    instead of once per usage poll.
    """
    fetched = tier_fetched_at(record)
    if fetched is not None and now - fetched < ttl:
        return False
    attempted = _parse_iso((record or {}).get("tierAttemptedAt"))
    if attempted is not None and 0 <= now - attempted < retry:
        return False
    return True


def tier_display(
    record: dict | None, *, compact: bool = False, model: str = FABLE_MODEL_NAME
) -> str | None:
    """The text inside the bracketed label: ``"Team premium · Fable"``.

    ``compact`` gives the menu-bar form (``"Team std"``, Fable flag shown
    only as ``"no Fable"`` since presence is the norm there). None when
    nothing is known yet, so surfaces can omit the label for accounts never
    profiled (API-key slots have no tier to show). A 401 on the profile with
    no cached tier renders :data:`RELOGIN_TIER_TEXT`.
    """
    if not isinstance(record, dict):
        return None
    tier = record.get("planTier")
    access = record.get("fableAccess")
    err = record.get("tierError")
    return format_tier(
        tier if isinstance(tier, str) else None,
        access if isinstance(access, bool) else None,
        err if isinstance(err, str) else None,
        compact=compact,
        model=model,
    )


def format_tier(
    tier: str | None,
    fable_access: bool | None,
    tier_error: str | None = None,
    *,
    compact: bool = False,
    model: str = FABLE_MODEL_NAME,
) -> str | None:
    """:func:`tier_display` on already-extracted values (what interactive
    surfaces hold on their snapshot rows)."""
    tier = tier.strip() if isinstance(tier, str) and tier.strip() else None
    if tier is None and tier_error == "http-401":
        return RELOGIN_TIER_TEXT
    parts: list[str] = []
    if tier is not None:
        parts.append(_COMPACT.get(tier, tier) if compact else tier)
    if fable_access is True and not compact:
        parts.append(model)
    elif fable_access is False:
        parts.append(f"no {model}")
    return " · ".join(parts) if parts else None


def tier_json_fields(record: dict | None) -> dict:
    """Additive ``--json`` keys for an account row. Always emitted (null when
    unknown) so scripts can key on them without probing for presence."""
    rec = record if isinstance(record, dict) else {}
    tier = rec.get("planTier")
    fetched_at = rec.get("tierFetchedAt")
    out: dict = {
        "planTier": tier if isinstance(tier, str) else None,
        "rateLimitTier": rec.get("rateLimitTier") if isinstance(rec.get("rateLimitTier"), str) else None,
        "organizationType": rec.get("organizationType") if isinstance(rec.get("organizationType"), str) else None,
        "fableAccess": rec.get("fableAccess") if isinstance(rec.get("fableAccess"), bool) else None,
        "tierFetchedAt": fetched_at if isinstance(fetched_at, str) else None,
    }
    err = rec.get("tierError")
    if isinstance(err, str) and err:
        out["tierError"] = err
    return out


__all__ = [
    "FABLE_MODEL_NAME",
    "PlanTier",
    "RECORD_KEYS",
    "RELOGIN_TIER_TEXT",
    "TIER_RETRY_S",
    "TIER_TTL_S",
    "fable_access",
    "fable_record_fields",
    "format_tier",
    "merge_record_fields",
    "tier_display",
    "tier_due",
    "tier_error_fields",
    "tier_fetched_at",
    "tier_from_profile",
    "tier_json_fields",
    "tier_label",
    "tier_record_fields",
]
