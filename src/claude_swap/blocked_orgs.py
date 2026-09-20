"""Blocked-organizations record: a reversible ranking filter for ``cswap auto``.

Measured origin (2026-09-20 05:19-07:03 CST): the active pool seat's
organization disabled Claude Code access. Every turn on that seat died with an
API error while its usage windows read 13% — so the engine, which ranks on
usage windows only, neither noticed nor moved. It failed over 1 h 44 min later
and only because the usage endpoint happened to return 429. Every other seat
of that organization stayed a silent-death candidate the whole time.

This module is the record the engine reads to keep such seats out of the
ranking. Three properties are load-bearing; each has a test that names it:

1. KEYED ON THE ORG LABEL exactly as ``cswap list`` prints it in brackets
   (the roster's ``organizationName``), never on the email domain. Pools hold
   addresses whose domain and organization differ, and an org's seats do not
   all share a domain. Matching is exact and case-sensitive; ``block-org``
   refuses a label no managed account carries, so a typo cannot install a
   block that gates nothing.

2. AN ENTRY NEVER EXPIRES INTO ELIGIBILITY ON A TIMER. There is no TTL and no
   ``until``: nothing in this module reads the clock to decide whether an
   entry applies, and unknown keys in an entry (a hand-added ``until``) are
   ignored rather than honoured. An entry clears by ``cswap unblock-org`` or
   by the watchdog's RESOLVED evidence, because a timer re-admits a dead seat
   silently — the failure this exists to stop. Each entry carries
   ``written_at`` (epoch seconds), ``written_by`` (``hand`` | ``watchdog``)
   and ``evidence`` (the error line, or the operator's note).

3. FAIL SAFE ON THE RECORD ITSELF. :func:`load` never raises. A missing or
   unparseable record blocks NOTHING and says why in ``problem``; the engine
   turns that into one loud line per tick. A broken file must not be able to
   stop the rotation loop, and it must not be able to empty the candidate
   list either.

Blocking is a ranking filter only. It never disables or enables a seat — pool
membership is not this module's to change — and a blocked seat stays a valid
explicit ``cswap switch <num|email>`` target.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.locking import FileLock
from claude_swap.settings import atomic_write_json

RECORD_FILENAME = "blocked-orgs.json"
RECORD_SCHEMA_VERSION = 1
LOCK_FILENAME = ".blocked-orgs.lock"

WRITER_HAND = "hand"
WRITER_WATCHDOG = "watchdog"
WRITERS = (WRITER_HAND, WRITER_WATCHDOG)

# What ``cswap list`` prints in the brackets for a seat with no organization.
# It is a display placeholder, not an organization: the seats that show it
# belong to unrelated personal accounts, so it can never be blocked as a unit.
PERSONAL_TAG = "personal"

PROBLEM_MISSING = "missing"

_logger = logging.getLogger("claude-swap")


class BlockedOrgsError(ClaudeSwitchError):
    """A block/unblock request that was refused (bad label, wrong writer)."""


@dataclass(frozen=True)
class BlockedOrg:
    label: str
    written_at: float
    written_by: str
    evidence: str

    def to_json(self) -> dict:
        return {
            "written_at": self.written_at,
            "written_by": self.written_by,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class BlockedOrgsRecord:
    """The record as read. ``problem`` is "" when the file was read cleanly;
    otherwise it says why nothing is blocked (``missing``, ``unreadable:
    ...``, ``unparseable: ...``) and ``orgs`` is empty."""

    orgs: dict[str, BlockedOrg] = field(default_factory=dict)
    problem: str = ""

    def is_blocked(self, label: str) -> bool:
        """Exact, case-sensitive match on a non-empty organization label."""
        return bool(label) and label in self.orgs

    def to_json(self) -> dict:
        return {
            "schemaVersion": RECORD_SCHEMA_VERSION,
            "orgs": {label: org.to_json() for label, org in sorted(self.orgs.items())},
        }


def record_path(backup_dir: Path) -> Path:
    return backup_dir / RECORD_FILENAME


def _coerce_entry(label: str, raw: dict) -> BlockedOrg:
    """An entry blocks because its KEY is present. The three fields describe
    it and are never consulted for the decision, so a missing or mistyped one
    degrades the description, not the block. Any other key — a hand-added
    ``until``, ``ttl``, ``expires_at`` — is dropped on the floor: see
    property 2 in the module docstring."""
    written_at = raw.get("written_at")
    if isinstance(written_at, bool) or not isinstance(written_at, (int, float)):
        written_at = 0.0
    written_by = raw.get("written_by")
    if not isinstance(written_by, str) or not written_by:
        written_by = "unknown"
    evidence = raw.get("evidence")
    if not isinstance(evidence, str):
        evidence = ""
    return BlockedOrg(label, float(written_at), written_by, evidence)


def load(path: Path) -> BlockedOrgsRecord:
    """Read the record. NEVER raises; see property 3 in the module docstring.

    All-or-nothing on shape: a record whose ``orgs`` table is malformed
    anywhere blocks nothing at all rather than the entries that happened to
    parse. Half a record is a guess about what the writer meant, and the
    caller's loud line is only loud if it is not competing with a block that
    appears to be working.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return BlockedOrgsRecord(problem=PROBLEM_MISSING)
    except (OSError, UnicodeDecodeError) as e:
        return BlockedOrgsRecord(problem=f"unreadable: {type(e).__name__}: {e}")
    try:
        raw = json.loads(text)
    except ValueError as e:  # JSONDecodeError is a ValueError
        return BlockedOrgsRecord(problem=f"unparseable: {e}")
    except RecursionError:
        return BlockedOrgsRecord(problem="unparseable: nested too deeply")
    if not isinstance(raw, dict):
        return BlockedOrgsRecord(problem="unparseable: top level is not an object")
    table = raw.get("orgs")
    if not isinstance(table, dict):
        return BlockedOrgsRecord(problem="unparseable: no 'orgs' object")
    orgs: dict[str, BlockedOrg] = {}
    for label, entry in table.items():
        if not isinstance(label, str) or not label.strip():
            return BlockedOrgsRecord(problem="unparseable: empty org label")
        if not isinstance(entry, dict):
            return BlockedOrgsRecord(
                problem=f"unparseable: entry for {label!r} is not an object"
            )
        orgs[label] = _coerce_entry(label, entry)
    return BlockedOrgsRecord(orgs=orgs)


def _lock(backup_dir: Path) -> FileLock:
    return FileLock(backup_dir / LOCK_FILENAME)


def _load_for_write(path: Path) -> BlockedOrgsRecord:
    """The record to modify. A missing file starts empty. An unreadable or
    unparseable one is moved aside — never overwritten, never deleted — so a
    writer (the watchdog, at 05:19 with nobody awake) can still record a
    block, and whatever the broken file held survives for a human to read."""
    record = load(path)
    if not record.problem or record.problem == PROBLEM_MISSING:
        return BlockedOrgsRecord(orgs=dict(record.orgs))
    aside = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
    try:
        os.replace(path, aside)
    except OSError as e:
        raise BlockedOrgsError(
            f"{path} is {record.problem} and could not be moved aside ({e}); "
            "fix or move it by hand, then retry"
        ) from e
    _logger.warning(
        "blocked-orgs record was %s; moved aside to %s and starting empty",
        record.problem,
        aside,
    )
    return BlockedOrgsRecord()


def _check_writer(by: str) -> None:
    if by not in WRITERS:
        raise BlockedOrgsError(
            f"written_by must be one of {', '.join(WRITERS)} (got {by!r})"
        )


def init(backup_dir: Path) -> bool:
    """Create an empty record if none exists; True when one was written.

    A record that is MISSING is reported loudly every tick, because on a box
    that has ever blocked an org a vanished file means a block silently
    stopped applying. Running this once at install time is what makes that
    line mean something from then on."""
    path = record_path(backup_dir)
    with _lock(backup_dir):
        if load(path).problem != PROBLEM_MISSING:
            return False
        atomic_write_json(path, BlockedOrgsRecord().to_json())
    _logger.info("blocked-orgs record initialised (empty) at %s", path)
    return True


def block(
    backup_dir: Path,
    label: str,
    *,
    by: str,
    evidence: str,
    known_labels: set[str] | None = None,
    force: bool = False,
    now: float | None = None,
) -> BlockedOrg:
    """Record ``label`` as blocked; returns the entry now in force.

    ``known_labels`` is the set of labels the roster currently shows; with it
    (and without ``force``) an unknown label is refused, naming a
    case-insensitive near miss when there is one.

    A ``watchdog`` write never takes over an entry a human wrote: the hand
    entry stays as it is and is returned, so the watchdog's later RESOLVED
    cannot clear a block an operator placed for reasons of their own. A
    ``hand`` write replaces either kind.
    """
    _check_writer(by)
    label = label.strip()
    evidence = evidence.strip()
    if not label:
        raise BlockedOrgsError("an organization label is required")
    if label.lower() == PERSONAL_TAG:
        raise BlockedOrgsError(
            f"'{PERSONAL_TAG}' is what cswap list shows for seats with no "
            "organization — it is not one organization and cannot be blocked"
        )
    if not evidence:
        raise BlockedOrgsError(
            "--evidence is required: the error line, or a note saying why"
        )
    if known_labels is not None and label not in known_labels and not force:
        near = sorted(k for k in known_labels if k.lower() == label.lower())
        hint = f" — did you mean {near[0]!r}?" if near else ""
        known = ", ".join(sorted(known_labels)) or "(none)"
        raise BlockedOrgsError(
            f"no managed account shows the org label {label!r}{hint} "
            f"Labels match exactly, as cswap list prints them: {known}. "
            "Use --force to record it anyway."
        )
    path = record_path(backup_dir)
    with _lock(backup_dir):
        record = _load_for_write(path)
        existing = record.orgs.get(label)
        if existing is not None and existing.written_by == WRITER_HAND and by != WRITER_HAND:
            return existing
        entry = BlockedOrg(
            label, float(time.time() if now is None else now), by, evidence
        )
        orgs = dict(record.orgs)
        orgs[label] = entry
        atomic_write_json(path, BlockedOrgsRecord(orgs=orgs).to_json())
    _logger.warning("org blocked: %s (by %s) evidence: %s", label, by, evidence)
    return entry


def unblock(
    backup_dir: Path,
    label: str,
    *,
    by: str,
    evidence: str = "",
) -> BlockedOrg | None:
    """Clear ``label``; returns the entry removed, or None if it was not there.

    The ONLY way an entry stops applying. The watchdog must say what resolved
    it, and may clear only what it wrote itself. The file is kept — an empty
    record, never an absent one — so "missing" stays an anomaly.
    """
    _check_writer(by)
    label = label.strip()
    evidence = evidence.strip()
    if not label:
        raise BlockedOrgsError("an organization label is required")
    if by == WRITER_WATCHDOG and not evidence:
        raise BlockedOrgsError(
            "the watchdog clears a block only with its RESOLVED evidence "
            "(--evidence)"
        )
    path = record_path(backup_dir)
    with _lock(backup_dir):
        record = _load_for_write(path)
        existing = record.orgs.get(label)
        if existing is None:
            return None
        if by == WRITER_WATCHDOG and existing.written_by != WRITER_WATCHDOG:
            raise BlockedOrgsError(
                f"{label!r} was blocked by {existing.written_by}; the watchdog "
                "clears only its own entries — unblock it by hand"
            )
        orgs = {k: v for k, v in record.orgs.items() if k != label}
        atomic_write_json(path, BlockedOrgsRecord(orgs=orgs).to_json())
    _logger.warning(
        "org unblocked: %s (by %s)%s",
        label,
        by,
        f" evidence: {evidence}" if evidence else "",
    )
    return existing
