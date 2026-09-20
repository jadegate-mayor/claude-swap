"""The blocked-orgs record and the auto-switch filter built on it.

Measured origin (2026-09-20 05:19-07:03 CST): the active pool seat's
organization disabled Claude Code access. Every turn died with an API error
while the seat's usage windows read 13%, so an engine that ranks on usage
windows neither noticed nor moved for 1 h 44 min.

Five behaviours, each pinned below:

1. the ranking excludes a blocked org's seats, keyed on the ORG LABEL that
   ``cswap list`` prints in brackets — never the email domain;
2. an entry never expires into eligibility on a timer;
3. an ACTIVE seat whose org becomes blocked fails over at once;
4. a missing or unparseable record excludes nothing, says so once per tick,
   and never crashes the loop;
5. when every otherwise-eligible seat is blocked the engine stays put and
   raises the alarm instead of thrashing.

Nothing here disables or enables a seat: a block is a ranking filter.
"""

from __future__ import annotations

import json
import logging
import sys
from unittest.mock import patch

import pytest

from claude_swap import blocked_orgs, cli
from claude_swap.autoswitch import (
    MODEL_WINDOW_MISSING,
    ORG_BLOCKED,
    AllEligibleSeatsBlockedEvent,
    AllExhaustedEvent,
    BlockedOrgsRecordEvent,
    NoSwitchEvent,
    OrgBlockedSkipEvent,
    PollEvent,
    SwitchEvent,
    TickOutcome,
)
from claude_swap.blocked_orgs import BlockedOrgsError
from claude_swap.switcher import ClaudeAccountSwitcher
from tests.test_autoswitch import EngineHarness, _iso_at
from tests.test_autoswitch_model_window import _seat

H = 3600.0
DAY = 24 * H

# Deliberately crossed: the seat on the acme.example DOMAIN belongs to another
# org, and the Acme ORG's seats sit on unrelated domains. A filter keyed on
# the domain gets every one of these wrong.
SEATS = {
    1: ("ops@initech.example", "Initech"),
    2: ("someone@acme.example", "Initech"),
    3: ("contractor@gmail.example", "Acme"),
    4: ("other@umbrella.example", "Acme"),
    5: ("solo@personal.example", ""),
}


def _harness(temp_home, seats=(1, 2, 3, 4), *, init=True, **settings) -> EngineHarness:
    (temp_home / ".claude").mkdir(parents=True, exist_ok=True)
    h = EngineHarness(temp_home, **settings)
    for n in seats:
        email, org = SEATS[n]
        h.seed(n, email)
        data = h.switcher._get_sequence_data()
        data["accounts"][str(n)]["organizationName"] = org
        data["accounts"][str(n)]["organizationUuid"] = f"org-{org}" if org else ""
        h.switcher._write_json(h.switcher.sequence_file, data)
    make_live(h, seats[0])
    if init:
        blocked_orgs.init(h.switcher.backup_dir)
    return h


def make_live(h: EngineHarness, num: int) -> None:
    """Point the live login at seat ``num`` (org-aware, unlike the base
    harness's, whose seats have no organization)."""
    email, org = SEATS[num]
    (h.temp_home / ".claude" / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {"accessToken": "sk-live", "refreshToken": "rt-live"},
    }))
    (h.temp_home / ".claude.json").write_text(json.dumps({
        "oauthAccount": {
            "emailAddress": email,
            "accountUuid": f"uuid-{num}",
            "organizationUuid": f"org-{org}" if org else "",
            "organizationName": org,
        },
    }))


def _block(h: EngineHarness, label: str = "Acme", **kw) -> None:
    kw.setdefault("by", "hand")
    kw.setdefault("evidence", "API Error: organization has disabled Claude Code")
    blocked_orgs.block(h.switcher.backup_dir, label, **kw)


def _of(h: EngineHarness, kind) -> list:
    return [e for e in h.events if isinstance(e, kind)]


def _path(h: EngineHarness):
    return blocked_orgs.record_path(h.switcher.backup_dir)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_an_entry_carries_when_who_and_why_keyed_on_the_label(self, tmp_path):
        entry = blocked_orgs.block(
            tmp_path, "Acme", by="hand", evidence="org disabled", now=1_789_861_558
        )
        assert (entry.written_at, entry.written_by, entry.evidence) == (
            1_789_861_558.0, "hand", "org disabled",
        )
        on_disk = json.loads(blocked_orgs.record_path(tmp_path).read_text())
        assert on_disk == {
            "schemaVersion": 1,
            "orgs": {
                "Acme": {
                    "written_at": 1_789_861_558.0,
                    "written_by": "hand",
                    "evidence": "org disabled",
                }
            },
        }
        record = blocked_orgs.load(blocked_orgs.record_path(tmp_path))
        assert record.problem == ""
        assert record.is_blocked("Acme")

    def test_matching_is_the_exact_label_never_a_domain_or_a_near_miss(self, tmp_path):
        blocked_orgs.block(tmp_path, "Acme", by="hand", evidence="x")
        record = blocked_orgs.load(blocked_orgs.record_path(tmp_path))
        for not_it in ("acme", "acme.example", "Acme ", "Acm", ""):
            assert not record.is_blocked(not_it), not_it

    def test_a_missing_record_blocks_nothing_and_says_so(self, tmp_path):
        record = blocked_orgs.load(tmp_path / "blocked-orgs.json")
        assert record.orgs == {}
        assert record.problem == blocked_orgs.PROBLEM_MISSING
        assert not record.is_blocked("Acme")

    @pytest.mark.parametrize(
        "text",
        [
            "{not json",
            "",
            "[]",
            '"Acme"',
            "{}",
            '{"orgs": ["Acme"]}',
            '{"orgs": {"Acme": "blocked"}}',
            '{"orgs": {"": {}}}',
            '{"orgs": {"Acme": {}, "Other": 7}}',
            "[" * 100_000,
        ],
    )
    def test_an_unparseable_record_blocks_nothing_and_never_raises(self, tmp_path, text):
        path = tmp_path / "blocked-orgs.json"
        path.write_text(text)
        record = blocked_orgs.load(path)
        # All-or-nothing: not even the entries that happened to parse.
        assert record.orgs == {}
        assert record.problem.startswith("unparseable")

    def test_an_unreadable_record_blocks_nothing_and_never_raises(self, tmp_path):
        path = tmp_path / "blocked-orgs.json"
        path.write_bytes(b"\xff\xfe\x00 not utf-8 \xff")
        record = blocked_orgs.load(path)
        assert record.orgs == {}
        assert record.problem.startswith("unreadable")
        path.unlink()
        path.mkdir()  # a directory where the file should be
        assert blocked_orgs.load(path).problem.startswith("unreadable")

    def test_no_field_in_an_entry_can_expire_it(self, tmp_path):
        """CONDITION 2. There is no TTL and no ``until``; a hand-added one is
        ignored, and an ancient ``written_at`` is only a description."""
        path = tmp_path / "blocked-orgs.json"
        path.write_text(json.dumps({
            "schemaVersion": 1,
            "orgs": {
                "Acme": {
                    "written_at": 1.0,           # 1970
                    "written_by": "hand",
                    "evidence": "x",
                    "until": 2.0,                # long past
                    "ttl": 0,
                    "expires_at": "1970-01-01T00:00:03Z",
                    "ttl_seconds": 1,
                },
            },
        }))
        record = blocked_orgs.load(path)
        assert record.problem == ""
        assert record.is_blocked("Acme")

    def test_the_module_never_reads_a_clock_to_decide(self):
        """Structural half of CONDITION 2: ``load`` and ``is_blocked`` — the
        only two things a decision goes through — must not touch time."""
        clocks = {"time", "datetime", "monotonic", "clock", "now", "today"}
        for fn in (blocked_orgs.load, blocked_orgs._parse,
                   blocked_orgs.BlockedOrgsRecord.is_blocked,
                   blocked_orgs._coerce_entry):
            assert clocks.isdisjoint(fn.__code__.co_names), fn.__name__
        # The guard can see a clock when there is one (it is not vacuous).
        assert not clocks.isdisjoint(blocked_orgs.block.__code__.co_names)

    def test_an_entry_with_damaged_fields_still_blocks(self, tmp_path):
        path = tmp_path / "blocked-orgs.json"
        path.write_text('{"orgs": {"Acme": {"written_at": "tuesday", "written_by": 3}}}')
        record = blocked_orgs.load(path)
        assert record.is_blocked("Acme")
        org = record.orgs["Acme"]
        assert (org.written_at, org.written_by, org.evidence) == (0.0, "unknown", "")

    @pytest.mark.parametrize(
        "written_at",
        ["1" + "0" * 400, "1e999", "-1e999", "NaN", "1e300", "true"],
        ids=["int-too-big-for-float", "inf", "-inf", "nan", "finite-unrenderable", "bool"],
    )
    def test_an_unrepresentable_timestamp_cannot_break_the_record(
        self, tmp_path, written_at, capsys
    ):
        """Review round 1: an integer no float can hold raised OverflowError
        OUT of load() — crashing list/status/block, and (through the engine's
        belt) silently dropping every block. `1e999` loaded as inf and crashed
        the human listing. A description must never break the record."""
        path = tmp_path / "blocked-orgs.json"
        path.write_text(
            '{"orgs": {"Acme": {"written_at": %s, "written_by": "hand"}}}' % written_at
        )
        record = blocked_orgs.load(path)
        assert record.problem == ""
        assert record.is_blocked("Acme")            # the block survives
        expected = 1e300 if written_at == "1e300" else 0.0
        assert record.orgs["Acme"].written_at == expected
        json.dumps(record.to_json(), allow_nan=False)      # still portable JSON
        assert cli._time_label(record.orgs["Acme"].written_at)  # and printable

    def test_load_keeps_its_promise_even_if_parsing_itself_breaks(self, tmp_path):
        path = tmp_path / "blocked-orgs.json"
        path.write_text('{"orgs": {"Acme": {}}}')
        with patch.object(blocked_orgs, "_coerce_entry", side_effect=RuntimeError("x")):
            record = blocked_orgs.load(path)
        assert record.orgs == {} and record.problem.startswith("unparseable")

    def test_unblock_is_the_only_exit_and_keeps_the_file(self, tmp_path):
        blocked_orgs.block(tmp_path, "Acme", by="hand", evidence="x")
        removed = blocked_orgs.unblock(tmp_path, "Acme", by="hand")
        assert removed is not None and removed.label == "Acme"
        path = blocked_orgs.record_path(tmp_path)
        assert path.exists()  # empty, never absent: "missing" stays an anomaly
        record = blocked_orgs.load(path)
        assert record.problem == "" and record.orgs == {}
        assert blocked_orgs.unblock(tmp_path, "Acme", by="hand") is None

    def test_the_watchdog_clears_only_its_own_entries_and_only_with_evidence(self, tmp_path):
        blocked_orgs.block(tmp_path, "ByHand", by="hand", evidence="operator's reasons")
        blocked_orgs.block(tmp_path, "ByDog", by="watchdog", evidence="ORG-DISABLED verdict")
        with pytest.raises(BlockedOrgsError, match="RESOLVED evidence"):
            blocked_orgs.unblock(tmp_path, "ByDog", by="watchdog")
        with pytest.raises(BlockedOrgsError, match="clears only its own"):
            blocked_orgs.unblock(tmp_path, "ByHand", by="watchdog", evidence="RESOLVED")
        assert blocked_orgs.unblock(
            tmp_path, "ByDog", by="watchdog", evidence="RESOLVED 07:03 turn ok"
        ) is not None
        record = blocked_orgs.load(blocked_orgs.record_path(tmp_path))
        assert set(record.orgs) == {"ByHand"}
        # A human may clear either kind.
        blocked_orgs.block(tmp_path, "ByDog", by="watchdog", evidence="again")
        assert blocked_orgs.unblock(tmp_path, "ByDog", by="hand") is not None

    def test_a_watchdog_write_never_takes_over_a_hand_entry(self, tmp_path):
        blocked_orgs.block(tmp_path, "Acme", by="hand", evidence="mine", now=10)
        kept = blocked_orgs.block(tmp_path, "Acme", by="watchdog", evidence="dog", now=20)
        assert (kept.written_by, kept.evidence, kept.written_at) == ("hand", "mine", 10.0)
        # ...while a human takes over a watchdog's.
        blocked_orgs.block(tmp_path, "Other", by="watchdog", evidence="dog", now=30)
        taken = blocked_orgs.block(tmp_path, "Other", by="hand", evidence="mine", now=40)
        assert (taken.written_by, taken.written_at) == ("hand", 40.0)

    def test_refusals(self, tmp_path):
        with pytest.raises(BlockedOrgsError, match="--evidence is required"):
            blocked_orgs.block(tmp_path, "Acme", by="hand", evidence="  ")
        with pytest.raises(BlockedOrgsError, match="not one organization"):
            blocked_orgs.block(tmp_path, "personal", by="hand", evidence="x")
        with pytest.raises(BlockedOrgsError, match="written_by must be"):
            blocked_orgs.block(tmp_path, "Acme", by="cron", evidence="x")
        with pytest.raises(BlockedOrgsError, match="label is required"):
            blocked_orgs.block(tmp_path, "  ", by="hand", evidence="x")
        assert not blocked_orgs.record_path(tmp_path).exists()

    def test_a_label_no_account_shows_is_refused_with_the_near_miss(self, tmp_path):
        known = {"Acme", "Initech"}
        with pytest.raises(BlockedOrgsError, match="did you mean 'Acme'"):
            blocked_orgs.block(
                tmp_path, "acme", by="hand", evidence="x", known_labels=known
            )
        with pytest.raises(BlockedOrgsError, match="no managed account shows"):
            blocked_orgs.block(
                tmp_path, "acme.example", by="hand", evidence="x", known_labels=known
            )
        entry = blocked_orgs.block(
            tmp_path, "NotYetAdded", by="hand", evidence="x",
            known_labels=known, force=True,
        )
        assert entry.label == "NotYetAdded"

    def test_writing_over_a_corrupt_record_moves_it_aside_intact(self, tmp_path):
        path = blocked_orgs.record_path(tmp_path)
        path.write_text("{corrupt")
        blocked_orgs.block(tmp_path, "Acme", by="watchdog", evidence="x")
        assert blocked_orgs.load(path).is_blocked("Acme")
        asides = list(tmp_path.glob("blocked-orgs.json.corrupt-*"))
        assert len(asides) == 1 and asides[0].read_text() == "{corrupt"

    def test_init_writes_an_empty_record_once(self, tmp_path):
        assert blocked_orgs.init(tmp_path) is True
        assert blocked_orgs.load(blocked_orgs.record_path(tmp_path)).problem == ""
        blocked_orgs.block(tmp_path, "Acme", by="hand", evidence="x")
        assert blocked_orgs.init(tmp_path) is False  # never clobbers
        assert blocked_orgs.load(blocked_orgs.record_path(tmp_path)).is_blocked("Acme")


# ---------------------------------------------------------------------------
# The ranking filter
# ---------------------------------------------------------------------------


class TestRankingExcludesABlockedOrg:
    def test_keyed_on_the_bracket_label_not_the_email_domain(self, temp_home):
        h = _harness(temp_home)
        _block(h, "Acme")
        outcome = h.tick_with_usage({
            "1": _seat(five_h=95),
            "2": _seat(five_h=60),   # on the acme.example DOMAIN, Initech org
            "3": _seat(five_h=1),    # Acme org on another domain: most room
            "4": _seat(five_h=2),    # Acme org on another domain
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        poll = _of(h, PollEvent)[0]
        assert poll.skipped == {"3": ORG_BLOCKED, "4": ORG_BLOCKED}
        assert poll.to_json()["skippedCandidates"] == {"3": ORG_BLOCKED, "4": ORG_BLOCKED}
        assert "#3: 5h 1% · 7d 10% (skipped: org-blocked)" in poll.human()
        assert "skipped" not in poll.human().split("#2:")[1].split("#3:")[0]

    def test_the_skip_line_is_logged_every_tick(self, temp_home):
        h = _harness(temp_home)
        _block(h)
        usage = {"1": _seat(), "2": _seat(), "3": _seat(), "4": _seat()}
        for _ in range(3):
            h.tick_with_usage(usage)
        skips = _of(h, OrgBlockedSkipEvent)
        assert len(skips) == 3
        assert skips[0].human() == "skip: org blocked Acme (#3, #4)"
        assert skips[0].to_json()["organization"] == "Acme"
        assert skips[0].to_json()["accounts"] == [3, 4]

    def test_without_the_block_the_same_seat_wins(self, temp_home):
        h = _harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _seat(five_h=95), "2": _seat(five_h=60),
            "3": _seat(five_h=1), "4": _seat(five_h=2),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        assert _of(h, PollEvent)[0].skipped == {}
        assert _of(h, OrgBlockedSkipEvent) == []

    def test_at_limit_escape_still_refuses_it(self, temp_home):
        h = _harness(temp_home)
        _block(h)
        outcome = h.tick_with_usage({
            "1": _seat(five_h=100), "2": _seat(five_h=88),
            "3": _seat(five_h=0), "4": _seat(five_h=0),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert _of(h, SwitchEvent)[0].trigger == "at-limit"

    def test_failover_still_refuses_it(self, temp_home):
        h = _harness(temp_home)
        _block(h)
        usage = {"1": None, "2": _seat(five_h=70), "3": _seat(five_h=0), "4": _seat(five_h=0)}
        for _ in range(2):
            assert h.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert _of(h, SwitchEvent)[0].trigger == "failover"

    def test_consume_first_still_refuses_it(self, temp_home):
        h = _harness(temp_home, strategy="consume-first")
        _block(h)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _seat(seven_d_reset=now + 5 * DAY),
            "2": _seat(seven_d_reset=now + 3 * DAY),
            "3": _seat(seven_d_reset=now + 1 * DAY),   # soonest — and blocked
            "4": _seat(seven_d_reset=now + 2 * DAY),   # next — and blocked
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert _of(h, SwitchEvent)[0].trigger == "consume-first"

    def test_the_block_outranks_a_missing_model_window_on_the_poll_line(self, temp_home):
        h = _harness(temp_home, model="Fable")
        _block(h)
        h.tick_with_usage({
            "1": _seat(fable=20),
            "2": _seat(),            # no Fable window, not blocked
            "3": _seat(),            # no Fable window AND blocked
            "4": _seat(fable=5),     # blocked
        })
        assert _of(h, PollEvent)[0].skipped == {
            "2": MODEL_WINDOW_MISSING, "3": ORG_BLOCKED, "4": ORG_BLOCKED,
        }

    def test_a_seat_with_no_organization_is_never_blocked(self, temp_home):
        h = _harness(temp_home, seats=(1, 5))
        # Not reachable through the CLI ("personal" is refused); a hand edit
        # naming the placeholder `cswap list` shows for org-less seats.
        _path(h).write_text(json.dumps({"orgs": {"personal": {}}}))
        h.tick_with_usage({"1": _seat(five_h=95), "5": _seat(five_h=5)})
        assert h.active_number() == 5
        assert _of(h, OrgBlockedSkipEvent) == []
        assert _of(h, BlockedOrgsRecordEvent) == []

    def test_blocking_never_disables_or_enables_a_seat(self, temp_home):
        h = _harness(temp_home)
        before = json.loads(h.switcher.sequence_file.read_text())["accounts"]
        _block(h)
        h.tick_with_usage({
            "1": _seat(five_h=95), "2": _seat(five_h=60),
            "3": _seat(five_h=1), "4": _seat(five_h=2),
        })
        blocked_orgs.unblock(h.switcher.backup_dir, "Acme", by="hand")
        after = json.loads(h.switcher.sequence_file.read_text())["accounts"]
        assert all("disabled" not in acct for acct in after.values())
        assert {n: a.get("disabled") for n, a in before.items()} == {
            n: a.get("disabled") for n, a in after.items()
        }
        assert h.switcher.switchable_account_numbers() == ["1", "2", "3", "4"]

    def test_unblocking_readmits_the_seat_on_the_next_tick_without_a_restart(self, temp_home):
        h = _harness(temp_home, cooldown_seconds=0)
        _block(h)
        usage = {
            "1": _seat(five_h=95), "2": _seat(five_h=92),
            "3": _seat(five_h=1), "4": _seat(five_h=50),
        }
        # Nothing unblocked is healthy enough to land on proactively.
        assert h.tick_with_usage(usage) is TickOutcome.BLOCKED
        assert h.active_number() == 1
        blocked_orgs.unblock(h.switcher.backup_dir, "Acme", by="hand")
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 3


class TestNoTimerExpiry:
    """CONDITION 2 at the engine: time passing re-admits nothing."""

    def test_a_year_later_the_seat_is_still_skipped(self, temp_home):
        h = _harness(temp_home)
        _block(h, now=h.clock.now)
        usage = {
            "1": _seat(five_h=95), "2": _seat(five_h=60),
            "3": _seat(five_h=1), "4": _seat(five_h=2),
        }
        h.clock.now += 400 * DAY
        with patch("time.time", return_value=h.clock.now):
            assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert _of(h, PollEvent)[0].skipped == {"3": ORG_BLOCKED, "4": ORG_BLOCKED}

    def test_a_hand_added_until_in_the_past_is_ignored(self, temp_home):
        h = _harness(temp_home)
        _path(h).write_text(json.dumps({
            "schemaVersion": 1,
            "orgs": {"Acme": {
                "written_at": 1.0, "written_by": "hand", "evidence": "x",
                "until": 2.0, "ttl": 1, "expires_at": _iso_at(3.0),
            }},
        }))
        h.tick_with_usage({
            "1": _seat(five_h=95), "2": _seat(five_h=60),
            "3": _seat(five_h=1), "4": _seat(five_h=2),
        })
        assert h.active_number() == 2


class TestActiveSeatBecomesBlocked:
    """「an active seat that becomes blocked triggers a failover the way
    active-usage-unknown 3/3 does」 — and at once: the record is a verdict."""

    def _on_acme(self, temp_home, **settings) -> EngineHarness:
        h = _harness(temp_home, seats=(3, 1, 2, 4), **settings)
        assert h.switcher.current_account_number() == "3"
        return h

    def test_fails_over_on_the_first_tick_after_the_block(self, temp_home):
        h = self._on_acme(temp_home)
        # The 9/20 shape: the dead seat reads 13% — the numbers show nothing.
        usage = {
            "3": _seat(five_h=13), "1": _seat(five_h=40),
            "2": _seat(five_h=50), "4": _seat(five_h=0),
        }
        assert h.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert h.active_number() == 3
        _block(h)
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 1            # most room among the unblocked
        switch = _of(h, SwitchEvent)[0]
        assert switch.trigger == "failover"
        skip = _of(h, OrgBlockedSkipEvent)[-1]
        assert skip.active is True
        assert skip.human() == (
            "skip: org blocked Acme (#3, #4) — includes the ACTIVE seat, "
            "failing over"
        )

    def test_the_failover_ignores_the_cooldown(self, temp_home):
        h = _harness(temp_home, seats=(1, 2, 3, 4), cooldown_seconds=3600)
        usage = {
            "1": _seat(five_h=95), "2": _seat(five_h=80),
            "3": _seat(five_h=10), "4": _seat(five_h=20),
        }
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED   # 1 -> 3
        assert h.active_number() == 3
        make_live(h, 3)
        _block(h)                                                 # seconds later
        usage["1"] = _seat(five_h=30)
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 1                             # not sibling #4
        assert _of(h, SwitchEvent)[-1].trigger == "failover"

    def test_once_unblocked_the_seat_it_left_is_a_landing_spot_again(self, temp_home):
        h = self._on_acme(temp_home, cooldown_seconds=0)
        _block(h)
        usage = {
            "3": _seat(five_h=13), "1": _seat(five_h=40),
            "2": _seat(five_h=50), "4": _seat(five_h=0),
        }
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED    # 3 -> 1
        make_live(h, 1)
        usage["1"] = _seat(five_h=96)
        usage["2"] = _seat(five_h=97)
        # Still blocked: the no-return bar is not what holds it out.
        assert h.tick_with_usage(usage) is TickOutcome.BLOCKED
        blocked_orgs.unblock(h.switcher.backup_dir, "Acme", by="hand")
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 4                              # 0% used

    def test_it_refetches_the_possible_targets_before_choosing_where_to_land(self, temp_home):
        """Review round 1 (P1). The failover fires with the active's windows
        anywhere (13% on the day), so the collector never escalated and the
        peers' rows can be as old as the candidate cadence allows. A peer
        that read as room then may be at its wall now."""
        from tests.test_autoswitch import _entry_for

        h = self._on_acme(temp_home)
        _block(h)
        stale = {
            "3": _seat(five_h=13), "1": _seat(five_h=10),   # looked best...
            "2": _seat(five_h=50), "4": _seat(five_h=0),
        }
        fresh = {**stale, "1": _seat(five_h=100)}           # ...is at its wall
        calls: list = []

        def entries(fetch=None, **_kw):
            calls.append(fetch)
            rows = fresh if fetch else stale
            return {n: _entry_for(v, h.clock.now) for n, v in rows.items()}

        with patch.object(h.switcher, "usage_entries_by_account", side_effect=entries):
            assert h.engine.tick() is TickOutcome.SWITCHED
        assert h.active_number() == 2                        # not the spent #1
        # Only seats that could be targets: never the blocked sibling, and
        # never the active row (it is the evidence that fired).
        assert {"1", "2"} in calls

    def test_the_refetch_honours_a_post_429_exhausted_plan(self, temp_home):
        from dataclasses import replace

        from claude_swap import poll_policy
        from tests.test_autoswitch import _entry_for

        h = self._on_acme(temp_home)
        _block(h)
        now = h.clock.now
        rows = {
            "3": _entry_for(_seat(five_h=13), now),
            "1": replace(
                _entry_for(_seat(five_h=100), now),
                next_poll_at=now + 10 * poll_policy.EXHAUSTED_INTERVAL_S,
                poll_interval_s=10 * poll_policy.EXHAUSTED_INTERVAL_S,
            ),
            "2": _entry_for(_seat(five_h=50), now),
            "4": _entry_for(_seat(five_h=0), now),
        }
        with patch.object(
            h.switcher, "usage_entries_by_account", return_value=rows
        ) as fetch:
            assert h.engine.tick() is TickOutcome.SWITCHED
        fetched = [c.kwargs.get("fetch") for c in fetch.call_args_list]
        assert {"2"} in fetched and {"1", "2"} not in fetched

    def test_it_never_lands_on_a_sibling_seat_of_the_same_org(self, temp_home):
        h = self._on_acme(temp_home)
        _block(h)
        h.tick_with_usage({
            "3": _seat(five_h=13), "1": _seat(five_h=89),
            "2": _seat(five_h=89.5), "4": _seat(five_h=0),   # sibling: pristine
        })
        assert h.active_number() == 1


class TestRecordFailsSafe:
    """CONDITION 3, first half."""

    USAGE = {
        "1": _seat(five_h=95), "2": _seat(five_h=60),
        "3": _seat(five_h=1), "4": _seat(five_h=2),
    }

    def _assert_nothing_excluded_and_loud(self, h, caplog, problem_prefix):
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            outcome = h.tick_with_usage(self.USAGE)
        assert outcome is TickOutcome.SWITCHED       # the loop carried on
        assert h.active_number() == 3                # and NOTHING was excluded
        assert _of(h, PollEvent)[0].skipped == {}
        loud = _of(h, BlockedOrgsRecordEvent)
        assert len(loud) == 1                        # ONE line per tick
        assert loud[0].problem.startswith(problem_prefix)
        assert "NO org is being excluded" in loud[0].human()
        assert loud[0].human().startswith("WARNING:")
        logged = [r for r in caplog.records if "NO org is excluded" in r.getMessage()]
        assert len(logged) == 1 and logged[0].levelno == logging.WARNING

    def test_missing_record(self, temp_home, caplog):
        h = _harness(temp_home, init=False)
        assert not _path(h).exists()
        self._assert_nothing_excluded_and_loud(h, caplog, "missing")
        assert "cswap blocked-orgs --init" in _of(h, BlockedOrgsRecordEvent)[0].human()

    def test_a_record_that_vanishes_stops_excluding_and_says_so(self, temp_home, caplog):
        h = _harness(temp_home)
        _block(h)
        _path(h).unlink()
        self._assert_nothing_excluded_and_loud(h, caplog, "missing")

    @pytest.mark.parametrize("text", ["{truncated", "[]", '{"orgs": {"Acme": 1}}'])
    def test_unparseable_record(self, temp_home, caplog, text):
        h = _harness(temp_home)
        _path(h).write_text(text)
        self._assert_nothing_excluded_and_loud(h, caplog, "unparseable")

    def test_it_is_loud_every_tick_not_once(self, temp_home):
        h = _harness(temp_home, init=False)
        quiet = {"1": _seat(), "2": _seat(), "3": _seat(), "4": _seat()}
        for _ in range(4):
            assert h.tick_with_usage(quiet) is TickOutcome.NO_ACTION
        assert len(_of(h, BlockedOrgsRecordEvent)) == 4

    def test_still_one_line_when_the_tick_ranks_twice(self, temp_home):
        # consume-first re-ranks on a refetch inside one tick.
        h = _harness(temp_home, init=False, strategy="consume-first")
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _seat(seven_d_reset=now + 5 * DAY),
            "2": _seat(seven_d_reset=now + 1 * DAY),
            "3": _seat(seven_d_reset=now + 6 * DAY),
            "4": _seat(seven_d_reset=now + 7 * DAY),
        })
        assert outcome is TickOutcome.SWITCHED
        assert len(_of(h, BlockedOrgsRecordEvent)) == 1

    def test_a_clean_record_is_silent(self, temp_home):
        h = _harness(temp_home)
        h.tick_with_usage(self.USAGE)
        assert _of(h, BlockedOrgsRecordEvent) == []

    def test_even_a_loader_that_raises_cannot_stop_the_tick(self, temp_home):
        h = _harness(temp_home)
        with patch.object(blocked_orgs, "load", side_effect=RuntimeError("boom")):
            assert h.tick_with_usage(self.USAGE) is TickOutcome.SWITCHED
        assert _of(h, BlockedOrgsRecordEvent)[0].problem.startswith("unreadable")

    def test_with_no_active_account_it_is_still_reported(self, temp_home):
        h = _harness(temp_home, init=False)
        (temp_home / ".claude.json").unlink()
        (temp_home / ".claude" / ".credentials.json").unlink()
        assert h.tick_with_usage({}) is TickOutcome.NO_ACTION
        assert len(_of(h, BlockedOrgsRecordEvent)) == 1


class TestAllEligibleSeatsBlocked:
    """CONDITION 3, second half: stay put and alarm — never thrash."""

    def test_a_trigger_that_needs_a_target_stays_put_and_alarms(self, temp_home, caplog):
        h = _harness(temp_home, seats=(1, 3, 4))
        _block(h)
        usage = {"1": _seat(five_h=95), "3": _seat(five_h=1), "4": _seat(five_h=2)}
        with caplog.at_level(logging.ERROR, logger="claude-swap"):
            for _ in range(5):
                assert h.tick_with_usage(usage) is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert _of(h, SwitchEvent) == []
        alarms = _of(h, AllEligibleSeatsBlockedEvent)
        assert len(alarms) == 5                              # every tick
        assert alarms[0].human() == (
            "ALARM: all eligible seats blocked (Acme: #3, #4) — staying on "
            "Account-1; unblock an org or add a seat"
        )
        assert alarms[0].to_json()["stayingOn"] == 1
        assert not _of(h, AllExhaustedEvent)
        assert h.engine._blocked_wait_long is False          # normal cadence
        assert sum(
            "all eligible seats blocked" in r.getMessage() for r in caplog.records
        ) == 5

    def test_at_the_wall_it_still_does_not_land_on_a_blocked_seat(self, temp_home):
        h = _harness(temp_home, seats=(1, 3, 4))
        _block(h)
        outcome = h.tick_with_usage(
            {"1": _seat(five_h=100), "3": _seat(five_h=0), "4": _seat(five_h=0)}
        )
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 1

    def test_a_blocked_active_seat_with_only_blocked_siblings_stays_put(self, temp_home):
        h = _harness(temp_home, seats=(3, 4))
        _block(h)
        usage = {"3": _seat(five_h=13), "4": _seat(five_h=0)}
        for _ in range(4):
            assert h.tick_with_usage(usage) is TickOutcome.BLOCKED
        assert h.active_number() == 3
        assert _of(h, SwitchEvent) == []
        alarm = _of(h, AllEligibleSeatsBlockedEvent)[0]
        assert alarm.active_blocked is True
        assert "staying on Account-3, whose own org is blocked" in alarm.human()

    def test_a_blocked_active_seat_with_no_other_seat_at_all(self, temp_home):
        h = _harness(temp_home, seats=(3,))
        _block(h)
        assert h.tick_with_usage({"3": _seat(five_h=13)}) is TickOutcome.BLOCKED
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 1
        assert h.engine._blocked_wait_long is False

    def test_a_healthy_consume_first_active_alarms_but_is_not_blocked(self, temp_home):
        h = _harness(temp_home, seats=(1, 3, 4), strategy="consume-first")
        _block(h)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _seat(seven_d_reset=now + 5 * DAY),
            "3": _seat(seven_d_reset=now + 1 * DAY),
            "4": _seat(seven_d_reset=now + 2 * DAY),
        })
        # Not wanting to move is not "blocked" (the --once exit-code contract),
        # but a pool one wall away from dead is worth a line every tick.
        assert outcome is TickOutcome.NO_ACTION
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 1
        assert _of(h, NoSwitchEvent)[-1].reason == "already-consuming-soonest"
        assert "org blocked (#3 Acme, #4 Acme)" in _of(h, NoSwitchEvent)[-1].detail

    def test_a_healthy_best_strategy_active_alarms_too(self, temp_home):
        """Review round 1 (P2). `best` returns below-threshold long before
        candidate selection; the alarm is about the POOL, so it is judged
        before any trigger is classified — and changes no outcome."""
        h = _harness(temp_home, seats=(1, 3, 4))            # strategy: best
        _block(h)
        usage = {"1": _seat(five_h=20), "3": _seat(five_h=1), "4": _seat(five_h=2)}
        for _ in range(3):
            assert h.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 3
        assert _of(h, NoSwitchEvent)[-1].reason == "below-threshold"

    def test_a_cooldown_return_does_not_swallow_the_alarm(self, temp_home):
        h = _harness(temp_home, seats=(2, 1, 3, 4), cooldown_seconds=3600)
        assert h.tick_with_usage({
            "2": _seat(five_h=95), "1": _seat(five_h=50),
            "3": _seat(five_h=1), "4": _seat(five_h=100),
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 3
        make_live(h, 3)
        _block(h, "Initech")                       # seats 1 and 2
        h.events.clear()
        # Wants to move (95%), but is inside the cooldown; the only unblocked
        # peer is spent and the two with room are blocked.
        assert h.tick_with_usage({
            "3": _seat(five_h=95), "4": _seat(five_h=100),
            "1": _seat(five_h=50), "2": _seat(five_h=60),
        }) is TickOutcome.NO_ACTION
        assert _of(h, NoSwitchEvent)[-1].reason == "cooldown"
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 1

    def test_no_alarm_on_a_tick_that_then_fails_over_successfully(self, temp_home):
        """Review round 2 (P2). With the active seat blocked, the unblocked
        peer may not have been read yet when the tick starts; the refetch
        reads it and the failover succeeds. An alarm before that switch would
        be a false outage report."""
        from tests.test_autoswitch import _entry_for

        h = _harness(temp_home, seats=(3, 1, 4))
        _block(h)
        first = {"3": _seat(five_h=13), "1": None, "4": _seat(five_h=0)}
        fresh = {**first, "1": _seat(five_h=20)}

        def entries(fetch=None, **_kw):
            rows = fresh if fetch else first
            return {n: _entry_for(v, h.clock.now) for n, v in rows.items()}

        with patch.object(h.switcher, "usage_entries_by_account", side_effect=entries):
            assert h.engine.tick() is TickOutcome.SWITCHED
        assert h.active_number() == 1
        assert _of(h, AllEligibleSeatsBlockedEvent) == []

    def test_an_unread_unblocked_seat_is_not_evidence_of_a_dead_pool(self, temp_home):
        h = _harness(temp_home, seats=(1, 2, 3, 4))
        _block(h)
        # Healthy active; adaptive polling has not read seat 2 this tick.
        h.tick_with_usage(
            {"1": _seat(five_h=20), "2": None, "3": _seat(five_h=1), "4": _seat(five_h=2)}
        )
        assert _of(h, AllEligibleSeatsBlockedEvent) == []
        # A seat in a KNOWN unusable state is evidence.
        from claude_swap.json_output import USAGE_RELOGIN_REQUIRED

        h.tick_with_usage({
            "1": _seat(five_h=20), "2": USAGE_RELOGIN_REQUIRED,
            "3": _seat(five_h=1), "4": _seat(five_h=2),
        })
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 1

    def test_an_enabled_api_key_fallback_means_the_pool_is_not_dead(self, temp_home):
        """Review round 2 (P2)."""
        h = _harness(temp_home, seats=(1, 3, 4), include_api_key_accounts=True)
        _block(h)
        usage = {"1": _seat(five_h=20), "3": _seat(five_h=1), "4": _seat(five_h=2)}
        real_kind = h.switcher.account_kind_for
        h.seed(2, SEATS[2][0])
        with patch.object(
            h.switcher, "account_kind_for",
            side_effect=lambda n: "api_key" if str(n) == "2" else real_kind(n),
        ):
            h.tick_with_usage(usage)
        assert _of(h, AllEligibleSeatsBlockedEvent) == []
        # (The same pool WITHOUT a fallback alarms: see
        # test_a_healthy_best_strategy_active_alarms_too.)

    def test_the_alarm_is_raised_once_per_tick_however_it_is_reached(self, temp_home):
        h = _harness(temp_home, seats=(3, 4))
        _block(h)
        h.tick_with_usage({"3": _seat(five_h=13), "4": _seat(five_h=0)})
        assert len(_of(h, AllEligibleSeatsBlockedEvent)) == 1

    def test_no_alarm_while_an_unblocked_seat_has_real_headroom(self, temp_home):
        h = _harness(temp_home)
        _block(h)
        outcome = h.tick_with_usage({
            "1": _seat(five_h=95), "2": _seat(five_h=93),   # room, but over the bar
            "3": _seat(five_h=1), "4": _seat(five_h=2),
        })
        assert outcome is TickOutcome.BLOCKED
        assert _of(h, AllEligibleSeatsBlockedEvent) == []
        stop = _of(h, NoSwitchEvent)[-1]
        assert stop.reason == "no-qualifying-candidate"
        assert "2 candidate(s) skipped: org blocked (#3 Acme, #4 Acme)" in stop.detail

    def test_no_alarm_when_the_blocked_seats_were_exhausted_anyway(self, temp_home):
        h = _harness(temp_home, seats=(1, 3, 4))
        _block(h)
        h.tick_with_usage(
            {"1": _seat(five_h=95), "3": _seat(five_h=100), "4": _seat(five_h=100)}
        )
        # The block is not what stands in the way: nothing to alarm about.
        assert _of(h, AllEligibleSeatsBlockedEvent) == []

    def test_nothing_blocked_never_alarms(self, temp_home):
        h = _harness(temp_home, seats=(1, 3, 4))
        h.tick_with_usage(
            {"1": _seat(five_h=100), "3": _seat(five_h=100), "4": _seat(five_h=100)}
        )
        assert _of(h, AllEligibleSeatsBlockedEvent) == []
        assert len(_of(h, AllExhaustedEvent)) == 1


# ---------------------------------------------------------------------------
# CLI and display
# ---------------------------------------------------------------------------


@pytest.fixture
def store(temp_home):
    h = _harness(temp_home, init=False)
    with patch("os.geteuid", return_value=1000, create=True):
        yield h


def _run(action: str, *argv: str) -> None:
    cli._blocked_orgs_command(action, list(argv))


class TestCli:
    def test_block_then_list_then_unblock(self, store, capsys):
        _run("block-org", "Acme", "--evidence", "org disabled Claude Code")
        out = capsys.readouterr().out
        assert "Blocked Acme" in out and "#3, #4" in out
        assert "unblock-org 'Acme'" in out
        record = store.switcher.blocked_orgs_record()
        assert record.orgs["Acme"].written_by == "hand"
        assert record.orgs["Acme"].evidence == "org disabled Claude Code"
        assert record.orgs["Acme"].written_at > 0

        _run("blocked-orgs")
        out = capsys.readouterr().out
        assert "Acme" in out and "by hand" in out and "org disabled Claude Code" in out

        _run("unblock-org", "Acme")
        assert "Unblocked Acme" in capsys.readouterr().out
        assert store.switcher.blocked_orgs_record().orgs == {}

    def test_the_email_domain_is_refused_as_a_label(self, store, capsys):
        with pytest.raises(SystemExit) as exc:
            _run("block-org", "acme.example", "--evidence", "x")
        assert exc.value.code == 1
        assert "no managed account shows the org label" in capsys.readouterr().err
        assert not _path(store).exists()

    def test_evidence_is_mandatory_to_block(self, store):
        with pytest.raises(SystemExit) as exc:
            _run("block-org", "Acme")
        assert exc.value.code == 2

    def test_there_is_no_timer_flag(self, store):
        for flag in ("--until", "--ttl", "--for", "--expires"):
            with pytest.raises(SystemExit) as exc:
                _run("block-org", "Acme", "--evidence", "x", flag, "1h")
            assert exc.value.code == 2, flag

    def test_the_watchdog_writes_and_clears_its_own(self, store, capsys):
        _run("block-org", "Acme", "--by", "watchdog", "--evidence", "ORG-DISABLED")
        assert store.switcher.blocked_orgs_record().orgs["Acme"].written_by == "watchdog"
        with pytest.raises(SystemExit):
            _run("unblock-org", "Acme", "--by", "watchdog")
        _run("unblock-org", "Acme", "--by", "watchdog", "--evidence", "RESOLVED")
        assert store.switcher.blocked_orgs_record().orgs == {}

    def test_unblocking_what_is_not_blocked_fails(self, store, capsys):
        with pytest.raises(SystemExit) as exc:
            _run("unblock-org", "Acme")
        assert exc.value.code == 1

    def test_init_and_json(self, store, capsys):
        _run("blocked-orgs", "--json")
        assert json.loads(capsys.readouterr().out)["problem"] == "missing"
        _run("blocked-orgs", "--init", "--json")
        payload = json.loads(capsys.readouterr().out)
        assert payload["orgs"] == {} and "problem" not in payload
        assert payload["path"] == str(_path(store))

    def test_a_broken_record_is_an_error_not_an_empty_list(self, store, capsys):
        _path(store).write_text("{broken")
        with pytest.raises(SystemExit) as exc:
            _run("blocked-orgs")
        assert exc.value.code == 1
        assert "NO org is being excluded" in capsys.readouterr().err

    @pytest.mark.parametrize("action", ["block-org", "unblock-org", "blocked-orgs"])
    def test_dispatched_from_main(self, temp_home, action):
        with patch("claude_swap.cli._blocked_orgs_command") as fn, \
             patch.object(sys, "argv", ["cswap", action, "Acme"]):
            cli.main()
        fn.assert_called_once_with(action, ["Acme"])


class TestListAndStatusShowTheBlock:
    def _entries(self, h):
        from tests.test_autoswitch import _entry_for

        return {str(n): _entry_for(_seat(), h.clock.now) for n in (1, 2, 3, 4)}

    def test_list_marks_the_seats_and_says_how_to_lift_it(self, store, capsys):
        _block(store)
        with patch.object(
            ClaudeAccountSwitcher, "_collect_usage_entries",
            return_value=self._entries(store),
        ):
            store.switcher.list_accounts()
        out = capsys.readouterr()
        text = out.out + out.err
        lines = {ln.split(":")[0].strip(): ln for ln in text.splitlines() if "[" in ln}
        assert "(org blocked)" in lines["3"] and "(org blocked)" in lines["4"]
        assert "(org blocked)" not in lines["2"]          # the acme.example DOMAIN
        assert "cswap unblock-org 'Acme'" in text

    def test_list_json_is_additive(self, store):
        with patch.object(
            ClaudeAccountSwitcher, "_collect_usage_entries",
            return_value=self._entries(store),
        ):
            clean = store.switcher.list_accounts(json_output=True)
            _block(store)
            payload = store.switcher.list_accounts(json_output=True)
        assert "blockedOrgs" not in clean
        assert all("orgBlocked" not in row for row in clean["accounts"])
        assert payload["blockedOrgs"] == ["Acme"]
        assert {r["number"] for r in payload["accounts"] if r.get("orgBlocked")} == {3, 4}

    def test_list_reports_a_broken_record(self, store, capsys):
        _path(store).write_text("{broken")
        with patch.object(
            ClaudeAccountSwitcher, "_collect_usage_entries",
            return_value=self._entries(store),
        ):
            store.switcher.list_accounts()
            payload = store.switcher.list_accounts(json_output=True)
        out = capsys.readouterr()
        assert "NO org is being excluded" in out.out + out.err
        assert payload["blockedOrgsProblem"].startswith("unparseable")

    def test_status_shows_it_for_the_active_seat(self, temp_home, capsys):
        h = _harness(temp_home, seats=(3, 1))
        _block(h)
        from tests.test_autoswitch import _entry_for

        with patch.object(
            ClaudeAccountSwitcher, "_active_account_usage",
            return_value=_entry_for(_seat(), h.clock.now),
        ):
            h.switcher.status()
            payload = h.switcher.status(json_output=True)
        out = capsys.readouterr()
        assert "This seat's org is blocked: Acme" in out.out + out.err
        assert payload["active"]["orgBlocked"] is True

    def test_an_explicit_switch_onto_a_blocked_seat_is_allowed_and_warned(self, store, capsys):
        _block(store)
        store.switcher.switch_to("3")
        out = capsys.readouterr()
        assert store.active_number() == 3           # a filter, not a lock
        assert "org is blocked (Acme)" in out.out + out.err
        make_live(store, 3)
        store.switcher.switch_to("2")
        out = capsys.readouterr()
        assert "org is blocked" not in out.out + out.err
