"""Plan-tier labels (subscription tier + Fable access) on every account surface.

Pure mapping/caching rules in ``claude_swap.plan_tier``; the profile read in
``oauth``; the collector's persist path in the switcher; and the renderers
(``cswap list`` / ``status`` text + JSON, TUI card and mini rows, menu bar,
auto-engine tick line).
"""

from __future__ import annotations

import json
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import menubar, oauth, plan_tier
from claude_swap.autoswitch import PollEvent
from claude_swap.credentials import ActiveCredentials
from claude_swap.json_output import account_row
from claude_swap.models import AccountSnapshot
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.tui.widgets import account_card_text, mini_account_text
from claude_swap.usage_store import FetchRecord, UsageEntry


# --------------------------------------------------------------------------- #
# Fake profile payloads — one per observed tier row, plus the documented extras
# --------------------------------------------------------------------------- #
def _profile(tier: str | None, org_type: str | None, *, has_max=False, has_pro=False) -> dict:
    org: dict = {"uuid": "org-1", "name": "Org", "billing_type": "stripe_subscription"}
    if tier is not None:
        org["rate_limit_tier"] = tier
    if org_type is not None:
        org["organization_type"] = org_type
    return {
        "account": {
            "uuid": "acct-1",
            "email": "user@example.com",
            "has_claude_max": has_max,
            "has_claude_pro": has_pro,
        },
        "organization": org,
    }


MAX_20X = _profile("default_claude_max_20x", "claude_max", has_max=True)
MAX_5X = _profile("default_claude_max_5x", "claude_max", has_max=True)
TEAM_PREMIUM = _profile("default_claude_max_5x", "claude_team")
TEAM_STANDARD = _profile("default_raven", "claude_team")
PRO = _profile("default_claude_pro", "claude_pro", has_pro=True)
ENTERPRISE = _profile("enterprise_tier_x", "claude_enterprise")
UNKNOWN = _profile("default_quokka_9x", "claude_zebra")

USAGE_WITH_FABLE = {
    "five_hour": {"pct": 10.0},
    "seven_day": {"pct": 20.0},
    "scoped": [{"name": "Fable", "pct": 32.0}],
}
USAGE_NO_FABLE = {"five_hour": {"pct": 10.0}, "seven_day": {"pct": 20.0}}


# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #
class TestTierLabel:
    @pytest.mark.parametrize(
        "payload, label",
        [
            (MAX_20X, "Max 20x"),
            (MAX_5X, "Max 5x"),
            (TEAM_PREMIUM, "Team premium"),
            (TEAM_STANDARD, "Team standard"),
            (PRO, "Pro"),
            (ENTERPRISE, "Enterprise"),
        ],
    )
    def test_observed_rows(self, payload, label):
        tier = plan_tier.tier_from_profile(payload)
        assert tier is not None
        assert tier.label == label

    def test_unknown_tier_shows_raw_string_not_a_guess(self):
        tier = plan_tier.tier_from_profile(UNKNOWN)
        assert tier is not None
        assert tier.label == "default_quokka_9x"
        assert tier.rate_limit_tier == "default_quokka_9x"
        assert tier.organization_type == "claude_zebra"

    def test_max_with_pro_flag_is_still_max(self):
        # Max plans include Pro; the flag must not demote the label.
        assert plan_tier.tier_label("default_claude_max_20x", "claude_max", True, True) == "Max 20x"

    def test_pro_flag_alone(self):
        assert plan_tier.tier_label(None, None, has_claude_pro=True) == "Pro"

    def test_team_with_unrecognised_tier_shows_raw(self):
        assert plan_tier.tier_label("default_new_thing", "claude_team") == "default_new_thing"

    def test_nothing_usable_is_none(self):
        assert plan_tier.tier_label(None, None) is None
        assert plan_tier.tier_label("", "  ") is None
        assert plan_tier.tier_from_profile({"account": {"uuid": "x"}}) is None
        assert plan_tier.tier_from_profile("not a dict") is None
        assert plan_tier.tier_from_profile({"organization": "oops"}) is None

    def test_raw_fields_preserved_verbatim(self):
        tier = plan_tier.tier_from_profile(TEAM_STANDARD)
        assert tier == plan_tier.PlanTier(
            label="Team standard",
            rate_limit_tier="default_raven",
            organization_type="claude_team",
            has_claude_max=False,
            has_claude_pro=False,
        )


class TestFableAccess:
    def test_present(self):
        assert plan_tier.fable_access(USAGE_WITH_FABLE) is True

    def test_case_insensitive(self):
        assert plan_tier.fable_access({"five_hour": {"pct": 1}, "scoped": [{"name": "fable", "pct": 0}]}) is True

    def test_absent_with_windows_is_false(self):
        assert plan_tier.fable_access(USAGE_NO_FABLE) is False

    def test_no_evidence_is_none(self):
        assert plan_tier.fable_access(None) is None
        assert plan_tier.fable_access({}) is None
        assert plan_tier.fable_access("api key") is None
        # scoped-only payload naming another model: no 5h/7d to prove the
        # payload was complete, and no Fable — still no verdict
        assert plan_tier.fable_access({"scoped": [{"name": "Other", "pct": 1}]}) is None


# --------------------------------------------------------------------------- #
# Cache TTL, stale-on-error, display
# --------------------------------------------------------------------------- #
class TestCacheRules:
    def test_due_when_never_fetched(self):
        assert plan_tier.tier_due(None, now=1000.0)
        assert plan_tier.tier_due({}, now=1000.0)
        assert plan_tier.tier_due({"tierFetchedAt": "garbage"}, now=1000.0)

    def test_not_due_inside_ttl_due_after(self):
        now = 1_800_000_000.0
        rec = plan_tier.tier_record_fields(plan_tier.tier_from_profile(MAX_5X), now)
        assert plan_tier.tier_fetched_at(rec) == pytest.approx(now, abs=1)
        assert not plan_tier.tier_due(rec, now + 3600)
        assert not plan_tier.tier_due(rec, now + plan_tier.TIER_TTL_S - 1)
        assert plan_tier.tier_due(rec, now + plan_tier.TIER_TTL_S)

    def test_error_keeps_cached_tier(self):
        now = 1_800_000_000.0
        rec = plan_tier.tier_record_fields(plan_tier.tier_from_profile(MAX_5X), now)
        plan_tier.merge_record_fields(rec, plan_tier.fable_record_fields(True))
        changed = plan_tier.merge_record_fields(rec, plan_tier.tier_error_fields("network", now + 5))
        assert changed
        assert rec["planTier"] == "Max 5x"
        assert rec["fableAccess"] is True
        assert rec["tierError"] == "network"
        assert rec["tierAttemptedAt"] != rec["tierFetchedAt"]
        # ...and is still displayed
        assert plan_tier.tier_display(rec) == "Max 5x · Fable"

    def test_success_clears_error(self):
        rec = {"tierError": "network"}
        plan_tier.merge_record_fields(
            rec, plan_tier.tier_record_fields(plan_tier.tier_from_profile(TEAM_STANDARD), 5.0)
        )
        assert rec["tierError"] is None

    def test_failed_read_is_retried_hourly_not_per_poll(self):
        now = 1_800_000_000.0
        rec: dict = {}
        plan_tier.merge_record_fields(rec, plan_tier.tier_error_fields("network", now))
        assert not plan_tier.tier_due(rec, now + 60)
        assert not plan_tier.tier_due(rec, now + plan_tier.TIER_RETRY_S - 1)
        assert plan_tier.tier_due(rec, now + plan_tier.TIER_RETRY_S)
        # a fresh success is never re-read inside its TTL, whatever the attempt stamp
        rec.update(plan_tier.tier_record_fields(plan_tier.tier_from_profile(PRO), now + 7200))
        assert not plan_tier.tier_due(rec, now + 7200 + plan_tier.TIER_RETRY_S + 1)
        # a clock that went backwards does not wedge the retry
        assert plan_tier.tier_due({"tierAttemptedAt": plan_tier._iso_z(now + 9999)}, now)

    def test_none_fable_never_overwrites_known_value(self):
        rec = {"fableAccess": True}
        assert plan_tier.fable_record_fields(None) == {}
        assert not plan_tier.merge_record_fields(rec, plan_tier.fable_record_fields(None))
        assert rec["fableAccess"] is True

    def test_merge_reports_no_change_for_identical_values(self):
        rec = {"planTier": "Pro", "fableAccess": False}
        assert not plan_tier.merge_record_fields(rec, {"planTier": "Pro", "fableAccess": False})

    def test_401_without_cache_reads_as_relogin(self):
        assert plan_tier.tier_display({"tierError": "http-401"}) == "tier: re-login needed"
        # a cached tier wins over the 401 note
        assert plan_tier.tier_display({"planTier": "Pro", "tierError": "http-401"}) == "Pro"

    def test_unknown_record_has_no_label(self):
        assert plan_tier.tier_display(None) is None
        assert plan_tier.tier_display({}) is None
        assert plan_tier.tier_display({"tierError": "network"}) is None


class TestDisplayStrings:
    """The exact bracket contents each surface prints."""

    @pytest.mark.parametrize(
        "payload, access, full, compact",
        [
            (TEAM_PREMIUM, True, "Team premium · Fable", "Team prem"),
            (TEAM_STANDARD, False, "Team standard · no Fable", "Team std · no Fable"),
            (MAX_20X, True, "Max 20x · Fable", "Max 20x"),
            (MAX_5X, None, "Max 5x", "Max 5x"),
            (PRO, False, "Pro · no Fable", "Pro · no Fable"),
            (ENTERPRISE, True, "Enterprise · Fable", "Ent"),
            (UNKNOWN, True, "default_quokka_9x · Fable", "default_quokka_9x"),
        ],
    )
    def test_full_and_compact(self, payload, access, full, compact):
        rec = plan_tier.tier_record_fields(plan_tier.tier_from_profile(payload), 0.0)
        rec.update(plan_tier.fable_record_fields(access))
        assert plan_tier.tier_display(rec) == full
        assert plan_tier.tier_display(rec, compact=True) == compact

    def test_fable_only_when_tier_unknown(self):
        assert plan_tier.tier_display({"fableAccess": False}) == "no Fable"
        assert plan_tier.tier_display({"fableAccess": True}) == "Fable"


class TestJsonFields:
    def test_always_emits_the_five_keys(self):
        out = plan_tier.tier_json_fields(None)
        assert out == {
            "planTier": None,
            "rateLimitTier": None,
            "organizationType": None,
            "fableAccess": None,
            "tierFetchedAt": None,
        }

    def test_projects_record(self):
        rec = plan_tier.tier_record_fields(plan_tier.tier_from_profile(TEAM_STANDARD), 1_800_000_000.0)
        rec.update(plan_tier.fable_record_fields(False))
        out = plan_tier.tier_json_fields(rec)
        assert out["planTier"] == "Team standard"
        assert out["rateLimitTier"] == "default_raven"
        assert out["organizationType"] == "claude_team"
        assert out["fableAccess"] is False
        assert out["tierFetchedAt"] == "2027-01-15T08:00:00Z"
        assert "tierError" not in out

    def test_error_is_additive(self):
        assert plan_tier.tier_json_fields({"tierError": "http-401"})["tierError"] == "http-401"
        assert "tierAttemptedAt" not in plan_tier.tier_json_fields({"tierAttemptedAt": "2027-01-01T00:00:00Z"})

    def test_account_row_merges_tier_fields(self):
        row = account_row(
            1, "a@x.com", "", "", True, None,
            tier=plan_tier.tier_json_fields({"planTier": "Max 20x", "fableAccess": True}),
        )
        assert row["planTier"] == "Max 20x"
        assert row["fableAccess"] is True
        # the base schema is untouched
        assert row["number"] == 1 and row["usageStatus"] == "unavailable"

    def test_account_row_without_tier_is_unchanged(self):
        row = account_row(1, "a@x.com", "", "", True, None)
        assert "planTier" not in row


# --------------------------------------------------------------------------- #
# The profile read itself
# --------------------------------------------------------------------------- #
@pytest.mark.no_oauth_profile_fake
class TestFetchPlanProfile:
    def _response(self, payload: dict):
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_success_returns_raw_body(self):
        seen = {}

        def fake_urlopen(req, timeout):
            seen["auth"] = req.get_header("Authorization")
            seen["timeout"] = timeout
            return self._response(TEAM_PREMIUM)

        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=fake_urlopen):
            out = oauth.fetch_oauth_plan_profile("sk-live")
        assert out.error is None
        assert out.data == TEAM_PREMIUM
        assert seen["auth"] == "Bearer sk-live"
        assert seen["timeout"] == 5.0

    def test_401_is_classified(self):
        err = urllib.error.HTTPError("u", 401, "unauthorized", {}, None)
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err):
            out = oauth.fetch_oauth_plan_profile("sk-dead")
        assert out == oauth.ProfileOutcome(None, error="http-401")

    def test_network_failure_is_transient_kind(self):
        with patch(
            "claude_swap.oauth.urllib.request.urlopen",
            side_effect=urllib.error.URLError("unreachable"),
        ):
            out = oauth.fetch_oauth_plan_profile("sk-x")
        assert out.data is None and out.error == "network"

    def test_non_object_body_is_bad_response(self):
        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=self._response([1, 2])):
            assert oauth.fetch_oauth_plan_profile("sk-x").error == "bad-response"

    def test_usage_outcome_carries_accepted_token(self):
        usage_body = {"five_hour": {"utilization": 5.0}, "seven_day": {"utilization": 6.0}}
        with patch("claude_swap.oauth.urllib.request.urlopen", return_value=self._response(usage_body)):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@x.com", json.dumps({"claudeAiOauth": {"accessToken": "sk-ok"}}), is_active=True
            )
        assert outcome.error is None
        assert outcome.access_token == "sk-ok"

    def test_usage_outcome_has_no_token_on_failure(self):
        err = urllib.error.HTTPError("u", 429, "slow down", {}, None)
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=err):
            outcome = oauth.try_fetch_usage_for_account(
                "1", "a@x.com", json.dumps({"claudeAiOauth": {"accessToken": "sk-ok"}}), is_active=True
            )
        assert outcome.error == "http-429"
        assert outcome.access_token is None


# --------------------------------------------------------------------------- #
# Collector integration: fetch → cache → render
# --------------------------------------------------------------------------- #
def _seed(switcher: ClaudeAccountSwitcher, seq: dict) -> None:
    switcher._setup_directories()
    switcher._write_json(switcher.sequence_file, seq)


def _record(switcher: ClaudeAccountSwitcher, num: str) -> dict:
    return json.loads(switcher.sequence_file.read_text())["accounts"][num]


ACTIVE = json.dumps({"claudeAiOauth": {"accessToken": "sk-active", "expiresAt": 4_000_000_000_000}})
BACKUP = json.dumps({"claudeAiOauth": {"accessToken": "sk-backup", "expiresAt": 4_000_000_000_000}})


class TestCollectorPersistsTier:
    def _run_list(self, switcher, usage_outcome, profile_outcome, json_output=False, refresh=False):
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch.object(switcher, "_read_account_credentials", return_value=BACKUP), \
             patch("claude_swap.oauth.try_fetch_usage_for_account", return_value=usage_outcome), \
             patch("claude_swap.oauth.fetch_oauth_plan_profile",
                   return_value=profile_outcome) as profile_mock:
            payload = switcher.list_accounts(json_output=json_output, refresh=refresh)
        return payload, profile_mock

    def test_first_pass_fetches_profile_and_caches_tier(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)

        _, profile_mock = self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_WITH_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(TEAM_PREMIUM),
        )

        # one profile read per account, with the token the usage endpoint accepted
        assert profile_mock.call_count == 2
        assert {c.args[0] for c in profile_mock.call_args_list} == {"sk-accepted"}
        rec = _record(switcher, "1")
        assert rec["planTier"] == "Team premium"
        assert rec["rateLimitTier"] == "default_claude_max_5x"
        assert rec["organizationType"] == "claude_team"
        assert rec["hasClaudeMax"] is False
        assert rec["fableAccess"] is True
        assert rec["tierError"] is None
        assert rec["tierFetchedAt"].endswith("Z")
        # alias/identity fields untouched
        assert rec["email"] == "test@example.com" and rec["uuid"] == "uuid-1"

        out = capsys.readouterr().out
        assert "  1: test@example.com [personal] [Team premium · Fable] (active)" in out
        assert "  2: account2@example.com [personal] [Team premium · Fable]" in out

    def test_team_standard_seat_renders_no_fable(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(TEAM_STANDARD),
        )
        assert _record(switcher, "2")["fableAccess"] is False
        assert "[Team standard · no Fable]" in capsys.readouterr().out

    def test_fresh_cache_skips_the_profile_read(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        for rec in sample_sequence_data["accounts"].values():
            rec.update(plan_tier.tier_record_fields(plan_tier.tier_from_profile(MAX_20X), time.time() - 60))
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        _, profile_mock = self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_WITH_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(TEAM_STANDARD),
        )
        profile_mock.assert_not_called()
        # ...but the Fable flag is still refreshed from the usage payload
        assert _record(switcher, "1")["planTier"] == "Max 20x"
        assert _record(switcher, "1")["fableAccess"] is True

    def test_expired_cache_refetches(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        for rec in sample_sequence_data["accounts"].values():
            rec.update(plan_tier.tier_record_fields(
                plan_tier.tier_from_profile(TEAM_PREMIUM), time.time() - plan_tier.TIER_TTL_S - 5
            ))
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        _, profile_mock = self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(TEAM_STANDARD),
        )
        assert profile_mock.call_count == 2
        rec = _record(switcher, "1")
        assert rec["planTier"] == "Team standard"
        assert rec["fableAccess"] is False

    def test_transient_error_keeps_cached_tier(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        for rec in sample_sequence_data["accounts"].values():
            rec.update(plan_tier.tier_record_fields(
                plan_tier.tier_from_profile(MAX_5X), time.time() - plan_tier.TIER_TTL_S - 5
            ))
            rec["fableAccess"] = True
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_WITH_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(None, error="network"),
        )
        rec = _record(switcher, "1")
        assert rec["planTier"] == "Max 5x"  # never blanked
        assert rec["tierError"] == "network"
        assert "[Max 5x · Fable]" in capsys.readouterr().out

    def test_401_without_cache_shows_relogin(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(None, error="http-401"),
        )
        assert _record(switcher, "1")["tierError"] == "http-401"
        assert "[tier: re-login needed]" in capsys.readouterr().out

    def test_no_profile_read_without_an_accepted_token(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        """A usage failure (no token echoed back) must not cost a second request."""
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        _, profile_mock = self._run_list(
            switcher,
            oauth.UsageOutcome(None, error="http-429"),
            oauth.ProfileOutcome(TEAM_PREMIUM),
        )
        profile_mock.assert_not_called()
        assert "planTier" not in _record(switcher, "1")

    def test_json_rows_carry_additive_keys(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        payload, _ = self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_WITH_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(MAX_20X),
            json_output=True,
        )
        row = next(a for a in payload["accounts"] if a["number"] == 1)
        assert row["planTier"] == "Max 20x"
        assert row["rateLimitTier"] == "default_claude_max_20x"
        assert row["organizationType"] == "claude_max"
        assert row["fableAccess"] is True
        assert row["tierFetchedAt"].endswith("Z")
        assert row["usageStatus"] == "ok"  # nothing renamed

    def test_status_text_and_json(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        sample_sequence_data["accounts"]["1"].update(
            plan_tier.tier_record_fields(plan_tier.tier_from_profile(TEAM_STANDARD), time.time())
        )
        sample_sequence_data["accounts"]["1"]["fableAccess"] = False
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch("claude_swap.oauth.try_fetch_usage_for_account",
                   return_value=oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-a")):
            payload = switcher.status(json_output=True)
            switcher.status()
        assert payload["active"]["planTier"] == "Team standard"
        assert payload["active"]["fableAccess"] is False
        assert payload["active"]["rateLimitTier"] == "default_raven"
        out = capsys.readouterr().out
        assert "Status: Account-1 (test@example.com [personal]) [Team standard · no Fable]" in out

    def test_refresh_forces_a_profile_read_even_with_fresh_cache(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        for rec in sample_sequence_data["accounts"].values():
            rec.update(plan_tier.tier_record_fields(plan_tier.tier_from_profile(MAX_20X), time.time()))
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        _, profile_mock = self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(TEAM_STANDARD),
            refresh=True,
        )
        # exactly one read per slot: the direct probe covered both, so the
        # collect pass did not read them again
        assert profile_mock.call_count == 2
        assert _record(switcher, "1")["planTier"] == "Team standard"
        assert _record(switcher, "2")["planTier"] == "Team standard"

    def test_refresh_falls_back_to_the_collect_pass_for_unprobeable_slots(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        """Slot 2's stored token is expired: the direct probe skips it and the
        collect pass (whose fetch would refresh it) is asked to cover it."""
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        for rec in sample_sequence_data["accounts"].values():
            rec.update(plan_tier.tier_record_fields(plan_tier.tier_from_profile(MAX_20X), time.time()))
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        expired_backup = json.dumps({"claudeAiOauth": {"accessToken": "sk-old", "expiresAt": 1}})
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch.object(switcher, "_read_account_credentials", return_value=expired_backup), \
             patch("claude_swap.oauth.try_fetch_usage_for_account",
                   return_value=oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-refreshed")), \
             patch("claude_swap.oauth.fetch_oauth_plan_profile",
                   return_value=oauth.ProfileOutcome(TEAM_STANDARD)) as profile_mock:
            switcher.list_accounts(refresh=True)
        tokens = [c.args[0] for c in profile_mock.call_args_list]
        assert tokens.count("sk-active") == 1      # slot 1: direct probe only
        assert tokens.count("sk-refreshed") == 1   # slot 2: collect pass only
        assert profile_mock.call_count == 2

    def test_tier_fields_never_reach_the_usage_store(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        self._run_list(
            switcher,
            oauth.UsageOutcome(USAGE_WITH_FABLE, access_token="sk-accepted"),
            oauth.ProfileOutcome(TEAM_PREMIUM),
        )
        rows = json.dumps(switcher._usage_store._read_rows())
        assert "sk-accepted" not in rows
        assert "access_token" not in rows and "profile" not in rows

    def test_tier_due_slots_respects_ttl_and_force(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        sample_sequence_data["accounts"]["1"].update(
            plan_tier.tier_record_fields(plan_tier.tier_from_profile(PRO), time.time())
        )
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        assert switcher._tier_due_slots({"1", "2"}) == {"2"}
        assert switcher._tier_due_slots({"1", "2"}, force={"1"}) == {"1", "2"}
        assert switcher._tier_due_slots({"2"}, force={"1"}) == {"2"}  # force is scoped to the pass

    def test_persist_ignores_removed_slot_and_writes_only_on_change(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        before = switcher.sequence_file.stat().st_mtime_ns
        switcher._persist_tier_fields({"9": {"planTier": "Pro"}})  # no such slot
        assert switcher.sequence_file.stat().st_mtime_ns == before
        switcher._persist_tier_fields({"1": {"planTier": "Pro"}})
        assert _record(switcher, "1")["planTier"] == "Pro"

    def test_persist_drops_update_when_slot_changed_identity(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """A swap committing while the profile request was in flight: the
        slot number now names another account, so the answer is dropped."""
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        switcher._persist_tier_fields(
            {"1": {"planTier": "Pro"}, "2": {"planTier": "Max 5x"}},
            {"1": ("someone-else@example.com", ""), "2": ("account2@example.com", "")},
        )
        assert "planTier" not in _record(switcher, "1")
        assert _record(switcher, "2")["planTier"] == "Max 5x"

    def test_status_reflects_a_tier_cached_in_the_same_call(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch("claude_swap.oauth.try_fetch_usage_for_account",
                   return_value=oauth.UsageOutcome(USAGE_WITH_FABLE, access_token="sk-a")), \
             patch("claude_swap.oauth.fetch_oauth_plan_profile",
                   return_value=oauth.ProfileOutcome(MAX_20X)):
            payload = switcher.status(json_output=True)
        assert payload["active"]["planTier"] == "Max 20x"
        assert payload["active"]["fableAccess"] is True

        # text form, on a fresh roster (usage now cached, tier not yet)
        data = json.loads(switcher.sequence_file.read_text())
        for rec in data["accounts"].values():
            for key in plan_tier.RECORD_KEYS:
                rec.pop(key, None)
        switcher._write_json(switcher.sequence_file, data)
        switcher._usage_store._write_rows({})  # force a fresh fetch
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch("claude_swap.oauth.try_fetch_usage_for_account",
                   return_value=oauth.UsageOutcome(USAGE_NO_FABLE, access_token="sk-a")), \
             patch("claude_swap.oauth.fetch_oauth_plan_profile",
                   return_value=oauth.ProfileOutcome(TEAM_STANDARD)):
            switcher.status()
        assert "[Team standard · no Fable]" in capsys.readouterr().out

    def test_refresh_prefers_a_live_session_credential(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        """An inactive slot run under `cswap run` holds its newest token in
        the session profile; the backup is a consumed generation."""
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        expired_backup = json.dumps({"claudeAiOauth": {"accessToken": "sk-old", "expiresAt": 1}})
        session = json.dumps({"claudeAiOauth": {"accessToken": "sk-session", "expiresAt": 4_000_000_000_000}})
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch.object(switcher, "_read_account_credentials", return_value=expired_backup), \
             patch("claude_swap.switcher.ClaudeAccountSwitcher._session_dir",
                   return_value=temp_home / "sess"), \
             patch("claude_swap.session.read_session_credentials",
                   side_effect=lambda d: session), \
             patch("claude_swap.session.session_identity_drifted", return_value=False), \
             patch("claude_swap.oauth.fetch_oauth_plan_profile",
                   return_value=oauth.ProfileOutcome(TEAM_PREMIUM)) as profile_mock:
            probed = switcher.refresh_plan_tiers(switcher._build_accounts_info())
        assert probed == {"1", "2"}
        tokens = sorted(c.args[0] for c in profile_mock.call_args_list)
        assert tokens == ["sk-active", "sk-session"]
        assert _record(switcher, "2")["planTier"] == "Team premium"

    def test_probe_on_add_skips_expired_token(self, temp_home: Path, sample_sequence_data: dict):
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        expired = json.dumps({"claudeAiOauth": {"accessToken": "sk-old", "expiresAt": 1}})
        with patch("claude_swap.oauth.fetch_oauth_plan_profile") as m:
            assert switcher._probe_plan_tier("1", expired) is False
        m.assert_not_called()
        assert "planTier" not in _record(switcher, "1")

    def test_probe_on_add_writes_tier(self, temp_home: Path, sample_sequence_data: dict):
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        with patch("claude_swap.oauth.fetch_oauth_plan_profile",
                   return_value=oauth.ProfileOutcome(ENTERPRISE)) as m:
            assert switcher._probe_plan_tier("1", ACTIVE) is True
        m.assert_called_once_with("sk-active")
        assert _record(switcher, "1")["planTier"] == "Enterprise"

    def test_plan_tier_labels_for_the_engine(self, temp_home: Path, sample_sequence_data: dict):
        sample_sequence_data["accounts"]["2"].update(
            plan_tier.tier_record_fields(plan_tier.tier_from_profile(TEAM_STANDARD), 0.0)
        )
        sample_sequence_data["accounts"]["2"]["fableAccess"] = False
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        assert switcher.plan_tier_labels() == {"2": "Team std · no Fable"}
        assert switcher.plan_tier_labels(compact=False) == {"2": "Team standard · no Fable"}

    def test_snapshot_rows_carry_tier(self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        sample_sequence_data["accounts"]["1"].update(
            plan_tier.tier_record_fields(plan_tier.tier_from_profile(MAX_5X), time.time())
        )
        sample_sequence_data["accounts"]["1"]["fableAccess"] = True
        switcher = ClaudeAccountSwitcher()
        _seed(switcher, sample_sequence_data)
        with patch.object(switcher, "_read_active_credentials",
                          return_value=ActiveCredentials(ACTIVE, False)), \
             patch.object(switcher, "_read_account_credentials", return_value=""), \
             patch("claude_swap.oauth.try_fetch_usage_for_account",
                   return_value=oauth.UsageOutcome(None)):
            snap = switcher.accounts_snapshot()
        one = next(a for a in snap.accounts if a.number == "1")
        two = next(a for a in snap.accounts if a.number == "2")
        assert one.plan_tier == "Max 5x" and one.fable_access is True
        assert one.tier_label == "Max 5x · Fable"
        assert one.tier_label_compact == "Max 5x"
        assert two.plan_tier is None and two.tier_label is None


# --------------------------------------------------------------------------- #
# Interactive surfaces
# --------------------------------------------------------------------------- #
def _snap(number: str, *, active=False, **tier) -> AccountSnapshot:
    return AccountSnapshot(
        number=number, email=f"u{number}@example.com", org_name="", org_uuid="",
        is_active=active, kind="oauth", switchable=True,
        usage=UsageEntry(last_good={"five_hour": {"pct": 12.0}}, fetched_at=time.time(), age_s=1.0),
        **tier,
    )


class TestTuiRows:
    def test_card_shows_tier_after_org_tag(self):
        acc = _snap("1", active=True, plan_tier="Team premium", fable_access=True)
        card = account_card_text(acc, 100).plain
        assert "u1@example.com  [personal]  [Team premium · Fable]   ● active" in card

    def test_mini_shows_tier(self):
        acc = _snap("2", plan_tier="Team standard", fable_access=False)
        assert "[personal]  [Team standard · no Fable]" in mini_account_text(acc, time.time()).plain

    def test_unknown_tier_adds_nothing(self):
        acc = _snap("3")
        assert "[personal]   ● " not in account_card_text(acc, 100).plain  # not active
        assert account_card_text(acc, 100).plain.count("[") == 1
        assert mini_account_text(acc, time.time()).plain.count("[") == 1

    def test_relogin_note_on_card(self):
        acc = _snap("4", tier_error="http-401")
        assert "[tier: re-login needed]" in account_card_text(acc, 100).plain


class TestMenuBar:
    def test_label_with_tier(self):
        label = menubar.format_account_label(
            2, "loc@example.com", {"five_hour": {"pct": 10.0}}, alias="dev", tier="Team std · no Fable"
        )
        assert label.startswith("2  dev  (loc@example.com)  [Team std · no Fable]  ")

    def test_label_without_tier_is_unchanged(self):
        assert menubar.format_account_label(2, "loc@example.com", None) == "2  loc@example.com  usage unavailable"

    def test_adapt_snapshot_adds_compact_tiers(self):
        class Snap:
            accounts = [
                _snap("1", active=True, plan_tier="Max 20x", fable_access=True),
                _snap("2", plan_tier="Team standard", fable_access=False),
                _snap("3"),
            ]
        adapted = menubar._adapt_snapshot(Snap())
        assert adapted["tiers"] == {"1": "Max 20x", "2": "Team std · no Fable"}
        # tuple rows keep their 8-field shape
        assert len(adapted["accounts"][0]) == 8

    def test_adapt_snapshot_tolerates_rows_without_tier_attr(self):
        class Acc:
            number, email, is_active, alias, disabled = "1", "a@x.com", True, "", False
            usage = UsageEntry()

        class Snap:
            accounts = [Acc()]
        assert menubar._adapt_snapshot(Snap())["tiers"] == {}
        assert menubar._adapt_snapshot(type("S", (), {"accounts": []})()) == menubar.EMPTY_SNAPSHOT


class TestTickLine:
    def test_tiers_in_human_line_and_json(self):
        ev = PollEvent(
            active={"number": 1, "email": "a@x.com"},
            headroom={"1": 60.0, "2": 90.0, "3": None},
            threshold=90.0,
            windows={"2": {"5h": 10.0, "7d": 5.0}},
            fetch_errors={"3": "http-429"},
            tiers={"2": "Team std · no Fable", "3": "Max 20x"},
        )
        line = ev.human()
        assert "#2: 5h 10% · 7d 5% · Team std · no Fable" in line
        assert "#3: ? (http-429) · Max 20x" in line
        assert ev._fields()["planTiers"] == {"2": "Team std · no Fable", "3": "Max 20x"}

    def test_no_tiers_leaves_line_and_json_unchanged(self):
        ev = PollEvent(active={"number": 1, "email": "a@x.com"}, headroom={"1": 60.0, "2": 90.0}, threshold=90.0)
        assert ev.human().endswith("| others: #2: 10%")
        assert "planTiers" not in ev._fields()


class TestFetchRecordInMemoryFields:
    def test_defaults_and_replace(self):
        rec = FetchRecord(usage=USAGE_WITH_FABLE)
        assert rec.access_token is None and rec.profile is None
        fields = ClaudeAccountSwitcher._tier_fields_from_record(
            FetchRecord(usage=USAGE_WITH_FABLE, profile=oauth.ProfileOutcome(TEAM_PREMIUM)), 0.0
        )
        assert fields["planTier"] == "Team premium" and fields["fableAccess"] is True

    def test_failure_and_sentinel_records_yield_nothing(self):
        assert ClaudeAccountSwitcher._tier_fields_from_record(FetchRecord(error="http-429"), 0.0) == {}
        assert ClaudeAccountSwitcher._tier_fields_from_record(FetchRecord(sentinel="api key"), 0.0) == {}

    def test_profile_without_tier_fields_records_a_distinct_error(self):
        fields = ClaudeAccountSwitcher._tier_fields_from_record(
            FetchRecord(usage=USAGE_NO_FABLE, profile=oauth.ProfileOutcome({"account": {"uuid": "x"}})), 0.0
        )
        assert fields["fableAccess"] is False
        assert fields["tierError"] == "no-tier-fields"
        assert fields["tierAttemptedAt"] == "1970-01-01T00:00:00Z"
        assert "planTier" not in fields
