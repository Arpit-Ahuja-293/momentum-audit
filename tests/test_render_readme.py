import json
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import render_readme


def minimal_payload():
    return {
        "provenance": {
            "data_start": "2010-01-04", "data_end": "2026-08-28", "n_tickers": 101,
            "universe_scraped_on": "2026-08-31", "seed": 20260831, "bps_per_side": 7.5,
            "permutation_draws": 1000, "bootstrap_draws": 1000, "sweep_max_draws": 500,
            "n_configs": 32, "git_commit": "abc123", "run_on": "2026-08-31",
            "execution_lag_days": 2, "risk_free_rate": 0.0,
        },
        "baseline": {"sharpe": 0.41, "ann_return": 0.031, "ann_vol": 0.076,
                     "max_drawdown": -0.28, "turnover_one_way": 3.4, "hit_rate": 0.54,
                     "n_days": 4100, "start": "2011-01-03", "end": "2026-08-28",
                     "config": {"lookback": 12, "skip": 1, "rebalance_months": 1, "decile": 0.1}},
        "walkforward": {"summary": {"sharpe": 0.12, "ann_return": 0.009,
                                    "max_drawdown": -0.22, "turnover_one_way": 3.5},
                        "mean_is_sharpe": 0.62, "is_oos_gap": 0.50, "folds": []},
        "nulls": {"permutation": {"mean": -0.05, "std": 0.33, "q95": 0.51, "pvalue": 0.108},
                  "block_bootstrap": {"mean": 0.0, "std": 0.3, "q95": 0.49, "pvalue": 0.13}},
        "sweep": {"best_key": "lb12_sk1_rb1_dc20", "table": [{"key": "lb12_sk1_rb1_dc20", "sharpe": 0.77}],
                  "best_config": {"lookback": 12, "skip": 1, "rebalance_months": 1, "decile": 0.2},
                  "deflated_sharpe": {"dsr": 0.21, "n_trials": 32},
                  "bonferroni": {"n_survivors_corrected": 0, "n_survivors_raw": 3,
                                 "threshold": 0.0015625, "n_tests": 32},
                  "sweep_max_null": {"pvalue": 0.42, "q95": 1.02, "mean": 0.71},
                  "per_config_pvalues": {}},
        "costs": {"breakeven_bps_return_oos": 9.2, "breakeven_bps_sharpe_oos": 9.2,
                  "breakeven_bps_return_full": 14.0, "breakeven_bps_sharpe_full": 14.0,
                  "full_sample_curve": [], "oos_curve": []},
        "references": {"long_only_decile": {"sharpe": 0.7, "ann_return": 0.11,
                                            "max_drawdown": -0.4},
                       "spy": {"sharpe": 0.8, "ann_return": 0.13, "max_drawdown": -0.34}},
    }


def test_context_contains_every_headline_number():
    ctx = render_readme.build_context(minimal_payload())
    for key in ["baseline_sharpe", "oos_sharpe", "is_oos_gap", "permutation_pvalue",
                "dsr", "sweep_max_pvalue", "breakeven_bps_oos", "baseline_ann_return",
                "max_drawdown", "turnover", "n_configs", "data_start", "data_end",
                "n_tickers", "permutation_draws", "bonferroni_survivors"]:
        assert key in ctx, f"missing context key: {key}"


def test_numbers_are_formatted_not_raw_floats():
    ctx = render_readme.build_context(minimal_payload())
    assert ctx["baseline_sharpe"] == "0.41"
    assert ctx["permutation_pvalue"] == "0.108"
    assert ctx["baseline_ann_return"] == "3.1%"
    assert ctx["max_drawdown"] == "-28.0%"


def test_breakeven_none_renders_as_words_not_null():
    payload = minimal_payload()
    payload["costs"]["breakeven_bps_return_oos"] = None
    ctx = render_readme.build_context(payload)
    assert "50" in ctx["breakeven_bps_oos"] or "survives" in ctx["breakeven_bps_oos"].lower()


def test_render_substitutes_placeholders():
    out = render_readme.render("Sharpe was {{baseline_sharpe}}.", {"baseline_sharpe": "0.41"})
    assert out == "Sharpe was 0.41."


def test_render_raises_on_an_unknown_placeholder():
    with pytest.raises(KeyError, match="mystery"):
        render_readme.render("Value: {{mystery}}", {"baseline_sharpe": "0.41"})


def test_verdict_language_follows_the_evidence():
    """A p-value that fails to reject must not produce triumphant prose."""
    payload = minimal_payload()
    ctx = render_readme.build_context(payload)
    assert ctx["verdict_word"] in {"does not survive", "survives"}
    assert ctx["verdict_word"] == "does not survive"

    payload["nulls"]["permutation"]["pvalue"] = 0.001
    payload["sweep"]["sweep_max_null"]["pvalue"] = 0.004
    payload["walkforward"]["summary"]["sharpe"] = 0.9
    payload["costs"]["breakeven_bps_return_oos"] = 45.0
    assert render_readme.build_context(payload)["verdict_word"] == "survives"


def test_breakeven_context_reads_as_a_sentence_either_way():
    payload = minimal_payload()
    ctx = render_readme.build_context(payload)
    assert ctx["breakeven_sentence"].endswith(".")
    assert "9.2 bps" in ctx["breakeven_sentence"]

    payload["costs"]["breakeven_bps_return_oos"] = None
    ctx = render_readme.build_context(payload)
    assert ctx["breakeven_sentence"].endswith(".")
    assert "never" in ctx["breakeven_sentence"].lower()
    assert "reaches zero at survives" not in ctx["breakeven_sentence"]


def test_walkforward_verb_follows_the_direction_of_the_gap():
    payload = minimal_payload()
    assert render_readme.build_context(payload)["oos_verb"] == "falls to"

    payload["walkforward"]["mean_is_sharpe"] = 0.10
    payload["walkforward"]["is_oos_gap"] = -0.02
    payload["walkforward"]["summary"]["sharpe"] = 0.12
    assert render_readme.build_context(payload)["oos_verb"] == "holds at"


def test_bonferroni_note_calls_out_an_unresolvable_threshold():
    payload = minimal_payload()
    payload["sweep"]["bonferroni"]["p_resolution"] = 1 / 201
    payload["sweep"]["bonferroni"]["resolvable"] = False
    note = render_readme.build_context(payload)["bonferroni_note"]
    assert "resolve" in note.lower() or "cannot" in note.lower()

    payload["sweep"]["bonferroni"]["p_resolution"] = 1 / 1001
    payload["sweep"]["bonferroni"]["resolvable"] = True
    assert render_readme.build_context(payload)["bonferroni_note"] == ""


def test_reference_sharpes_are_reported_on_both_windows():
    """A benchmark on a different window than the headline must be labelled.

    The walk-forward series starts years after the full sample, so a single
    unlabelled reference Sharpe would be compared against whichever headline
    number the reader's eye landed on.
    """
    payload = minimal_payload()
    payload["references"] = {
        "window_full_sample": {"long_only_decile": {"sharpe": 1.05},
                               "spy": {"sharpe": 0.86}},
        "window_oos": {"long_only_decile": {"sharpe": 0.91},
                       "spy": {"sharpe": 0.78}},
    }
    ctx = render_readme.build_context(payload)
    assert ctx["long_only_sharpe"] == "1.05"
    assert ctx["spy_sharpe"] == "0.86"
    assert ctx["long_only_sharpe_oos"] == "0.91"
    assert ctx["spy_sharpe_oos"] == "0.78"


def test_reference_windows_fall_back_for_payloads_without_them():
    """Runs written before the windowed references still render."""
    ctx = render_readme.build_context(minimal_payload())
    assert ctx["long_only_sharpe"] == "0.70"
    assert ctx["long_only_sharpe_oos"] == "0.70"


def test_both_deflated_sharpe_variants_reach_the_context():
    payload = minimal_payload()
    payload["sweep"]["deflated_sharpe"] = {
        "dsr": 0.87, "n_trials": 32, "variance_source": "across_trial",
    }
    payload["sweep"]["deflated_sharpe_null_variance"] = {"dsr": 0.42, "n_trials": 32}
    ctx = render_readme.build_context(payload)
    assert ctx["dsr"] == "0.870"
    assert ctx["dsr_null_variance"] == "0.420"
    assert ctx["dsr_variance_source"] == "across_trial"


def test_a_dirty_working_tree_is_disclosed_next_to_the_commit():
    """A commit hash only traces a run if the tree matched it at the time."""
    payload = minimal_payload()
    payload["provenance"]["git_dirty"] = True
    caveat = render_readme.build_context(payload)["provenance_caveat"]
    assert "uncommitted" in caveat
    assert "does not by itself reproduce" in caveat


def test_a_clean_working_tree_adds_no_caveat():
    payload = minimal_payload()
    payload["provenance"]["git_dirty"] = False
    assert render_readme.build_context(payload)["provenance_caveat"] == ""


def test_an_unrecorded_tree_state_is_not_reported_as_clean():
    """Runs written before git_dirty existed must not claim to have been clean."""
    payload = minimal_payload()
    payload["provenance"].pop("git_dirty", None)
    caveat = render_readme.build_context(payload)["provenance_caveat"]
    assert "not recorded" in caveat
