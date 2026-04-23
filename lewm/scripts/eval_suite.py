#!/usr/bin/env python3
"""E5 JEPA Eval Suite orchestrator.

Given a run_dir + epoch, runs (or reads) the selected first-wave tests
and emits a unified report.

Usage:
    python lewm/scripts/eval_suite.py \\
        --run-dir /media/hdd1/.../synthetic-all-clean-fs10-aux \\
        --epoch 1 \\
        --cache-dir /home/scad/vsr-tmp/lewm-datasets \\
        [--include-clusters A B C D E]
        [--tests 1 3 5 9 10 11]
        [--skip-tests 3]
        [--probe-dataset <name>]
        [--reference-state-head /.../state_head.pt]
        [--reference-tag <short_name>]

Behavior:
    - Reads <run-dir>/config.yaml for ctx_len, n_preds, action_norm_ref, datasets.
    - For each selected test, checks <run-dir>/epoch_<N>/<subdir>/...json.
        If present, loads it. If absent, runs the sub-script.
    - Writes <run-dir>/epoch_<N>/eval_suite_report.{txt,json} with
        cluster-organized results.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lewm.eval.suite_runner import (
    infer_config_from_run_dir, resolve_requested_tests,
    EvalTarget, load_all_available_results,
    _run_encoder_z_probe, _run_action_response_group,
    _run_rollout_fidelity, _run_cross_head, _run_predicted_z_probe,
)


KIN = ["x", "y", "vx", "vy", "angle", "ang_vel"]


def _section_header(letter: str, title: str) -> list[str]:
    return ["─" * 70, f"Cluster {letter}: {title}", "─" * 70]


def _format_cluster_a(results: dict) -> list[str]:
    lines = _section_header(
        "A", "What does z decode to? (per-frame state decodability)")
    enc = results.get("encoder_z_probe")
    pred = results.get("predicted_z_probe")
    lines.append("Brief: Q1 — does z carry state info on its own (encoder-z)?")
    lines.append("Does predicted-z add decodability beyond what encoder-z already has?")
    lines.append("")
    if enc is None and pred is None:
        lines.append("  No data. Run with --include-clusters A to populate.")
        lines.append("")
        return lines

    if enc and pred:
        lines.append("                 Encoder-z   Predicted-z   Δ (pred − enc)")
        lines.append(f"  Mean R²        {enc['mean_r2']:.3f}       {pred['mean_r2']:.3f}       {pred['mean_r2'] - enc['mean_r2']:+.3f}")
        for dim in KIN:
            e = enc.get("per_dim_r2", {}).get(dim)
            p = pred.get("per_dim_r2", {}).get(dim)
            if e is None or p is None:
                continue
            lines.append(f"  {dim:<14s} {e:.3f}       {p:.3f}       {p - e:+.3f}")
        delta = pred["mean_r2"] - enc["mean_r2"]
        per_dim_deltas = {d: pred["per_dim_r2"][d] - enc["per_dim_r2"][d]
                          for d in KIN
                          if d in pred.get("per_dim_r2", {}) and d in enc.get("per_dim_r2", {})}
        lines.append("")
        if per_dim_deltas and delta > 0.02:
            top_dim, top_delta = max(per_dim_deltas.items(), key=lambda kv: kv[1])
            lines.append(f"Read: predictor ADDS decodability (+{delta:.3f} mean R²), biggest")
            lines.append(f"gain on {top_dim} (+{top_delta:.3f}). Encoder saturates on easy")
            lines.append("dims; predictor fills in the hard ones.")
        elif delta < -0.02:
            lines.append(f"Read: predictor decodability BELOW encoder ({delta:+.3f} mean R²).")
            lines.append("Unusual — predicted-z is dropping information. Investigate.")
        else:
            lines.append(f"Read: predictor decodability ≈ encoder ({delta:+.3f} mean R²).")
            lines.append("Predictor may still add action-conditioning (see Cluster C) but")
            lines.append("doesn't change per-frame state decodability.")
    elif enc:
        lines.append(f"  Encoder-z probe    mean R² = {enc['mean_r2']:.3f}")
        for dim in KIN:
            e = enc.get("per_dim_r2", {}).get(dim)
            if e is not None:
                lines.append(f"    {dim:<14s} {e:.3f}")
        lines.append("")
        lines.append("  Predicted-z probe not available. Re-run Cluster A (or any of B/C/D")
        lines.append("  which have it as a prerequisite) to compare encoder vs predictor.")
    else:
        assert pred is not None
        lines.append(f"  Predicted-z probe  mean R² = {pred['mean_r2']:.3f}")
        for dim in KIN:
            p = pred.get("per_dim_r2", {}).get(dim)
            if p is not None:
                lines.append(f"    {dim:<14s} {p:.3f}")
        lines.append("")
        lines.append("  Encoder-z probe not available — can't compare encoder vs predictor.")
    lines.append("")
    return lines


def _format_cluster_b(results: dict) -> list[str]:
    lines = _section_header(
        "B", "Is the predictor doing real work? (beyond identity pass-through)")
    t9 = results.get("test_9_predictor_modification")
    t10 = results.get("test_10_offline_predloss")
    t11 = results.get("test_11_encoder_forward_coherence")
    lines.append("Brief: Q2 — T9 is how much predictor modifies last-ctx z. T11 is")
    lines.append("natural frame-to-frame drift. T9/T11 ≈ 1 is ambiguous alone; T10")
    lines.append("disambiguates by comparing predictor MSE to identity-baseline MSE.")
    lines.append("")
    if not any([t9, t10, t11]):
        lines.append("  No data. Run with --include-clusters B to populate.")
        lines.append("")
        return lines

    if t9:
        lines.append(f"  T9  predictor modification ‖Δ‖/‖last_ctx‖     {t9['rel_norm']:.3f}")
    if t11:
        lines.append(f"  T11 encoder natural drift ‖Δ‖/‖last_ctx‖      {t11['rel_norm']:.3f}")
    ratio = None
    if t9 and t11:
        ratio = t9["rel_norm"] / max(t11["rel_norm"], 1e-8)
        lines.append(f"  T9 / T11 ratio                                {ratio:.3f}")
    if t10:
        lines.append(f"  T10 predictor MSE(pred, true_next_z)          {t10['mse']:.4f}")
        lines.append(f"  T10 identity-baseline MSE                     {t10['baseline_mse']:.4f}")
        lines.append(f"  Predictor beats identity baseline?            {'✓' if t10['beats_baseline'] else '✗'}")
    if t9 and t11:
        top9 = t9.get("top_dims", [])
        top11 = t11.get("top_dims", [])
        overlap = len(set(top9) & set(top11))
        lines.append("")
        lines.append("  Top-moving z-dims")
        lines.append(f"    T9  (predictor)       {top9}")
        lines.append(f"    T11 (natural drift)   {top11}")
        lines.append(f"    Overlap               {overlap} of {max(len(top9), len(top11), 1)}")

    if t9 and t11 and t10:
        beats = t10["beats_baseline"]
        assert ratio is not None
        if 0.5 <= ratio <= 1.3 and beats:
            verdict = ("predictor moves z on the scale of natural drift AND its output "
                       "matches the true next-frame z better than identity. Doing real "
                       "forward-model work.")
        elif 0.5 <= ratio <= 1.3 and not beats:
            verdict = ("predictor modifies z on natural-drift scale but does NOT beat "
                       "identity. Likely moving in a random direction of correct magnitude.")
        elif ratio < 0.5 and beats:
            verdict = ("predictor under-modifies z (less than natural drift) yet still "
                       "beats identity. Conservative — right direction, too small a step.")
        elif ratio < 0.5 and not beats:
            verdict = ("predictor barely modifies z AND doesn't beat identity. Near "
                       "pass-through / inert (Case 3).")
        elif ratio > 1.3 and beats:
            verdict = ("predictor modifies z MORE than natural drift AND beats identity. "
                       "Substantial transform into a different z-subspace, still accurate.")
        elif ratio > 1.3 and not beats:
            verdict = ("predictor modifies z MORE than natural drift but doesn't beat "
                       "identity. Overshoots — moves z far, in wrong directions.")
        else:
            verdict = "numbers in unusual regime — interpret manually."
        lines.append("")
        lines.append(f"Read: {verdict}")
    else:
        missing = [n for n, v in [("T9", t9), ("T10", t10), ("T11", t11)] if v is None]
        if missing:
            lines.append("")
            lines.append(f"Read: partial data — missing {', '.join(missing)}. Need all three")
            lines.append("for the ratio + baseline interpretation.")
    lines.append("")
    return lines


def _format_cluster_c(results: dict) -> list[str]:
    lines = _section_header(
        "C", "Predictor action-pathway — does the action matter?")
    t1 = results.get("test_1_raw_zdiff")
    t7 = results.get("test_7_fresh_state_head_action_response")
    t8 = results.get("test_8_in_model_aux_head_action_response")
    lines.append("Brief: Q3 — T1: ‖Δz‖/‖z‖ between predictor outputs under different")
    lines.append("actions (pure predictor-side signal). T7/T8: decoded Δ-state via")
    lines.append("fresh vs in-model state_head (agreement = decoder-robust signal).")
    lines.append("")
    if not any([t1, t7, t8]):
        lines.append("  No data. Run with --include-clusters C to populate.")
        lines.append("")
        return lines

    if t1:
        lines.append("  T1 raw z-differential (‖Δz‖/‖z‖ vs no-action baseline):")
        rels = []
        for action_name, data in t1.items():
            if not isinstance(data, dict):
                continue
            rel = data.get("rel_norm")
            if rel is None:
                continue
            rels.append(rel)
            lines.append(f"    {action_name:<18s} {rel:.3f}   top-dims {data.get('top_dims', [])}")
        if rels:
            max_rel = max(rels)
            if max_rel < 0.01:
                v1 = "action-INERT — predictor output identical regardless of action (Case 3)."
            elif max_rel < 0.05:
                v1 = "action-weak — predictor output barely shifts (borderline inert)."
            elif max_rel < 0.3:
                v1 = "action-active — predictor response varies substantially with action."
            else:
                v1 = "action-strong — large predictor response to action swap."
            lines.append("")
            lines.append(f"  Read (T1): {v1}")

            # Scale comparison between T1 (counterfactual action-swap magnitude)
            # and T9 (natural predictor modification under recorded action).
            # NOT a literal decomposition — just a scale reference.
            t9 = results.get("test_9_predictor_modification")
            if t9 and t9.get("rel_norm", 0) > 1e-6:
                ratio = max_rel / t9["rel_norm"]
                lines.append(f"  Scale ref: T1_max / T9 = {max_rel:.3f} / {t9['rel_norm']:.3f} = {ratio:.2f}")
                lines.append(f"  Read (T1 vs T9): counterfactual action-swap magnitude is "
                             f"{'comparable to' if 0.6 <= ratio <= 1.4 else 'smaller than' if ratio < 0.6 else 'larger than'} "
                             "the predictor's natural modification scale, meaning the action channel "
                             "materially contributes to z-change. (Not a literal decomposition of "
                             "predictor work — both measure different z-deltas.)")

        # T1 vs T7 disagreement check: action affects z substantially but decoded
        # kinematics barely move → signal is in kinematically illegible directions.
        if t1 and t7:
            t1_max = max(
                (d.get("rel_norm", 0) for d in t1.values() if isinstance(d, dict)),
                default=0,
            )
            t7_max_abs = max(abs(v) for v in t7.values() if isinstance(v, (int, float)))
            if t1_max >= 0.3 and t7_max_abs < 0.05:
                lines.append("")
                lines.append(f"  Read (T1 vs T7): T1 is large ({t1_max:.2f}) but T7's decoded "
                             f"effect is small (max |Δ-state| = {t7_max_abs:.4f}). Action signal "
                             "exists strongly in z-space but lives largely in directions that "
                             "don't decode to kinematic state. State_head reads little of it.")

    if t7:
        lines.append("")
        lines.append("  T7 decoded Δ-state under action (fresh state_head):")
        for k, v in t7.items():
            if isinstance(v, (int, float)):
                lines.append(f"    {k:<30s} {v:+.4f}")

    if t8:
        lines.append("")
        if t8.get("skipped"):
            lines.append(f"  T8 SKIPPED: {t8.get('reason', 'no in-model state_head')}")
        else:
            lines.append("  T8 decoded Δ-state under action (in-model aux state_head):")
            for k, v in t8.items():
                if k == "decoded" or not isinstance(v, (int, float)):
                    continue
                lines.append(f"    {k:<30s} {v:+.4f}")
            dvy7 = (t7 or {}).get("main_thrust_dvy") if t7 else None
            dvy8 = t8.get("main_thrust_dvy")
            if dvy7 is not None and dvy8 is not None and dvy7 != 0 and dvy8 != 0:
                agree_sign = (dvy7 > 0) == (dvy8 > 0)
                rel_diff = abs(dvy7 - dvy8) / max(abs(dvy7), abs(dvy8))
                if agree_sign and rel_diff < 0.3:
                    v = ("T7 and T8 AGREE on main-thrust Δvy sign + magnitude (<30% gap). "
                         "Action signal is robust across decoder choice.")
                elif agree_sign:
                    v = (f"T7 and T8 agree on sign but differ in magnitude ({rel_diff:.0%} gap). "
                         "Decoders give different confidence levels.")
                else:
                    v = ("T7 and T8 DISAGREE on main-thrust Δvy sign. One decoder is reading "
                         "something the other doesn't — investigate.")
                lines.append("")
                lines.append(f"  Read (T7 vs T8): {v}")
    lines.append("")
    return lines


def _format_cluster_d(results: dict) -> list[str]:
    lines = _section_header(
        "D", "Predictor physics fidelity — physics or memorized correlation?")
    t2 = results.get("test_2_action_magnitude_sweep")
    t3 = results.get("test_3_rollout_fidelity")
    t4 = results.get("test_4_ood_actions")
    lines.append("Brief: Q4 — T2 linearity R² of Δ-state vs action magnitude.")
    lines.append("T3 multi-step rollout MSE (stability). T4 OOD-action response.")
    lines.append("Physics-like → linear + stable + OOD-sensible.")
    lines.append("")
    if not any([t2, t3, t4]):
        lines.append("  No data. Run with --include-clusters D to populate.")
        lines.append("")
        return lines

    if t2:
        lines.append("  T2 action-magnitude sweep — per-magnitude decoded Δ-state:")
        main_rows = t2.get("main", [])
        side_rows = t2.get("side", [])
        def _cell(mean, sem):
            """Format as 'mean ± sem' when sem is available, else just mean.
            A trailing '*' marks rows where |mean| < 2*sem (not distinguishable
            from zero at 2σ)."""
            if sem is None:
                return f"{mean:>+10.4f}            "
            sig = "" if abs(mean) >= 2 * sem else " *"
            return f"{mean:>+10.4f} ± {sem:>7.4f}{sig}"

        if main_rows:
            lines.append("")
            lines.append("    Main thrust sweep (side=0):")
            lines.append(f"      {'mag':>5s}  {'Δvy ± SEM':>21s}  {'Δy ± SEM':>21s}")
            for row in main_rows:
                lines.append(f"      {row['mag']:>5.2f}  "
                             f"{_cell(row['dvy'], row.get('dvy_sem'))}  "
                             f"{_cell(row['dy'], row.get('dy_sem'))}")
        if side_rows:
            lines.append("")
            lines.append("    Side thrust sweep (main=0):")
            lines.append(f"      {'mag':>5s}  {'Δang_vel ± SEM':>21s}  {'Δx ± SEM':>21s}")
            for row in side_rows:
                lines.append(f"      {row['mag']:>5.2f}  "
                             f"{_cell(row['dang_vel'], row.get('dang_vel_sem'))}  "
                             f"{_cell(row['dx'], row.get('dx_sem'))}")
        if main_rows or side_rows:
            lines.append("    (* = |mean| < 2·SEM, not distinguishable from zero at 2σ)")

        lin = t2.get("linearity", {})
        main_r2 = lin.get("main_r2")
        side_r2 = lin.get("side_r2")
        if main_r2 is not None or side_r2 is not None:
            lines.append("")
            lines.append("    Linearity R² (Δ-state vs magnitude):")
            if main_r2 is not None:
                lines.append(f"      Main thrust → Δvy  R²     {main_r2:+.3f}")
            if side_r2 is not None:
                lines.append(f"      Side thrust → Δang_vel R² {side_r2:+.3f}")

        def _verdict(r2, max_abs_decoded):
            # Bucket by linearity, then flag if decoded magnitude is small.
            weak_decoded = (max_abs_decoded is not None and max_abs_decoded < 0.05)
            if r2 > 0.9:
                base = "linear + monotone"
            elif r2 > 0.5:
                base = "monotone but nonlinear"
            elif r2 >= 0:
                base = "weak/inconsistent response"
            else:
                base = "non-monotone (possibly step-function)"
            if weak_decoded and r2 > 0.5:
                return f"{base}, but weak in decoded state (max |Δ| = {max_abs_decoded:.4f})"
            if weak_decoded:
                return f"{base} and weak in decoded state (max |Δ| = {max_abs_decoded:.4f})"
            return base

        if main_r2 is not None and side_r2 is not None:
            main_max_dvy = max((abs(r.get("dvy", 0)) for r in main_rows), default=None)
            side_max_dang = max((abs(r.get("dang_vel", 0)) for r in side_rows), default=None)
            lines.append("")
            lines.append(f"  Read (T2): main = {_verdict(main_r2, main_max_dvy)};")
            lines.append(f"             side = {_verdict(side_r2, side_max_dang)}.")

    if t3:
        lines.append("")
        lines.append("  T3 rollout MSE vs GT per dataset:")
        per_ds = t3.get("per_dataset", {})
        growths = []
        for ds, vals in per_ds.items():
            m5 = vals.get("mse_at_5", float("nan"))
            m10 = vals.get("mse_at_10", float("nan"))
            m20 = vals.get("mse_at_20", float("nan"))
            lines.append(f"    {ds:<50s} @5={m5:.3f} @10={m10:.3f} @20={m20:.3f}")
            if m5 and m5 > 0 and m20:
                growths.append(m20 / m5)
        if growths:
            avg_g = sum(growths) / len(growths)
            if avg_g < 2.0:
                v3 = f"rollout stable — MSE@20/@5 ≈ {avg_g:.1f}× (errors don't compound fast)."
            elif avg_g < 5.0:
                v3 = f"rollout moderate growth — MSE@20/@5 ≈ {avg_g:.1f}× (some compounding)."
            elif avg_g < 20.0:
                v3 = f"rollout diverges — MSE@20/@5 ≈ {avg_g:.1f}× (significant compounding)."
            else:
                v3 = f"rollout EXPLODES — MSE@20/@5 ≈ {avg_g:.1f}× (off-trajectory fast)."
            lines.append("")
            lines.append(f"  Read (T3): {v3}")

    if t4:
        lines.append("")
        lines.append("  T4 OOD-action response (z-norm + decoded Δ-state):")
        lines.append(f"    {'name':<22s} {'‖Δz‖/‖z‖':>10s}  {'Δvy':>8s}  {'Δang_vel':>9s}  {'Δx':>8s}  {'Δy':>8s}")
        for item in t4.get("ood_actions", []):
            name = item.get("name", "?")
            rel = item.get("rel_norm_zdiff")
            d = item.get("decoded_dstate", {})
            if rel is None:
                continue
            lines.append(
                f"    {name:<22s} {rel:>10.3f}  "
                f"{d.get('vy', 0):>+8.4f}  {d.get('ang_vel', 0):>+9.4f}  "
                f"{d.get('x', 0):>+8.4f}  {d.get('y', 0):>+8.4f}"
            )
    lines.append("")
    return lines


def _format_cluster_e(results: dict) -> list[str]:
    lines = _section_header(
        "E", "Decoder portability — is state_head reading universal features?")
    t5 = results.get("test_5_cross_network_state_head")
    lines.append("Brief: Q5 — apply network A's state_head to network B's predicted-z.")
    lines.append("High cross-R² → portable decoder → signal is real, not decoder overfit.")
    lines.append("")
    if not t5:
        lines.append("  No data. Run with --include-clusters E and --reference-state-head PATH.")
        lines.append("")
        return lines

    if isinstance(t5, dict) and "mean_r2" in t5:
        tags = {"(single)": t5}
    elif isinstance(t5, dict):
        tags = t5
    else:
        tags = {}

    for tag, result in tags.items():
        mean = result.get("mean_r2") if isinstance(result, dict) else None
        if mean is None:
            continue
        lines.append(f"  Reference: {tag}")
        lines.append(f"    Cross-R² mean                {mean:.3f}")
        for dim in KIN:
            r2 = result.get("per_dim_r2", {}).get(dim)
            if r2 is not None:
                lines.append(f"    {dim:<14s}               {r2:+.3f}")
        lines.append("")
    return lines


def _format_coverage_and_classification(results: dict) -> list[str]:
    """Two footer sections: (1) spec-coverage status per cluster, (2) Q1-Q4
    case classification synthesized from whatever's present."""
    lines = ["─" * 70, "SPEC COVERAGE", "─" * 70]

    enc = "encoder_z_probe" in results
    pred = "predicted_z_probe" in results
    a_status = "✓ complete" if (enc and pred) else ("⚠ partial" if (enc or pred) else "✗ missing")
    lines.append(f"  [A] encoder vs predicted-z decodability:  {a_status}")

    b_have = {k for k in ("test_9_predictor_modification", "test_10_offline_predloss",
                          "test_11_encoder_forward_coherence") if k in results}
    b_status = "✓ complete" if len(b_have) == 3 else (
        f"⚠ partial ({len(b_have)}/3)" if b_have else "✗ missing")
    lines.append(f"  [B] predictor basic activity (T9/T10/T11): {b_status}")

    t1 = "test_1_raw_zdiff" in results
    t7 = "test_7_fresh_state_head_action_response" in results
    t8_raw = results.get("test_8_in_model_aux_head_action_response")
    t8_ok = bool(t8_raw) and not (isinstance(t8_raw, dict) and t8_raw.get("skipped"))
    t8_inapplicable = isinstance(t8_raw, dict) and t8_raw.get("skipped")
    t6_impl = False  # Test 6 per-stage action propagation not implemented in first wave.
    c_pieces = []
    c_pieces.append("T1 ✓" if t1 else "T1 ✗")
    c_pieces.append("T7 ✓" if t7 else "T7 ✗")
    if t8_ok:
        c_pieces.append("T8 ✓")
    elif t8_inapplicable:
        c_pieces.append("T8 n/a (no aux head)")
    else:
        c_pieces.append("T8 ✗")
    c_pieces.append("T6 not implemented")
    c_core_done = t1 and t7 and (t8_ok or t8_inapplicable)
    c_status = "⚠ partial — " + ", ".join(c_pieces) if not c_core_done or not t6_impl else "✓ complete"
    lines.append(f"  [C] predictor action-pathway:              {c_status}")

    t2 = "test_2_action_magnitude_sweep" in results
    t3 = "test_3_rollout_fidelity" in results
    t4 = "test_4_ood_actions" in results
    d_pieces = [f"T{n} {'✓' if have else '✗'}" for n, have in [(2, t2), (3, t3), (4, t4)]]
    t3_scenarios = len(results["test_3_rollout_fidelity"].get("per_dataset", {})) if t3 else 0
    if t3 and t3_scenarios < 4:
        d_pieces.append(f"T3 ran on {t3_scenarios}/4 scenarios")
    all_tests_present = t2 and t3 and t4
    scenarios_ok = t3_scenarios >= 4
    if all_tests_present and scenarios_ok:
        d_status = "✓ complete"
    elif t2 or t3 or t4:
        d_status = "⚠ partial — " + ", ".join(d_pieces)
    else:
        d_status = "✗ missing"
    lines.append(f"  [D] predictor physics fidelity:            {d_status}")

    t5 = results.get("test_5_cross_network_state_head")
    if t5:
        if isinstance(t5, dict) and "mean_r2" in t5:
            e_status = "✓ complete (1 reference)"
        elif isinstance(t5, dict):
            e_status = f"✓ complete ({len(t5)} reference tag(s))"
        else:
            e_status = "⚠ unexpected shape"
    else:
        e_status = "✗ missing (requires --reference-state-head)"
    lines.append(f"  [E] decoder portability:                   {e_status}")
    lines.append("")

    # Q1–Q4 case classification, synthesized from available data.
    lines.extend(["─" * 70, "CASE CLASSIFICATION (Q1–Q4)", "─" * 70])

    # Q1 — encoder state-structure
    if enc and pred:
        delta = results["predicted_z_probe"]["mean_r2"] - results["encoder_z_probe"]["mean_r2"]
        q1 = f"✓ yes — encoder R² {results['encoder_z_probe']['mean_r2']:.2f}; predictor adds {delta:+.3f}"
    elif enc:
        q1 = f"✓ yes — encoder R² {results['encoder_z_probe']['mean_r2']:.2f} (predicted-z not probed)"
    else:
        q1 = "? — encoder-z probe not run"
    lines.append(f"  Q1 encoder state-structure:         {q1}")

    # Q2 — predictor basic activity
    t10 = results.get("test_10_offline_predloss")
    t9v = results.get("test_9_predictor_modification")
    t11v = results.get("test_11_encoder_forward_coherence")
    if t10 and t9v and t11v:
        ratio = t9v["rel_norm"] / max(t11v["rel_norm"], 1e-8)
        fold = t10["baseline_mse"] / max(t10["mse"], 1e-8)
        if t10["beats_baseline"] and 0.5 <= ratio <= 1.4:
            q2 = f"✓ yes — beats identity {fold:.1f}×, T9/T11 ratio {ratio:.2f}"
        elif t10["beats_baseline"]:
            q2 = f"✓ yes (with caveats) — beats identity {fold:.1f}× but T9/T11 ratio {ratio:.2f}"
        else:
            q2 = f"✗ no — predictor does NOT beat identity baseline"
    else:
        q2 = "? — Cluster B not fully populated"
    lines.append(f"  Q2 predictor basic activity:        {q2}")

    # Q3 — action pathway
    if t1 and t7:
        t1_max = max((d.get("rel_norm", 0) for d in results["test_1_raw_zdiff"].values()
                      if isinstance(d, dict)), default=0)
        t7_max = max(abs(v) for v in results["test_7_fresh_state_head_action_response"].values()
                     if isinstance(v, (int, float)))
        if t1_max >= 0.3 and t7_max >= 0.05:
            q3 = f"✓ yes — T1 {t1_max:.2f} in z + T7 {t7_max:.3f} decoded"
        elif t1_max >= 0.3 and t7_max < 0.05:
            q3 = (f"partial — T1 {t1_max:.2f} (strong in z) but T7 {t7_max:.3f} "
                  "(weak in decoded state)")
        elif t1_max < 0.05:
            q3 = f"✗ no — T1 {t1_max:.2f} (action-inert in z)"
        else:
            q3 = f"weak — T1 {t1_max:.2f}"
    else:
        q3 = "? — Cluster C not fully populated"
    lines.append(f"  Q3 action-pathway:                  {q3}")

    # Q4 — physics fidelity
    q4_bits = []
    if t2:
        lin = results["test_2_action_magnitude_sweep"].get("linearity", {})
        main_r2 = lin.get("main_r2")
        if main_r2 is not None:
            main_max = max((abs(r.get("dvy", 0)) for r in
                            results["test_2_action_magnitude_sweep"].get("main", [])), default=0)
            q4_bits.append(f"T2-main R² {main_r2:+.2f} (max |Δvy| {main_max:.3f})")
    if t3:
        per_ds = results["test_3_rollout_fidelity"].get("per_dataset", {})
        growths = [v.get("mse_at_20", 0) / max(v.get("mse_at_5", 1e-9), 1e-9) for v in per_ds.values()]
        if growths:
            q4_bits.append(f"T3 rollout growth {sum(growths)/len(growths):.1f}×")
    if q4_bits:
        q4 = "partial — " + "; ".join(q4_bits)
    else:
        q4 = "? — Cluster D not populated"
    lines.append(f"  Q4 physics fidelity:                {q4}")
    lines.append("")

    # Provisional one-liner
    label_parts = []
    if "✓ yes" in q1: label_parts.append("encoder carries state info")
    if "✓ yes" in q2: label_parts.append("predictor beats identity")
    if "partial" in q3: label_parts.append("action matters in z but weakly in decoded state")
    elif "✓ yes" in q3: label_parts.append("action-conditioned in both z and decoded state")
    elif "✗ no" in q3: label_parts.append("action-inert")
    if label_parts:
        lines.append(f"  Provisional label: {'; '.join(label_parts)}.")
    lines.append("")

    lines.append("NOTE: this is a partial suite report. Test 6 (per-stage action signal")
    lines.append("propagation) is not yet implemented. T8 requires an aux-loss checkpoint.")
    lines.append("T5 requires --reference-state-head. Missing pieces are listed above.")
    lines.append("")
    return lines


def format_report(target) -> tuple[str, dict]:
    """Render a comprehensive report from whatever's on disk under epoch_dir.

    Returns (text, results_dict). The text is the human-readable report; the
    dict is the aggregated results suitable for the combined JSON sibling.
    """
    results = load_all_available_results(target)
    lines = [
        "=" * 70,
        "E5 JEPA Eval Suite Report",
        "=" * 70,
        f"Checkpoint: {target.ckpt_path()}",
        f"Epoch: {target.epoch}",
        f"Config: ctx_len={target.cfg['ctx_len']}, n_preds={target.cfg['n_preds']}, "
        f"kin_block={target.cfg['kin_block']}, action_norm_ref={target.cfg['action_norm_ref']}",
        "",
    ]
    lines.extend(_format_cluster_a(results))
    lines.extend(_format_cluster_b(results))
    lines.extend(_format_cluster_c(results))
    lines.extend(_format_cluster_d(results))
    lines.extend(_format_cluster_e(results))
    lines.extend(_format_coverage_and_classification(results))
    return "\n".join(lines), results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="Training run dir (contains config.yaml + checkpoints).")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--include-clusters", nargs="+", default=None,
                   choices=["A", "B", "C", "D", "E"])
    p.add_argument("--tests", nargs="+", type=int, default=None,
                   help="Explicit test numbers (e.g., 1 3 5 9). Overrides --include-clusters.")
    p.add_argument("--skip-tests", nargs="+", type=int, default=[])
    p.add_argument("--probe-dataset", default=None,
                   help="Action-response probe dataset. Default: first cfg dataset.")
    p.add_argument("--reference-state-head", default=None,
                   help="Path to state_head.pt from another checkpoint for Test 5.")
    p.add_argument("--reference-tag", default="ref",
                   help="Short name for the reference-head source (used as subdir suffix).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--probe-max-total-frames", type=int, default=900000,
                   help="Total frame budget across all datasets for encoder-z and "
                        "predicted-z probes. Divided evenly (floor) across datasets "
                        "at dispatch. Default 900000 — matches heur-only's single-"
                        "dataset baseline and gives all-clean runs (e.g. 6 datasets) "
                        "150k each. Same default works across networks.")
    p.add_argument("--probe-val-split", type=float, default=0.25,
                   help="Val split fraction for both probes. Default 0.25.")
    p.add_argument("--output-base-dir", default=None,
                   help="Override artifact location. Default: write under "
                        "<run-dir>/epoch_<N>/. When set, writes under "
                        "<output-base-dir>/epoch_<N>/ instead (useful for "
                        "experimental runs that shouldn't pollute the canonical "
                        "run_dir, or when run_dir is read-only).")
    p.add_argument("--report-only", action="store_true",
                   help="Skip all subprocess dispatch — just re-render the "
                        "comprehensive report from whatever JSONs are already on "
                        "disk under <epoch_dir>/. Useful after tweaking the report "
                        "format or when you want to regenerate without re-running.")
    p.add_argument("--rollout-scenarios", nargs="+", default=None,
                   help="Datasets to roll out for Test 3 (Cluster D). Default: "
                        "first 4 of the training config's datasets. Spec calls "
                        "for heuristic + impulse-main + impulse-side + "
                        "ground-stationary — pass explicitly for that set.")
    p.add_argument("--rollout-write-videos", type=int, default=10,
                   help="When Cluster D runs, write schematic-trajectory MP4s for "
                        "the first min(N, n_episodes) rollouts per scenario. "
                        "Default 10. Set 0 to disable.")
    p.add_argument("--force", action="store_true",
                   help="Ignore cached JSONs for the selected scope and re-run "
                        "every selected test from scratch. Other tests' cached "
                        "JSONs are preserved (AR merge-at-write rules still "
                        "apply for unaffected keys). Useful after changing the "
                        "underlying script (new metric fields, format fixes, etc).")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = infer_config_from_run_dir(run_dir)
    output_base_dir = Path(args.output_base_dir) if args.output_base_dir else None
    target = EvalTarget(run_dir=run_dir, epoch=args.epoch, cfg=cfg,
                        output_base_dir=output_base_dir)
    target.epoch_dir().mkdir(parents=True, exist_ok=True)
    if output_base_dir is not None:
        print(f"Output base dir: {output_base_dir} (run_dir unchanged)")

    if args.report_only:
        selected = []
        print("Report-only mode: skipping all subprocess dispatch.")
    else:
        selected = resolve_requested_tests(args.include_clusters, args.tests, args.skip_tests)
        print(f"Selected tests: {selected}")

    probe_dataset = args.probe_dataset or cfg["datasets"][0]
    cache_dir = Path(args.cache_dir)

    # State_head path for action-response + rollout
    probe_subdir = target.epoch_dir() / "predicted_z"
    state_head_path = probe_subdir / "state_head_normZ.pt"

    # Determine what needs to run
    needs_rollout = "test_3_rollout_fidelity" in selected
    needs_encoder_z = "encoder_z_probe" in selected
    needs_predicted_z = "predicted_z_probe" in selected
    needs_cross_head = "test_5_cross_network_state_head" in selected

    frames_per_dataset = args.probe_max_total_frames // max(1, len(cfg["datasets"]))
    print(f"Probe budget: {args.probe_max_total_frames} total / "
          f"{len(cfg['datasets'])} datasets = {frames_per_dataset} frames each")

    if needs_encoder_z:
        if args.force or not (target.epoch_dir() / "encoder_z" / "r2_report.json").exists():
            _run_encoder_z_probe(target, cache_dir,
                                 max_frames_per_dataset=frames_per_dataset,
                                 val_split=args.probe_val_split)

    # Granular AR check: does the existing JSON already contain every AR-family
    # test we want? If any selected AR test is missing from the file, re-run
    # (test_action_response.py merges new results into the existing JSON, so
    # B-keys from a prior run are preserved when C-keys are added now).
    ar_json_path = target.epoch_dir() / "action_response" / "action_response_report_normZ.json"
    ar_family = {t for t in selected if t in (
        "test_1_raw_zdiff", "test_2_action_magnitude_sweep", "test_4_ood_actions",
        "test_7_fresh_state_head_action_response",
        "test_8_in_model_aux_head_action_response",
        "test_9_predictor_modification", "test_10_offline_predloss",
        "test_11_encoder_forward_coherence",
    )}
    existing_ar_keys: set[str] = set()
    if ar_json_path.exists() and not args.force:
        existing_ar_keys = set(json.loads(ar_json_path.read_text()).get("results", {}).keys())
    missing_ar = ar_family - existing_ar_keys

    rf_json_exists = (target.epoch_dir() / "rollout_fidelity" / "rollout_fidelity.json").exists()
    pz_json_exists = (target.epoch_dir() / "predicted_z" / "r2_report_normZ.json").exists()
    ar_needs_run = bool(missing_ar)
    # Rollout re-runs if JSON missing, OR if videos requested but the video
    # dir is empty (JSON may pre-date video support). --force overrides.
    rf_video_dir = target.epoch_dir() / "rollout_fidelity" / "videos"
    rf_videos_missing = (args.rollout_write_videos > 0 and
                         (not rf_video_dir.exists() or not any(rf_video_dir.iterdir())))
    rf_needs_run = needs_rollout and (args.force or not rf_json_exists or rf_videos_missing)
    pz_needs_run = (
        (needs_predicted_z and (args.force or not pz_json_exists))
        or ((ar_needs_run or rf_needs_run) and not state_head_path.exists())
    )

    if ar_needs_run:
        print(f"Action-response needs keys: {sorted(missing_ar)} (existing: {sorted(existing_ar_keys)})")

    if pz_needs_run:
        _run_predicted_z_probe(target, cache_dir,
                               max_frames_per_dataset=frames_per_dataset,
                               val_split=args.probe_val_split)

    if ar_needs_run or rf_needs_run:
        if ar_needs_run:
            _run_action_response_group(target, cache_dir, state_head_path, probe_dataset,
                                       tests_requested=missing_ar)

        if rf_needs_run:
            _run_rollout_fidelity(target, cache_dir, state_head_path,
                                  write_videos=args.rollout_write_videos,
                                  scenarios=args.rollout_scenarios)

    if needs_cross_head:
        cross_head_json = target.epoch_dir() / f"cross_head_{args.reference_tag}" / "cross_head.json"
        if cross_head_json.exists() and not args.force:
            pass
        elif args.reference_state_head:
            predicted_z_cache = next(probe_subdir.glob("predicted_z_aligned_*.npz"))
            _run_cross_head(target, cache_dir, Path(args.reference_state_head),
                            args.reference_tag, predicted_z_cache)
        else:
            print("Cluster E SKIPPED: --reference-state-head not supplied. To run "
                  "Test 5 cross-network state-head, re-invoke with "
                  "--reference-state-head PATH --reference-tag <short_name>.")

    # Emit comprehensive report — reads everything available on disk
    # regardless of --selected (selected only drives what gets RUN above).
    txt, results = format_report(target)
    out_txt = target.epoch_dir() / "eval_suite_report.txt"
    out_json = target.epoch_dir() / "eval_suite_report.json"
    out_txt.write_text(txt)
    combined = {
        "schema_version": "e5-eval-suite-first-wave-v1",
        "model": str(target.ckpt_path()),
        "epoch": target.epoch,
        "cfg": cfg,
        "results": results,
    }
    out_json.write_text(json.dumps(combined, indent=2))
    print(txt)
    print(f"\nWrote {out_txt}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
