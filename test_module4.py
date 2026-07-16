"""
test_module4.py — Unit + Integration Tests for Module 4 (Drift Attribution)

Tests cover:
  1-2. Coordinate helpers & zero-forcing trajectory
  3.   Forward vs reverse trajectory divergence
  4.   Monte Carlo determinism / stochasticity
  5-6. Bhattacharyya coefficient identity & separation
  7.   Composite score boundary values
  8.   AIS score normalisation range
  9.   Full integration test with real Module 3 output
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Ensure project root is on the path so imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.drift_model import (
    meters_per_degree,
    simulate_trajectory,
    monte_carlo_endpoints,
    fit_gaussian_2d,
    bhattacharyya_coefficient,
    compute_s_drift,
    compute_composite_score,
    normalize_ais_scores,
    compute_s_temporal,
    compute_s_morphology,
    mock_env_provider,
    run_drift_attribution,
)
from src.report_export import export_attribution_report


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═══════════════════════════════════════════════════════════════

def test_meters_per_degree():
    """Test 1: meters_per_degree returns sensible values."""
    print("Test 1: meters_per_degree ...")

    m_lat_eq, m_lon_eq = meters_per_degree(0)
    assert abs(m_lat_eq - 111320.0) < 1.0, "m_per_deg_lat at equator should be ~111320"
    assert abs(m_lon_eq - 111320.0) < 1.0, "m_per_deg_lon at equator should be ~111320"

    # At non-zero latitude, m_per_deg_lon < m_per_deg_lat because cos(lat) < 1
    m_lat_45, m_lon_45 = meters_per_degree(45)
    assert m_lon_45 < m_lat_45, (
        f"At 45° lat, m_per_deg_lon ({m_lon_45:.1f}) should be < "
        f"m_per_deg_lat ({m_lat_45:.1f})"
    )

    m_lat_60, m_lon_60 = meters_per_degree(60)
    assert m_lon_60 < m_lon_45, (
        "m_per_deg_lon should decrease as latitude increases"
    )

    print("  ✅ PASSED\n")


def test_zero_forcing_trajectory():
    """Test 2: Zero wind/current → no displacement."""
    print("Test 2: simulate_trajectory with zero forcing ...")

    def zero_env(lat, lon, time):
        return {
            "wind_u": 0.0, "wind_v": 0.0,
            "current_u": 0.0, "current_v": 0.0,
            "stokes_u": 0.0, "stokes_v": 0.0,
        }

    start_lat, start_lon = 29.2, -94.8
    t0 = pd.Timestamp("2022-08-03 08:00:00")
    t1 = pd.Timestamp("2022-08-03 14:00:00")

    end_lat, end_lon = simulate_trajectory(
        start_lat, start_lon, t0, t1,
        env_provider=zero_env,
    )
    assert abs(end_lat - start_lat) < 1e-10, (
        f"Expected no lat change, got delta={end_lat - start_lat}"
    )
    assert abs(end_lon - start_lon) < 1e-10, (
        f"Expected no lon change, got delta={end_lon - start_lon}"
    )
    print("  ✅ PASSED\n")


def test_forward_vs_reverse():
    """Test 3: Forward and reverse trajectories produce opposite displacement."""
    print("Test 3: forward vs reverse trajectory ...")

    t0 = pd.Timestamp("2022-08-03 08:00:00")
    t1 = pd.Timestamp("2022-08-03 14:00:00")
    start_lat, start_lon = 29.2, -94.8

    fwd_lat, fwd_lon = simulate_trajectory(
        start_lat, start_lon, t0, t1,
        env_provider=mock_env_provider,
        wind_noise_sigma=0.0,
        reverse=False,
    )
    rev_lat, rev_lon = simulate_trajectory(
        start_lat, start_lon, t0, t1,
        env_provider=mock_env_provider,
        wind_noise_sigma=0.0,
        reverse=True,
    )

    fwd_dlat = fwd_lat - start_lat
    fwd_dlon = fwd_lon - start_lon
    rev_dlat = rev_lat - start_lat
    rev_dlon = rev_lon - start_lon

    # Displacements should be in roughly opposite directions
    assert fwd_dlat * rev_dlat <= 0 or abs(fwd_dlat) < 1e-9, (
        f"Lat displacements should be opposite: fwd={fwd_dlat:.6f}, rev={rev_dlat:.6f}"
    )
    assert fwd_dlon * rev_dlon <= 0 or abs(fwd_dlon) < 1e-9, (
        f"Lon displacements should be opposite: fwd={fwd_dlon:.6f}, rev={rev_dlon:.6f}"
    )

    # They should not be identical (i.e., there IS displacement)
    total_fwd = abs(fwd_dlat) + abs(fwd_dlon)
    assert total_fwd > 1e-6, "Forward trajectory should have non-zero displacement"

    print(f"  Forward  Δ(lat,lon) = ({fwd_dlat:+.6f}, {fwd_dlon:+.6f})")
    print(f"  Reverse  Δ(lat,lon) = ({rev_dlat:+.6f}, {rev_dlon:+.6f})")
    print("  ✅ PASSED\n")


def test_monte_carlo_determinism():
    """Test 4: sigma=0 → identical rows; sigma>0 → distinct rows."""
    print("Test 4: Monte Carlo determinism / stochasticity ...")

    t0 = pd.Timestamp("2022-08-03 08:00:00")
    t1 = pd.Timestamp("2022-08-03 14:00:00")
    n = 20

    # No noise → all identical
    pts_zero = monte_carlo_endpoints(
        29.2, -94.8, t0, t1,
        n_members=n, wind_noise_sigma=0.0,
    )
    unique_rows = len(np.unique(pts_zero, axis=0))
    assert unique_rows == 1, (
        f"With sigma=0, expected 1 unique row, got {unique_rows}"
    )

    # With noise → distinct
    np.random.seed(42)
    pts_noisy = monte_carlo_endpoints(
        29.2, -94.8, t0, t1,
        n_members=n, wind_noise_sigma=0.5,
    )
    unique_rows_noisy = len(np.unique(pts_noisy, axis=0))
    assert unique_rows_noisy > 1, (
        f"With sigma=0.5, expected >1 unique row, got {unique_rows_noisy}"
    )

    print(f"  sigma=0: {len(np.unique(pts_zero, axis=0))} unique endpoint(s)")
    print(f"  sigma=0.5: {unique_rows_noisy} unique endpoints out of {n}")
    print("  ✅ PASSED\n")


def test_bhattacharyya_identity():
    """Test 5: BC of a distribution against itself ≈ 1.0."""
    print("Test 5: Bhattacharyya self-identity ...")

    mean = np.array([29.2, -94.8])
    cov = np.array([[0.001, 0.0], [0.0, 0.001]])

    bc = bhattacharyya_coefficient(mean, cov, mean, cov)
    assert abs(bc - 1.0) < 1e-4, f"Self-BC should be ~1.0, got {bc:.6f}"

    print(f"  BC(self, self) = {bc:.6f}")
    print("  ✅ PASSED\n")


def test_bhattacharyya_separation():
    """Test 6: Two far-apart, tight distributions → BC ≈ 0."""
    print("Test 6: Bhattacharyya separation ...")

    mean1 = np.array([29.2, -94.8])
    mean2 = np.array([40.0, -80.0])   # ~1500 km away
    cov = np.array([[1e-6, 0.0], [0.0, 1e-6]])  # very tight

    bc = bhattacharyya_coefficient(mean1, cov, mean2, cov)
    assert bc < 0.01, f"Far-apart distributions should have BC ~0, got {bc:.6f}"

    print(f"  BC(NYC-area, Gulf) = {bc:.6e}")
    print("  ✅ PASSED\n")


def test_composite_score_bounds():
    """Test 7: Composite score at boundary values."""
    print("Test 7: composite_score boundary values ...")

    c_all_one = compute_composite_score(1.0, 1.0, 1.0, 1.0)
    assert abs(c_all_one - 1.0) < 1e-9, f"All 1.0 should give C=1.0, got {c_all_one}"

    c_all_zero = compute_composite_score(0.0, 0.0, 0.0, 0.0)
    assert abs(c_all_zero) < 1e-9, f"All 0.0 should give C=0.0, got {c_all_zero}"

    # Weighted check: 0.4*0.5 + 0.3*0.5 + 0.2*0.5 + 0.1*0.5 = 0.5
    c_half = compute_composite_score(0.5, 0.5, 0.5, 0.5)
    assert abs(c_half - 0.5) < 1e-9, f"All 0.5 should give C=0.5, got {c_half}"

    print(f"  C(all 1.0) = {c_all_one:.4f}")
    print(f"  C(all 0.0) = {c_all_zero:.4f}")
    print(f"  C(all 0.5) = {c_half:.4f}")
    print("  ✅ PASSED\n")


def test_normalize_ais_scores():
    """Test 8: Normalized S_AIS values are within [0, 1]."""
    print("Test 8: normalize_ais_scores range ...")

    df = pd.DataFrame({
        "MMSI": [111, 222, 333, 444],
        "anomaly_score": [0.3, 0.7, 0.1, 0.9],
    })
    result = normalize_ais_scores(df)

    assert "S_AIS" in result.columns, "S_AIS column missing"
    assert result["S_AIS"].min() >= 0.0, f"S_AIS min < 0: {result['S_AIS'].min()}"
    assert result["S_AIS"].max() <= 1.0, f"S_AIS max > 1: {result['S_AIS'].max()}"

    # Edge case: all identical scores
    df_same = pd.DataFrame({
        "MMSI": [111, 222],
        "anomaly_score": [0.5, 0.5],
    })
    result_same = normalize_ais_scores(df_same)
    assert all(result_same["S_AIS"] == 0.5), "Identical scores should all map to 0.5"

    print(f"  S_AIS values: {result['S_AIS'].tolist()}")
    print(f"  Identical-score edge case: {result_same['S_AIS'].tolist()}")
    print("  ✅ PASSED\n")


def test_s_temporal():
    """Test additional: compute_s_temporal night vs day."""
    print("Test (extra): compute_s_temporal ...")

    assert compute_s_temporal("2022-08-03 02:00:00") == 1.0, "2am should be night"
    assert compute_s_temporal("2022-08-03 22:00:00") == 1.0, "10pm should be night"
    assert compute_s_temporal("2022-08-03 12:00:00") == 0.5, "12pm should be day"
    assert compute_s_temporal("2022-08-03 08:00:00") == 0.5, "8am should be day"

    print("  ✅ PASSED\n")


def test_s_morphology():
    """Test additional: compute_s_morphology."""
    print("Test (extra): compute_s_morphology ...")

    # Same heading → max similarity
    assert abs(compute_s_morphology(90, 90) - 1.0) < 1e-9
    # Opposite heading → min similarity
    assert abs(compute_s_morphology(0, 180) - 0.0) < 1e-9
    # None inputs → neutral 0.5
    assert compute_s_morphology(None, 90) == 0.5
    assert compute_s_morphology(90, None) == 0.5
    assert compute_s_morphology(None, None) == 0.5

    print("  ✅ PASSED\n")


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TEST — Full Module 3 → Module 4 Pipeline
# ═══════════════════════════════════════════════════════════════

def run_integration_test():
    """
    Load real AIS data, run Module 3 to get Tier-1 candidates,
    then run Module 4 drift attribution and export a JSON report.
    """
    print("=" * 70)
    print("  INTEGRATION TEST — Module 3 → Module 4 Pipeline")
    print("=" * 70 + "\n")

    # ── Locate AIS data file ─────────────────────────────────────────────────
    ais_candidates = [
        "data/ais_gulf_test.csv",
        "data/ais/ais_gulf_2022_08_03.csv",
    ]
    ais_path = None
    for path in ais_candidates:
        if os.path.exists(path):
            ais_path = path
            break

    if ais_path is None:
        print("  ⚠ AIS data file not found. Skipping integration test.")
        print("    Looked for:", ais_candidates)
        return

    print(f"  Using AIS data: {ais_path}\n")

    # ── Spill event parameters ───────────────────────────────────────────────
    spill_lat = 29.2
    spill_lon = -94.8
    sar_time_str = "2022-08-03 14:00:00"
    discharge_time_str = "2022-08-03 08:00:00"

    # ── Run Module 3 ─────────────────────────────────────────────────────────
    from src.ais_filter import (
        load_ais, filter_vessel_types, clean_trajectories,
        compute_vessel_features, score_anomalies,
    )

    print("─── Module 3: AIS Pipeline ───")
    df = load_ais(ais_path, spill_lat, spill_lon, sar_time_str)
    df = filter_vessel_types(df)
    df_clean = clean_trajectories(df)
    features_df = compute_vessel_features(df_clean, spill_lat, spill_lon)
    candidates = score_anomalies(features_df)

    n_tier1 = candidates["tier1_flag"].sum() if len(candidates) > 0 else 0
    print(f"  → {len(candidates)} vessels scored, {n_tier1} Tier-1\n")

    if n_tier1 == 0:
        print("  ⚠ No Tier-1 vessels — cannot run drift attribution.")
        return

    # ── Run Module 4 ─────────────────────────────────────────────────────────
    print("─── Module 4: Drift Attribution ───")
    np.random.seed(42)  # reproducibility

    results_df = run_drift_attribution(
        candidates_df=candidates,
        ais_positions_df=df_clean,
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        sar_time_str=sar_time_str,
        discharge_time_str=discharge_time_str,
        n_members=100,
    )

    # ── Display results ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  DRIFT ATTRIBUTION RESULTS (sorted by composite score C)")
    print("═" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results_df.to_string(index=False))
    print()

    # ── Assertions ───────────────────────────────────────────────────────────
    assert len(results_df) > 0, "Results DataFrame should not be empty"
    assert (results_df["C"] > 0).any(), "At least one vessel should have C > 0"

    # Check descending sort
    c_values = results_df["C"].values
    for i in range(len(c_values) - 1):
        assert c_values[i] >= c_values[i + 1], (
            f"Results not sorted descending by C: index {i} ({c_values[i]}) "
            f"< index {i+1} ({c_values[i+1]})"
        )

    print("  ✅ Integration assertions PASSED\n")

    # ── Export FR-7 Report ───────────────────────────────────────────────────
    print("─── FR-7: Exporting attribution report ───")

    spill_geometry = {
        "centroid_lat": spill_lat,
        "centroid_lon": spill_lon,
        "area_km2": 12.5,        # placeholder for demo
        "elongation": 3.2,       # placeholder for demo
        "sar_acquisition_time": sar_time_str,
    }

    report_path = "outputs/reports/test_attribution_report.json"
    export_attribution_report(spill_geometry, results_df, report_path)

    # ── Verify round-trip ────────────────────────────────────────────────────
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"\n  Report contents ({report_path}):")
    print(json.dumps(report, indent=2))

    assert "spill_geometry" in report
    assert "candidates" in report
    assert len(report["candidates"]) == len(results_df)
    assert report["candidates"][0]["composite_confidence_score"] >= \
           report["candidates"][-1]["composite_confidence_score"]

    print("\n  ✅ FR-7 report round-trip PASSED\n")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MODULE 4 — UNIT TESTS")
    print("=" * 70 + "\n")

    test_meters_per_degree()
    test_zero_forcing_trajectory()
    test_forward_vs_reverse()
    test_monte_carlo_determinism()
    test_bhattacharyya_identity()
    test_bhattacharyya_separation()
    test_composite_score_bounds()
    test_normalize_ais_scores()
    test_s_temporal()
    test_s_morphology()

    print("═" * 70)
    print("  ALL UNIT TESTS PASSED ✅")
    print("═" * 70 + "\n")

    run_integration_test()

    print("=" * 70)
    print("  ALL TESTS COMPLETE ✅")
    print("=" * 70)
