"""
Module 4 — Bidirectional Lagrangian Drift Attribution

Simulates oil slick drift trajectories (forward from vessel, backward from
spill detection) using Euler-Maruyama integration with Monte Carlo ensemble
perturbation.  Computes a composite confidence score C for each Tier-1
suspect vessel identified by Module 3.

Default environmental forcing uses a mock provider (constant wind/current)
so the full pipeline runs end-to-end without any API keys.  A real
ERA5/CMEMS provider can be swapped in via the `env_provider` parameter
without changing any call sites.
"""

import warnings
import math
import numpy as np
import pandas as pd
from datetime import timedelta


# ── 1. Coordinate conversion helpers ─────────────────────────────────────────

def meters_per_degree(lat):
    """Return (meters_per_deg_lat, meters_per_deg_lon) at given latitude."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return m_per_deg_lat, m_per_deg_lon


# ── 2. Mock environmental data provider ──────────────────────────────────────

def mock_env_provider(lat, lon, time):
    """
    Returns a dict of forcing field values at a given position/time.

    Default mock returns constants so the pipeline runs without any API keys.
    Real ERA5/CMEMS providers must return the same dict shape and can be
    passed into simulate_trajectory() via the env_provider parameter.

    Returns:
        dict with keys: wind_u, wind_v, current_u, current_v,
                        stokes_u, stokes_v   (all in m/s)
    """
    return {
        "wind_u": 5.0,       # eastward wind, m/s
        "wind_v": 0.0,       # northward wind, m/s
        "current_u": 0.0,    # eastward current, m/s
        "current_v": 0.1,    # northward current, m/s
        "stokes_u": 0.0,     # eastward Stokes drift, m/s
        "stokes_v": 0.0,     # northward Stokes drift, m/s
    }


# ── 3. Core trajectory integrator (Euler-Maruyama) ───────────────────────────

def simulate_trajectory(
    start_lat,
    start_lon,
    start_time,
    end_time,
    env_provider=mock_env_provider,
    leeway_coefficient=0.035,
    wind_noise_sigma=0.0,
    dt_hours=1.0,
    reverse=False,
):
    """
    Integrate a single trajectory from start_time to end_time using
    Euler-Maruyama stepping.

    Convention for reverse integration:
        When reverse=True the velocity vector is NEGATED at every step,
        effectively running the dynamics backward in time.  The time axis
        always advances from start_time toward end_time regardless; the
        ``reverse`` flag only affects the *spatial* step direction.  This
        avoids the subtle bug where both a negative time delta and the
        reverse flag would cancel each other out.

    Args:
        start_lat, start_lon: initial position (degrees).
        start_time, end_time: pd.Timestamp objects.
        env_provider: callable(lat, lon, time) → dict of forcing fields.
        leeway_coefficient: fraction of wind speed transferred to slick.
        wind_noise_sigma: std-dev of Gaussian perturbation on wind (m/s).
                          Set > 0 only for Monte Carlo ensemble members.
        dt_hours: integration time-step size in hours.
        reverse: if True, negate velocity at every step (backward advection).

    Returns:
        (final_lat, final_lon) — position after integration.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    total_hours = abs((end_time - start_time).total_seconds()) / 3600.0
    n_steps = max(1, math.ceil(total_hours / dt_hours))
    dt_seconds = dt_hours * 3600.0

    # Determine temporal direction (always advance toward end_time)
    time_sign = 1.0 if end_time >= start_time else -1.0

    current_lat = float(start_lat)
    current_lon = float(start_lon)
    current_time = start_time

    for _ in range(n_steps):
        env = env_provider(current_lat, current_lon, current_time)

        # Optionally perturb wind
        if wind_noise_sigma > 0:
            wu = env["wind_u"] + np.random.normal(0, wind_noise_sigma)
            wv = env["wind_v"] + np.random.normal(0, wind_noise_sigma)
        else:
            wu = env["wind_u"]
            wv = env["wind_v"]

        # Composite velocity: leeway-adjusted wind + ocean current + Stokes
        velocity_u = wu * leeway_coefficient + env["current_u"] + env["stokes_u"]
        velocity_v = wv * leeway_coefficient + env["current_v"] + env["stokes_v"]

        # Negate velocity for backward (reverse) integration
        if reverse:
            velocity_u = -velocity_u
            velocity_v = -velocity_v

        m_lat, m_lon = meters_per_degree(current_lat)

        # Guard against degenerate m_lon near the poles
        if m_lon < 1.0:
            m_lon = 1.0

        dlat = (velocity_v * dt_seconds) / m_lat
        dlon = (velocity_u * dt_seconds) / m_lon

        current_lat += dlat
        current_lon += dlon
        current_time += timedelta(hours=time_sign * dt_hours)

    return current_lat, current_lon


# ── 4. Monte Carlo ensemble wrapper ──────────────────────────────────────────

def monte_carlo_endpoints(
    start_lat,
    start_lon,
    start_time,
    end_time,
    env_provider=mock_env_provider,
    leeway_coefficient=0.035,
    n_members=100,         # Phase-I report specifies N=500 for final numbers;
                            # 100 is default for fast laptop iteration.
    wind_noise_sigma=0.5,  # m/s, per report spec
    reverse=False,
):
    """
    Run simulate_trajectory() n_members times with wind_noise_sigma applied,
    collecting the ensemble of endpoint positions.

    Returns:
        np.ndarray of shape (n_members, 2) — columns are [lat, lon].
    """
    endpoints = np.empty((n_members, 2), dtype=np.float64)
    for i in range(n_members):
        lat, lon = simulate_trajectory(
            start_lat, start_lon, start_time, end_time,
            env_provider=env_provider,
            leeway_coefficient=leeway_coefficient,
            wind_noise_sigma=wind_noise_sigma,
            reverse=reverse,
        )
        endpoints[i, 0] = lat
        endpoints[i, 1] = lon
    return endpoints


# ── 5. Bhattacharyya coefficient (2D Gaussian) ───────────────────────────────

def fit_gaussian_2d(points, default_cov=None):
    """
    Fit a 2D Gaussian to an (N, 2) array of [lat, lon] points.

    Args:
        points: np.ndarray of shape (N, 2).
        default_cov: covariance to use when N <= 1.  Defaults to
                     np.eye(2) * (0.005)**2  (~500 m positional uncertainty).

    Returns:
        (mean, cov) — mean shape (2,), cov shape (2, 2).
    """
    if default_cov is None:
        default_cov = np.eye(2) * (0.005 ** 2)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    mean = np.mean(points, axis=0)

    if len(points) <= 1:
        return mean, default_cov.copy()

    cov = np.cov(points, rowvar=False)
    # If cov collapses to scalar (shouldn't with 2-col input), reshape
    if cov.ndim == 0:
        cov = np.eye(2) * float(cov)
    return mean, cov


def bhattacharyya_coefficient(mean1, cov1, mean2, cov2):
    """
    Compute the Bhattacharyya coefficient between two 2D Gaussians.

    D_B = (1/8) (μ1-μ2)^T Σ_avg^{-1} (μ1-μ2)
          + (1/2) ln( det(Σ_avg) / sqrt(det(Σ1) det(Σ2)) )

    BC  = exp(-D_B)

    Epsilon regularisation is added to covariance diagonals before
    inversion to avoid singular-matrix errors when ensemble spread
    is near zero.

    Returns:
        float in range (0, 1], where 1 = identical distributions.
    """
    eps = 1e-10
    mean1 = np.asarray(mean1, dtype=np.float64)
    mean2 = np.asarray(mean2, dtype=np.float64)
    cov1 = np.asarray(cov1, dtype=np.float64) + np.eye(2) * eps
    cov2 = np.asarray(cov2, dtype=np.float64) + np.eye(2) * eps

    cov_avg = (cov1 + cov2) / 2.0

    diff = mean1 - mean2

    # Mahalanobis-like term
    try:
        inv_cov_avg = np.linalg.inv(cov_avg)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse
        inv_cov_avg = np.linalg.pinv(cov_avg)

    term1 = (1.0 / 8.0) * diff @ inv_cov_avg @ diff

    det_avg = np.linalg.det(cov_avg)
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)

    # Protect against log of zero / negative det (numerical noise)
    det_avg = max(det_avg, eps)
    det1 = max(det1, eps)
    det2 = max(det2, eps)

    term2 = 0.5 * math.log(det_avg / math.sqrt(det1 * det2))

    d_b = term1 + term2
    bc = math.exp(-d_b)

    # Clamp to valid range
    return float(np.clip(bc, 0.0, 1.0))


# ── 6. Bidirectional S_drift computation ─────────────────────────────────────

def compute_s_drift(
    vessel_lat,
    vessel_lon,
    discharge_time,
    spill_lat,
    spill_lon,
    sar_time,
    env_provider=mock_env_provider,
    leeway_coefficient=0.035,
    n_members=100,
):
    """
    Compute S_drift as the AVERAGE of forward and backward Bhattacharyya
    coefficient checks.

    Forward:  ensemble from vessel position → SAR time  vs  observed spill.
    Backward: ensemble from spill position  → discharge time  vs  vessel.

    Returns:
        dict with keys: s_drift, forward_score, backward_score,
        forward_ensemble (Nx2 array), backward_ensemble (Nx2 array).
    """
    discharge_time = pd.to_datetime(discharge_time)
    sar_time = pd.to_datetime(sar_time)

    # ── Forward: vessel → spill ──────────────────────────────────────────────
    fwd_ensemble = monte_carlo_endpoints(
        vessel_lat, vessel_lon,
        discharge_time, sar_time,
        env_provider=env_provider,
        leeway_coefficient=leeway_coefficient,
        n_members=n_members,
        reverse=False,
    )
    fwd_mean, fwd_cov = fit_gaussian_2d(fwd_ensemble)
    spill_mean, spill_cov = fit_gaussian_2d(
        np.array([[spill_lat, spill_lon]])
    )
    forward_score = bhattacharyya_coefficient(fwd_mean, fwd_cov, spill_mean, spill_cov)

    # ── Backward: spill → vessel ─────────────────────────────────────────────
    bwd_ensemble = monte_carlo_endpoints(
        spill_lat, spill_lon,
        sar_time, discharge_time,
        env_provider=env_provider,
        leeway_coefficient=leeway_coefficient,
        n_members=n_members,
        reverse=True,
    )
    bwd_mean, bwd_cov = fit_gaussian_2d(bwd_ensemble)
    vessel_mean, vessel_cov = fit_gaussian_2d(
        np.array([[vessel_lat, vessel_lon]])
    )
    backward_score = bhattacharyya_coefficient(bwd_mean, bwd_cov, vessel_mean, vessel_cov)

    s_drift = (forward_score + backward_score) / 2.0

    return {
        "s_drift": s_drift,
        "forward_score": forward_score,
        "backward_score": backward_score,
        "forward_ensemble": fwd_ensemble,
        "backward_ensemble": bwd_ensemble,
    }


# ── 7. Supporting score components ───────────────────────────────────────────

def normalize_ais_scores(candidates_df):
    """
    Min-max normalize the 'anomaly_score' column from Module 3's output to
    a 0-1 range across the full candidate set, added as new column 'S_AIS'.

    If all scores are identical (edge case), set S_AIS = 0.5 for all rows.

    Returns:
        DataFrame with the new 'S_AIS' column added (original is not modified).
    """
    df = candidates_df.copy()
    scores = df["anomaly_score"]
    s_min, s_max = scores.min(), scores.max()

    if s_max - s_min < 1e-12:
        df["S_AIS"] = 0.5
    else:
        df["S_AIS"] = (scores - s_min) / (s_max - s_min)

    return df


def compute_s_temporal(discharge_time):
    """
    Returns 1.0 if discharge_time's UTC hour is in [20, 24) or [0, 6),
    else 0.5.

    Matches the is_night feature convention already used in src/features.py.
    """
    dt = pd.to_datetime(discharge_time)
    hour = dt.hour
    if hour >= 20 or hour < 6:
        return 1.0
    return 0.5


def compute_s_morphology(vessel_heading_deg=None, spill_elongation_angle_deg=None):
    """
    If both vessel_heading_deg and spill_elongation_angle_deg are provided,
    compute cosine similarity between them:
        angle_diff = vessel_heading_deg - spill_elongation_angle_deg
        return (cos(angle_diff) + 1) / 2     # rescale -1..1 → 0..1

    If either is None, return 0.5 as a neutral placeholder.
    (Full morphology matching requires SAR mask orientation analysis
    which is a future extension.)
    """
    if vessel_heading_deg is None or spill_elongation_angle_deg is None:
        return 0.5
    angle_diff = vessel_heading_deg - spill_elongation_angle_deg
    return (math.cos(math.radians(angle_diff)) + 1.0) / 2.0


# ── 8. Composite confidence score ────────────────────────────────────────────

def compute_composite_score(s_drift, s_ais, s_morphology, s_temporal):
    """
    C = 0.4 * s_drift + 0.3 * s_ais + 0.2 * s_morphology + 0.1 * s_temporal

    Returns:
        float in [0, 1].
    """
    return 0.4 * s_drift + 0.3 * s_ais + 0.2 * s_morphology + 0.1 * s_temporal


# ── 9. Master pipeline function ──────────────────────────────────────────────

def run_drift_attribution(
    candidates_df,
    ais_positions_df,
    spill_lat,
    spill_lon,
    sar_time_str,
    discharge_time_str,
    env_provider=mock_env_provider,
    n_members=100,
):
    """
    For each Tier-1 vessel in candidates_df, compute a composite confidence
    score C combining drift match, AIS anomaly, morphology, and temporal
    factors.

    Args:
        candidates_df:     Ranked DataFrame from Module 3's run_ais_pipeline().
        ais_positions_df:  Cleaned AIS DataFrame from clean_trajectories()
                           (needed to look up each vessel's last position
                           before discharge_time).
        spill_lat, spill_lon: centroid of the detected spill.
        sar_time_str:      UTC timestamp of SAR image acquisition.
        discharge_time_str: estimated UTC timestamp of bilge discharge.
        env_provider:      callable(lat, lon, time) → forcing dict.
        n_members:         Monte Carlo ensemble size (default 100;
                           raise to 500 for final published numbers).

    Returns:
        DataFrame sorted by C descending with columns:
        [MMSI, VesselName, vessel_category, C, S_drift, S_AIS, S_morphology,
         S_temporal, forward_score, backward_score, anomaly_score]
    """
    discharge_time = pd.to_datetime(discharge_time_str)
    sar_time = pd.to_datetime(sar_time_str)

    # ── Pre-compute scores that are the same for all vessels ─────────────────
    candidates_df = normalize_ais_scores(candidates_df)
    s_temporal = compute_s_temporal(discharge_time)

    # ── Ensure BaseDateTime is parsed ────────────────────────────────────────
    ais_df = ais_positions_df.copy()
    ais_df["BaseDateTime"] = pd.to_datetime(ais_df["BaseDateTime"])

    # ── Filter to Tier-1 vessels only ────────────────────────────────────────
    tier1 = candidates_df[candidates_df["tier1_flag"] == True].copy()  # noqa: E712

    if len(tier1) == 0:
        warnings.warn("No Tier-1 vessels found — returning empty results.")
        return pd.DataFrame(columns=[
            "MMSI", "VesselName", "vessel_category", "C", "S_drift",
            "S_AIS", "S_morphology", "S_temporal", "forward_score",
            "backward_score", "anomaly_score",
        ])

    results = []

    for _, row in tier1.iterrows():
        mmsi = row["MMSI"]
        vessel_name = row.get("VesselName", "UNKNOWN")
        vessel_category = row.get("vessel_category", "unknown")
        s_ais = row["S_AIS"]
        anomaly_score = row["anomaly_score"]

        print(f"  Processing vessel {mmsi} ({vessel_name})...")

        # ── Look up vessel's last AIS position at or before discharge_time ───
        vessel_ais = ais_df[ais_df["MMSI"] == mmsi].copy()
        before_discharge = vessel_ais[vessel_ais["BaseDateTime"] <= discharge_time]

        if len(before_discharge) > 0:
            last_pos = before_discharge.loc[before_discharge["BaseDateTime"].idxmax()]
        elif len(vessel_ais) > 0:
            # Fallback: use earliest available position
            last_pos = vessel_ais.loc[vessel_ais["BaseDateTime"].idxmin()]
            warnings.warn(
                f"  ⚠ No AIS position before discharge_time for MMSI {mmsi}; "
                f"using earliest available position."
            )
        else:
            warnings.warn(
                f"  ⚠ No AIS positions at all for MMSI {mmsi}; skipping."
            )
            continue

        vessel_lat = float(last_pos["LAT"])
        vessel_lon = float(last_pos["LON"])

        # ── Compute S_drift (bidirectional Monte Carlo) ──────────────────────
        drift_result = compute_s_drift(
            vessel_lat, vessel_lon, discharge_time,
            spill_lat, spill_lon, sar_time,
            env_provider=env_provider,
            n_members=n_members,
        )

        # ── Compute S_morphology ─────────────────────────────────────────────
        vessel_heading = None
        if "Heading" in last_pos.index:
            h = last_pos["Heading"]
            if pd.notna(h) and h != 511.0:  # 511 = heading not available
                vessel_heading = float(h)
        s_morphology = compute_s_morphology(
            vessel_heading_deg=vessel_heading,
            spill_elongation_angle_deg=None,  # placeholder — future extension
        )

        # ── Compute composite C ──────────────────────────────────────────────
        c = compute_composite_score(
            drift_result["s_drift"], s_ais, s_morphology, s_temporal,
        )

        results.append({
            "MMSI": mmsi,
            "VesselName": vessel_name,
            "vessel_category": vessel_category,
            "C": c,
            "S_drift": drift_result["s_drift"],
            "S_AIS": s_ais,
            "S_morphology": s_morphology,
            "S_temporal": s_temporal,
            "forward_score": drift_result["forward_score"],
            "backward_score": drift_result["backward_score"],
            "anomaly_score": anomaly_score,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("C", ascending=False).reset_index(drop=True)

    if len(results_df) > 0:
        top = results_df.iloc[0]
        print(
            f"\n  🏆 Top attributed vessel: {top['VesselName']} "
            f"(C={top['C']:.3f})"
        )

    return results_df
