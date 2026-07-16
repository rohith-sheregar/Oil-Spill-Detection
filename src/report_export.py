"""
src/report_export.py — FR-7 Attribution Report Exporter

Serializes the full attribution result to a machine-readable JSON file
containing spill geometry, candidate vessel MMSI, composite confidence
score, and supporting evidence fields.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd


def export_attribution_report(
    spill_geometry: dict,
    drift_results_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Serialize the full attribution result to a JSON file matching FR-7:
    "a machine-readable attribution report (JSON) containing spill geometry,
    candidate vessel MMSI, composite confidence score, and supporting
    evidence fields."

    Args:
        spill_geometry: dict with keys like centroid_lat, centroid_lon,
                        area_km2, elongation, sar_acquisition_time.
        drift_results_df: output of run_drift_attribution() — DataFrame
                          sorted by C descending.
        output_path: file path for the JSON output.
    """
    candidates = []
    for _, row in drift_results_df.iterrows():
        candidates.append({
            "mmsi": int(row["MMSI"]),
            "vessel_name": str(row.get("VesselName", "UNKNOWN")),
            "vessel_category": str(row.get("vessel_category", "unknown")),
            "composite_confidence_score": round(float(row["C"]), 6),
            "evidence": {
                "s_drift": round(float(row["S_drift"]), 6),
                "s_ais_anomaly": round(float(row["S_AIS"]), 6),
                "s_morphology": round(float(row["S_morphology"]), 6),
                "s_temporal": round(float(row["S_temporal"]), 6),
                "forward_drift_match": round(float(row["forward_score"]), 6),
                "backward_drift_match": round(float(row["backward_score"]), 6),
                "raw_ais_anomaly_score": round(float(row["anomaly_score"]), 6),
            },
        })

    report = {
        "spill_geometry": spill_geometry,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"💾 Attribution report saved → {output_path}")
