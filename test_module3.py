import os
import pandas as pd
from src.ais_filter import (
    load_ais, 
    filter_vessel_types, 
    clean_trajectories, 
    compute_vessel_features, 
    score_anomalies,
    run_ais_pipeline
)

def test_module3():
    filepath = "data/ais_gulf_test.csv"
    spill_lat = 29.2
    spill_lon = -94.8
    spill_time_str = "2022-08-03 14:00:00"
    
    if not os.path.exists(filepath):
        print(f"Skipping tests: {filepath} not found.")
        return

    # 1. load_ais returns non-empty DataFrame with correct columns
    df1 = load_ais(filepath, spill_lat, spill_lon, spill_time_str)
    assert len(df1) > 0, "load_ais returned empty DataFrame"
    assert 'MMSI' in df1.columns and 'BaseDateTime' in df1.columns, "load_ais missing required columns"

    # 2. filter_vessel_types keeps only tanker/cargo/fishing categories
    df2 = filter_vessel_types(df1)
    assert len(df2) > 0, "filter_vessel_types returned empty DataFrame"
    unique_categories = set(df2['vessel_category'].unique())
    assert unique_categories.issubset({'cargo', 'tanker', 'fishing'}), "filter_vessel_types kept invalid categories"

    # 3. compute_vessel_features returns exactly 6 feature columns per MMSI
    df3 = clean_trajectories(df2)
    df4 = compute_vessel_features(df3, spill_lat, spill_lon)
    
    expected_features = {'sog_variance', 'course_deviation', 'stop_event_count', 
                         'min_dist_to_spill', 'transit_directionality', 'ais_gap_minutes'}
    actual_features = set(df4.columns) - {'VesselName', 'vessel_category'}
    assert expected_features == actual_features, f"Feature columns mismatch. Expected {expected_features}, got {actual_features}"
    assert df4.index.name == 'MMSI', "Features DataFrame not indexed by MMSI"

    # 4. score_anomalies output has tier1_flag column and at least 1 True value
    df5 = score_anomalies(df4)
    assert 'tier1_flag' in df5.columns, "tier1_flag missing from anomaly scoring output"
    assert df5['tier1_flag'].sum() >= 1, "score_anomalies did not flag any Tier 1 vessels"

    # 5. Final output is sorted by anomaly_score descending
    scores = df5['anomaly_score'].values
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Final output is not sorted by anomaly_score descending"

    print("[SUCCESS] All Module 3 assertions passed.")

if __name__ == "__main__":
    test_module3()
    
    print("\n--- Running Master Pipeline ---")
    if os.path.exists("data/ais_gulf_test.csv"):
        results = run_ais_pipeline(
            filepath="data/ais_gulf_test.csv",
            spill_lat=29.2,
            spill_lon=-94.8,
            spill_time_str="2022-08-03 14:00:00"
        )
        if len(results) > 0:
            print("\nTop 10 Suspect Vessels:")
            print(results.head(10).to_string())
    else:
        print("Skipping pipeline execution: data/ais_gulf_test.csv not found.")
