import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

def haversine(lat1, lon1, lat2, lon2):
    """Compute the great-circle distance between two points on Earth in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def load_ais(filepath, spill_lat, spill_lon, spill_time_str, radius_km=50, time_window_hours=6):
    df = pd.read_csv(filepath)
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    
    spill_time = pd.to_datetime(spill_time_str)
    
    # Filter spatially
    df['dist_to_spill'] = haversine(df['LAT'].values, df['LON'].values, spill_lat, spill_lon)
    df = df[df['dist_to_spill'] <= radius_km].copy()
    
    # Filter temporally
    time_diff = (df['BaseDateTime'] - spill_time).dt.total_seconds() / 3600.0
    df = df[time_diff.abs() <= time_window_hours].copy()
    
    df = df.drop(columns=['dist_to_spill'])
    return df

def filter_vessel_types(df):
    def get_category(row):
        vt = row['VesselType']
        length = row['Length']
        
        if pd.isna(vt):
            return None
            
        if 70.0 <= vt <= 79.9:
            return 'cargo'
        elif 80.0 <= vt <= 89.9:
            return 'tanker'
        elif 30.0 <= vt <= 30.9:
            if pd.notna(length) and length > 50.0:
                return 'fishing'
        return None

    df['vessel_category'] = df.apply(get_category, axis=1)
    df = df.dropna(subset=['vessel_category']).copy()
    return df

def clean_trajectories(df):
    if len(df) == 0:
        df['dbscan_label'] = []
        return df

    lat_min, lat_max = df['LAT'].min(), df['LAT'].max()
    lon_min, lon_max = df['LON'].min(), df['LON'].max()
    time_min, time_max = df['BaseDateTime'].min(), df['BaseDateTime'].max()
    
    lat_range = lat_max - lat_min if lat_max > lat_min else 1.0
    lon_range = lon_max - lon_min if lon_max > lon_min else 1.0
    time_range = (time_max - time_min).total_seconds() if time_max > time_min else 1.0
    
    lat_norm = (df['LAT'] - lat_min) / lat_range
    lon_norm = (df['LON'] - lon_min) / lon_range
    time_norm = (df['BaseDateTime'] - time_min).dt.total_seconds() / time_range
    
    X = np.column_stack([lat_norm, lon_norm, time_norm])
    
    dbscan = DBSCAN(eps=0.01, min_samples=3)
    df['dbscan_label'] = dbscan.fit_predict(X)
    
    # Remove noise points
    df = df[df['dbscan_label'] != -1].copy()
    return df

def compute_vessel_features(df, spill_lat, spill_lon):
    features = []
    
    for mmsi, group in df.groupby('MMSI'):
        # sog_variance
        sog_var = group['SOG'].var() if len(group) > 1 else 0.0
        if pd.isna(sog_var): sog_var = 0.0
        
        # course_deviation
        cog = group['COG'].values
        if len(cog) > 1:
            diffs = np.abs(np.diff(cog))
            diffs = np.where(diffs > 180, 360 - diffs, diffs)
            course_dev = np.mean(diffs)
        else:
            course_dev = 0.0
            
        # stop_event_count
        stop_count = (group['SOG'] < 0.5).sum()
        
        # min_dist_to_spill
        dists = haversine(group['LAT'].values, group['LON'].values, spill_lat, spill_lon)
        min_dist = np.min(dists)
        
        # transit_directionality
        mean_heading = group['Heading'].mean()
        mean_lat = group['LAT'].mean()
        mean_lon = group['LON'].mean()
        
        dlon = np.radians(spill_lon - mean_lon)
        lat1 = np.radians(mean_lat)
        lat2 = np.radians(spill_lat)
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360
        
        angle_diff = mean_heading - bearing
        transit_dir = np.cos(np.radians(angle_diff))
        if pd.isna(transit_dir): transit_dir = 0.0
        
        # ais_gap_minutes
        times = group['BaseDateTime'].sort_values()
        if len(times) > 1:
            gaps = times.diff().dt.total_seconds() / 60.0
            max_gap = gaps.max()
        else:
            max_gap = 0.0
            
        vessel_name = group['VesselName'].iloc[0]
        vessel_category = group['vessel_category'].iloc[0]
        
        features.append({
            'MMSI': mmsi,
            'VesselName': vessel_name,
            'vessel_category': vessel_category,
            'sog_variance': sog_var,
            'course_deviation': course_dev,
            'stop_event_count': stop_count,
            'min_dist_to_spill': min_dist,
            'transit_directionality': transit_dir,
            'ais_gap_minutes': max_gap
        })
        
    df_features = pd.DataFrame(features)
    if len(df_features) > 0:
        df_features = df_features.set_index('MMSI')
    return df_features

def score_anomalies(features_df):
    if len(features_df) == 0:
        return features_df
        
    feature_cols = ['sog_variance', 'course_deviation', 'stop_event_count', 
                    'min_dist_to_spill', 'transit_directionality', 'ais_gap_minutes']
    
    X = features_df[feature_cols].fillna(0)
    
    clf = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    clf.fit(X)
    
    scores = -clf.score_samples(X)
    features_df['anomaly_score'] = scores
    
    # Top 5% tier1_flag
    threshold = np.percentile(scores, 95)
    features_df['tier1_flag'] = features_df['anomaly_score'] >= threshold
    
    features_df = features_df.sort_values('anomaly_score', ascending=False)
    
    cols = ['MMSI', 'VesselName', 'vessel_category', 'anomaly_score', 'tier1_flag'] + feature_cols
    return features_df.reset_index()[cols]

def run_ais_pipeline(filepath, spill_lat, spill_lon, spill_time_str):
    print("Starting AIS Pipeline...")
    
    df = load_ais(filepath, spill_lat, spill_lon, spill_time_str)
    print(f"  -> {len(df)} rows after spatial/temporal filter")
    
    df = filter_vessel_types(df)
    print(f"  -> {len(df)} rows after vessel type filter")
    
    df = clean_trajectories(df)
    print(f"  -> {len(df)} rows after DBSCAN cleaning")
    
    features_df = compute_vessel_features(df, spill_lat, spill_lon)
    print(f"  -> {len(features_df)} unique vessels featured")
    
    results = score_anomalies(features_df)
    if len(results) > 0:
        print(f"  -> {results['tier1_flag'].sum()} Tier-1 vessels identified")
    else:
        print("  -> 0 Tier-1 vessels identified")
        
    return results
