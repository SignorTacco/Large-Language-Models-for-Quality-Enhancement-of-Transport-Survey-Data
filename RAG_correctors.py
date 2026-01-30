import pandas as pd
import numpy as np
import requests
import json
import time
from tqdm import tqdm
import os

# --- SECTION 2: THE BRIDGE (ADAPTERS) ---

def get_osrm_route_bridge(lat1, lon1, lat2, lon2, transport_mode):
    """
    Local OSRM API call with Multi-Modal Routing.
    Selects the correct local port based on transport mode.
    """
    if pd.isna(lat1) or pd.isna(lat2):
        return (None, None)

    # --- PORT SELECTION LOGIC ---
    try:
        mode = int(transport_mode)
        if mode == 1:
            # Walk -> Port 5002
            port = 5002
        elif mode == 2:
            # Bike -> Port 5001
            port = 5001
        elif 3 <= mode <= 25:
            # Car -> Port 5000
            port = 5000
        else:
            return (None, None)
    except:
        return (None, None)

    # Local Docker URL 
    base_url = f"http://127.0.0.1:{port}/route/v1/driving"
    
    # OSRM expects: lon,lat;lon,lat
    coords_str = f"{lon1},{lat1};{lon2},{lat2}"
    url = f"{base_url}/{coords_str}?overview=false"

    try:
        # Timeout is 0.5s because local is instant
        r = requests.get(url, timeout=2) 
        if r.status_code == 200:
            data = r.json()
            if data['code'] == 'Ok' and data['routes']:
                route = data['routes'][0]
                dist_km = route['distance'] / 1000.0
                dur_min = route['duration'] / 60.0
                
                result = (round(dist_km, 1), np.ceil(dur_min))
                return result
    except Exception as e:
        # print(f"OSRM Error on port {port}: {e}")
        pass

    return (None, None)

# --- SECTION 3: MAIN LOGIC ---

def repair_hard_wrongs(df):
    """
    Main entry point.
    """
    # Filter for hard wrongs or invalid
    mask = (df['llm_verdict'].isin(['INVALID']))
    subset = df[mask].copy()
    
    print(f"--- Starting Correction Pipeline on {len(subset)} rows ---")
    
    # Initialize output columns if not already present
    new_float_cols = ['corrected_dist_km', 'corrected_time_min']
    
    # Initialize numeric columns with NaN if not present
    for col in new_float_cols:
        if col not in subset.columns:
            subset[col] = np.nan

    # Initialize text column specifically as 'object' (string) type if not present
    if 'note_correction' not in subset.columns:
        subset['note_correction'] = pd.Series(dtype='object')
        subset['note_correction'] = np.nan
        subset['note_correction'] = subset['note_correction'].astype(object)
            
    success_count = 0
    osrm_attempted_count = 0
    counter = 0

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Correcting rows"):
        # Skip if already processed
        if pd.notna(subset.at[idx, 'note_correction']):
            continue
        
        counter += 1
        
        notes = []
        
        # Always attempt OSRM recalc with finalized coords
        osrm_attempted_count += 1
        dist, time = get_osrm_route_bridge(row['rag_start_lat'], row['rag_start_lon'], row['rag_end_lat'], row['rag_end_lon'], row['transportmiddel'])
        
        if dist is not None:
            subset.at[idx, 'corrected_dist_km'] = dist
            subset.at[idx, 'corrected_time_min'] = time
            subset.at[idx, 'llm_verdict'] = 'corrected'
            notes.append("osrm_recalc")
            success_count += 1
        else:
            notes.append("osrm_api_fail")
        
        # Write notes
        if notes:
            subset.at[idx, 'note_correction'] = ";".join(notes)
        
        if counter % 200 == 0:
            print(f"Processed {counter} rows, fixed {success_count} so far.")

    print(f"--- Finished. Corrected {success_count} rows. OSRM attempted: {osrm_attempted_count} ---")
    return subset