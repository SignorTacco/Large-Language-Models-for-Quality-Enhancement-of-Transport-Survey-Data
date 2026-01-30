import pandas as pd
import numpy as np
import requests
import json
import time
from tqdm import tqdm
import os

# --- SECTION 1: IMPORTS ---
try:
    # 1. The Text Matcher
    from retriever_all_dk import AddressRetriever
    
    # 2. The Coordinate Lookup (Corrected Import)
    from lat_lon_dawa_request import dawa_geocode
    
except ImportError as e:
    AddressRetriever = None
    dawa_geocode = None
    print(f"CRITICAL WARNING: Could not import local modules. {e}")
    print("Ensure 'retriever_all_dk.py' and 'lat_lon_dawa_request.py' are in the folder.")

# --- SECTION 2: THE BRIDGE (ADAPTERS) ---

# Global cache for OSRM results
_OSRM_CACHE = {}

# Global cache for retrieve results
_RETRIEVE_CACHE = {}

# Global cache for coordinates from row_index
_ROW_COORDS_CACHE = {}

# Global cache for final coordinates results
_COORDS_CACHE = {}

# Global retriever instance (singleton)
_RETRIEVER_INSTANCE = None

def get_retriever():
    """Load the retriever instance (singleton)."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None and AddressRetriever:
        _RETRIEVER_INSTANCE = AddressRetriever()
    return _RETRIEVER_INSTANCE

def get_coords_from_text_bridge(raw_text):
    """
    1. MATCH: Uses AddressRetriever to find the best canonical address string and row_index.
    2. GEOCODE: Uses local index to get coords from row_index.
    """
    if raw_text in _COORDS_CACHE:
        return _COORDS_CACHE[raw_text]
    
    retriever = get_retriever()
    
    # Basic safety checks
    if not retriever or not raw_text or pd.isna(raw_text):
        _COORDS_CACHE[raw_text] = None
        return None
    if str(raw_text).lower() == 'nan':
        _COORDS_CACHE[raw_text] = None
        return None

    try:
        # Step A: Get best text match
        if raw_text in _RETRIEVE_CACHE:
            results = _RETRIEVE_CACHE[raw_text]
        else:
            results = retriever.retrieve(raw_text, topk_output=1)
            _RETRIEVE_CACHE[raw_text] = results
        
        if not results:
            _COORDS_CACHE[raw_text] = None
            return None
            
        best_score, best_addr_str, row_index = results[0]
        
        # Quality Filter (Adjust threshold if needed, default 40 for speed)
        if best_score < 40:
            _COORDS_CACHE[raw_text] = None
            return None

        # Step B: Get Coords from row_index
        if row_index in _ROW_COORDS_CACHE:
            lat, lon = _ROW_COORDS_CACHE[row_index]
        else:
            lat = retriever.rows.loc[row_index, 'lat']
            lon = retriever.rows.loc[row_index, 'lon']
            _ROW_COORDS_CACHE[row_index] = (lat, lon)
        
        result = (float(lat), float(lon), best_score, best_addr_str)
        _COORDS_CACHE[raw_text] = result
        return result
            
    except Exception as e:
        # print(f"Lookup failed for {raw_text}: {e}")
        _COORDS_CACHE[raw_text] = None
        return None

def get_osrm_route_bridge(lat1, lon1, lat2, lon2, transport_mode):
    """
    Local OSRM API call with Multi-Modal Routing.
    Selects the correct local port based on transport mode.
    """
    key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4), transport_mode)
    if key in _OSRM_CACHE:
        return _OSRM_CACHE[key]
    
    if pd.isna(lat1) or pd.isna(lat2):
        _OSRM_CACHE[key] = (None, None)
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
            _OSRM_CACHE[key] = (None, None)
            return (None, None)
    except:
        _OSRM_CACHE[key] = (None, None)
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
                _OSRM_CACHE[key] = result
                return result
    except Exception as e:
        # print(f"OSRM Error on port {port}: {e}")
        pass

    result = (None, None)
    _OSRM_CACHE[key] = result
    return result


# --- SECTION 3: MAIN LOGIC ---

def repair_hard_wrongs(df):
    """
    Main entry point.
    """
    # Filter for hard wrongs
    mask = (df['validation_status'] == 'hard_wrong')
    subset = df[mask].copy()
    
    print(f"--- Starting Correction Pipeline on {len(subset)} rows ---")
    
    checkpoint_file = 'corrections_checkpoint.csv'
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        subset_checkpoint = pd.read_csv(checkpoint_file, index_col=0)
        # Update subset with checkpoint data
        subset.update(subset_checkpoint)
        print(f"Resumed from checkpoint, {len(subset_checkpoint)} rows loaded")
    
    # Initialize output columns if not already present
    new_float_cols = ['corrected_start_lat', 'corrected_start_lon', 
                      'corrected_til_lat', 'corrected_til_lon', 
                      'corrected_dist_km', 'corrected_time_min']
    
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
    coords_fixed_count = 0
    osrm_attempted_count = 0

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Correcting rows"):
        # Skip if already processed
        if pd.notna(subset.at[idx, 'note_correction']):
            continue
        
        notes = []
        
        # Track current best coordinates (default to original)
        s_lat, s_lon = row['start_lat'], row['start_lon']
        e_lat, e_lon = row['til_lat'],   row['til_lon']
        
        coords_changed = False
        
        # -----------------------------------------------------
        # CASE 1: BAD COORDINATES (Fix with lower threshold)
        # -----------------------------------------------------
        if row['flag_coords_bad']:
            
            # Start Address
            s_txt = row['startTripText_raw'] # User input
            res_s = get_coords_from_text_bridge(s_txt)
            if res_s:
                s_lat, s_lon, score, txt = res_s
                subset.at[idx, 'corrected_start_lat'] = s_lat
                subset.at[idx, 'corrected_start_lon'] = s_lon
                notes.append(f"start_fix({score:.0f})")
                coords_changed = True
            
            # End Address
            e_txt = row['tiladrtext_raw'] # User input
            res_e = get_coords_from_text_bridge(e_txt)
            if res_e:
                e_lat, e_lon, score, txt = res_e
                subset.at[idx, 'corrected_til_lat'] = e_lat
                subset.at[idx, 'corrected_til_lon'] = e_lon
                notes.append(f"end_fix({score:.0f})")
                coords_changed = True
                
            # If coords fixed, mark as coords_changed
            if coords_changed:
                coords_fixed_count += 1

        # -----------------------------------------------------
        # CASE 2: ZERO DISTANCE (Skip)
        # -----------------------------------------------------
        elif row['flag_zero_dist']:
            subset.at[idx, 'note_correction'] = "skipped_zero_dist"
            continue

        # -----------------------------------------------------
        # CASE 3: BAD SPEED OR CHANGED COORDS (Recalc OSRM)
        # -----------------------------------------------------
        should_recalc = coords_changed or row['flag_speed_bad']
        
        if should_recalc:
            osrm_attempted_count += 1
            dist, time = get_osrm_route_bridge(s_lat, s_lon, e_lat, e_lon, row['transportmiddel'])
            
            if dist is not None:
                subset.at[idx, 'corrected_dist_km'] = dist
                subset.at[idx, 'corrected_time_min'] = time
                subset.at[idx, 'validation_status'] = 'corrected'
                notes.append("osrm_recalc")
                success_count += 1
            else:
                notes.append("osrm_api_fail")
        
        # Write notes
        if notes:
            subset.at[idx, 'note_correction'] = ";".join(notes)
            
        # Save checkpoint every 100 rows
        if (idx + 1) % 100 == 0:
            subset.to_csv(checkpoint_file, index=True)
            print(f"Saved checkpoint at row {idx + 1}")

    print(f"--- Finished. Corrected {success_count} rows. Coords fixed: {coords_fixed_count}, OSRM attempted: {osrm_attempted_count} ---")
    return subset