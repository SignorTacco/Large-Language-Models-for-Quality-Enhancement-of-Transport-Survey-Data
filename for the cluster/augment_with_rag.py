import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import gc

# --- CONFIGURATION ---
INPUT_CSV = "data/full_200k_dataset_with_prompts.csv"
OUTPUT_CSV = "data/full_200k_dataset_RAG_READY.csv"
INDEX_PATH = "data/address.index"
META_PATH = "data/address_meta.pkl"

def main():
    print("--- RAG AUGMENTATION (NATIVE PYTORCH MODE - SAFE MEMORY) ---")
    
    # 1. SETUP GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # 2. LOAD RESOURCES
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("Error: Index files not found.")
        sys.exit(1)

    print("Loading Metadata...")
    with open(META_PATH, "rb") as f:
        meta_data = pickle.load(f)

    # 3. LOAD & CONVERT INDEX TO PYTORCH
    print(f"Reading Index from {INDEX_PATH}...")
    cpu_index = faiss.read_index(INDEX_PATH)
    n_vectors = cpu_index.ntotal
    
    print(f"Transferring {n_vectors} vectors to GPU memory...")
    db_vectors_numpy = cpu_index.reconstruct_n(0, n_vectors)
    
    # Move to GPU
    db_tensor = torch.from_numpy(db_vectors_numpy).to(device, dtype=torch.float32)
    db_tensor = F.normalize(db_tensor, p=2, dim=1)
    
    # Free CPU memory
    del cpu_index
    del db_vectors_numpy
    gc.collect()
    
    print("Database Tensor Ready on GPU.")

    # 4. LOAD SURVEY DATA
    print(f"Loading Survey Data: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    mask_start = (df['startTripText_raw'].astype(str) == '-4')
    df.loc[mask_start, 'startTripText_raw'] = df.loc[mask_start, 'startstedadrsogeord']
    
    mask_end = (df['tiladrtext_raw'].astype(str) == '-4')
    df.loc[mask_end, 'tiladrtext_raw'] = df.loc[mask_end, 'tiladrsogeord']

    start_queries = df['startTripText_raw'].fillna("").astype(str).tolist()
    end_queries = df['tiladrtext_raw'].fillna("").astype(str).tolist()
    
    # 5. ENCODE & SEARCH
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    indices_start, scores_start = [], []
    indices_end, scores_end = [], []
    
    # CRITICAL FIX: Smaller batch size to prevent 30GB Result Matrix
    # 256 * 4M * 4bytes = ~4GB VRAM (Safe!)
    BATCH_SIZE = 256 
    
    print("Starting Search Loop...")
    
    # --- START ADDRESSES ---
    print("Processing Start Addresses...")
    for i in range(0, len(start_queries), BATCH_SIZE):
        batch_text = start_queries[i : i + BATCH_SIZE]
        
        # Encode (no_grad saves memory)
        with torch.no_grad():
            batch_embs = model.encode(batch_text, convert_to_tensor=True, show_progress_bar=False)
            batch_embs = F.normalize(batch_embs, p=2, dim=1)
            
            # Search: (256 x 384) * (384 x 4M) = (256 x 4M)
            sim_scores = torch.mm(batch_embs, db_tensor.T)
            
            # Get Top-1
            best_scores, best_indices = torch.max(sim_scores, dim=1)
            
            # Move to CPU immediately
            scores_start.extend(best_scores.cpu().numpy().tolist())
            indices_start.extend(best_indices.cpu().numpy().tolist())
            
            # Explicit cleanup
            del batch_embs
            del sim_scores
        
        if i % 10000 == 0:
            print(f"  Processed {i} / {len(start_queries)}...")

    # --- END ADDRESSES ---
    print("Processing End Addresses...")
    for i in range(0, len(end_queries), BATCH_SIZE):
        batch_text = end_queries[i : i + BATCH_SIZE]
        
        with torch.no_grad():
            batch_embs = model.encode(batch_text, convert_to_tensor=True, show_progress_bar=False)
            batch_embs = F.normalize(batch_embs, p=2, dim=1)
            
            sim_scores = torch.mm(batch_embs, db_tensor.T)
            
            best_scores, best_indices = torch.max(sim_scores, dim=1)
            
            scores_end.extend(best_scores.cpu().numpy().tolist())
            indices_end.extend(best_indices.cpu().numpy().tolist())
            
            del batch_embs
            del sim_scores
        
        if i % 10000 == 0:
            print(f"  Processed {i} / {len(end_queries)}...")

    # Clean GPU
    del db_tensor
    del model
    torch.cuda.empty_cache()

    # 6. CONSTRUCT DATASET
    print("Constructing Final Dataset...")
    
    rag_start_text, rag_start_lat, rag_start_lon = [], [], []
    rag_end_text,   rag_end_lat,   rag_end_lon   = [], [], []
    new_prompts = []

    for i in range(len(df)):
        match_s = meta_data[indices_start[i]]
        score_s = scores_start[i]
        
        match_e = meta_data[indices_end[i]]
        score_e = scores_end[i]
        
        rag_start_text.append(match_s['full_address_text'])
        rag_start_lat.append(match_s['lat'])
        rag_start_lon.append(match_s['lon'])
        
        rag_end_text.append(match_e['full_address_text'])
        rag_end_lat.append(match_e['lat'])
        rag_end_lon.append(match_e['lon'])

        row = df.iloc[i]
        input_context = (
            f"Trip Mode: {row.get('transportmiddel', 'Unknown')}\n"
            f"--- START LOCATION ---\n"
            f"User Input: '{row.get('startTripText_raw', '')}'\n"
            f"Verified Database Match: '{match_s['full_address_text']}' (Confidence: {score_s:.2f})\n"
            f"Verified Coords: ({match_s['lat']:.4f}, {match_s['lon']:.4f})\n"
            f"System Used Coords: ({row.get('start_lat', 0):.4f}, {row.get('start_lon', 0):.4f})\n"
            f"--- END LOCATION ---\n"
            f"User Input: '{row.get('tiladrtext_raw', '')}'\n"
            f"Verified Database Match: '{match_e['full_address_text']}' (Confidence: {score_e:.2f})\n"
            f"Verified Coords: ({match_e['lat']:.4f}, {match_e['lon']:.4f})\n"
            f"System Used Coords: ({row.get('til_lat', 0):.4f}, {row.get('til_lon', 0):.4f})\n"
            f"--- METRICS ---\n"
            f"Reported Dist: {row.get('stagelength_raw', 0)} km\n"
            f"System Calc Dist: {row.get('calc_dist_geo', 0)} km"
        )
        
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Validate this trip. Compare 'System Used Coords' vs 'Verified Coords'.
If System Coords are >1km from Verified Coords (and Confidence >0.75), flag as INVALID COORDINATES.
Otherwise, check speed/distance logic.

### Input:
{input_context}

### Response:
"""
        new_prompts.append(prompt)

    # 7. SAVE
    df['rag_start_address'] = rag_start_text
    df['rag_start_lat'] = rag_start_lat
    df['rag_start_lon'] = rag_start_lon
    df['rag_start_conf'] = scores_start
    
    df['rag_end_address'] = rag_end_text
    df['rag_end_lat'] = rag_end_lat
    df['rag_end_lon'] = rag_end_lon
    df['rag_end_conf'] = scores_end
    
    df['rag_prompt_text'] = new_prompts
    
    print(f"Saving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()