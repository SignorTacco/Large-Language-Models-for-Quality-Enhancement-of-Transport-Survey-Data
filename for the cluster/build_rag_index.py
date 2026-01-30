import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os

# --- CONFIGURATION ---
# Path to your source CSV (The 4 Million Rows)
ADDRESS_FILE = "data/address_all_denmark.csv" 
# Output paths
INDEX_FILE = "data/address.index"
META_FILE = "data/address_meta.pkl"

def main():
    # 0. CHECK FILE
    if not os.path.exists(ADDRESS_FILE):
        raise FileNotFoundError(f"Cannot find {ADDRESS_FILE}. Did you upload it?")

    print("1. Loading Embedding Model...")
    # This model works perfectly on the L40S GPU
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("2. Loading Address Data...")
    # We only read the columns we actually need to save RAM
    # Adjust 'line' if your CSV header is different (e.g., 'full_address_text')
    use_cols = ['line', 'lat', 'lon'] 
    df = pd.read_csv(ADDRESS_FILE, usecols=use_cols)
    
    # Drop rows with empty addresses or coords just in case
    print(f"   Original Rows: {len(df)}")
    df.dropna(subset=['line', 'lat', 'lon'], inplace=True)
    print(f"   Cleaned Rows:  {len(df)}")
    
    # Extract list of strings for embedding
    # We use the 'line' column as the full address text
    addresses = df['line'].astype(str).tolist()
    
    print("3. Encoding Addresses (GPU Accelerated)...")
    # Batch size 2048 is efficient on L40S (48GB VRAM)
    embeddings = model.encode(addresses, batch_size=2048, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize embeddings (Critical for Cosine Similarity search)
    faiss.normalize_L2(embeddings)

    print("4. Building FAISS Index...")
    dimension = embeddings.shape[1] # Should be 384 for MiniLM
    
    # We use "FlatIP" (Inner Product) for exact search. 
    # It is brute-force but extremely fast on GPU for 4M rows.
    res = faiss.StandardGpuResources()
    
    # Create index configuration
    index_flat = faiss.IndexFlatIP(dimension)
    
    # Move index to GPU
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    # Add vectors
    gpu_index.add(embeddings)
    print(f"   Index contains {gpu_index.ntotal} vectors.")
    
    print("5. Saving Index to Disk...")
    # Move back to CPU to save
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, INDEX_FILE)
    
    print("6. Saving Metadata...")
    # We save 'line' as 'full_address' to match the retrieval script logic
    meta_data = df.rename(columns={'line': 'full_address_text'}).to_dict('records')
    
    with open(META_FILE, "wb") as f:
        pickle.dump(meta_data, f)

    print("Done! RAG Index is ready in 'data/' folder.")

if __name__ == "__main__":
    main()