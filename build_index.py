# build_index.py
import pandas as pd
import pickle, pathlib, unicodedata
from collections import defaultdict

INPUT_CSV = "adresser_all_denmark_v1.csv"   # your big CSV
OUT_DIR   = pathlib.Path("addr_index"); OUT_DIR.mkdir(exist_ok=True)

NEEDED = ["vejnavn","husnr","postnr","postnrnavn","vejnavn_norm","lat","lon"]

def fold_for_match(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

print("Loading CSV…")
# dtype=str keeps diacritics; low_memory=False for safe types
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False, encoding="utf-8", low_memory=False)
missing = set(NEEDED) - set(df.columns)
if missing:
    raise RuntimeError(f"CSV missing columns: {missing}")

# Normalize/clean
df["vejnavn_norm"] = df["vejnavn_norm"].map(fold_for_match)
df["husnr_low"] = df["husnr"].str.lower()

# Keep only needed columns + helper
slim = df[["vejnavn","husnr","husnr_low","postnr","postnrnavn","vejnavn_norm","lat","lon"]].copy()

print("Building street -> rows map…")
street_to_rows = defaultdict(list)
for i, v in enumerate(slim["vejnavn_norm"].values):
    street_to_rows[v].append(i)

print("Building (street,house) -> rows map (optional but helpful)…")
street_house_to_rows = defaultdict(list)
for i, (v, h) in enumerate(zip(slim["vejnavn_norm"].values, slim["husnr_low"].values)):
    street_house_to_rows[(v, h)].append(i)

unique_streets = list(street_to_rows.keys())
print(f"Unique streets: {len(unique_streets):,} rows: {len(slim):,}")

print("Saving artifacts…")
slim.to_parquet(OUT_DIR / "rows.parquet", index=False)
with open(OUT_DIR / "unique_streets.pkl", "wb") as f:
    pickle.dump(unique_streets, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(OUT_DIR / "street_to_rows.pkl", "wb") as f:
    pickle.dump(street_to_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(OUT_DIR / "street_house_to_rows.pkl", "wb") as f:
    pickle.dump(street_house_to_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Index build complete.")