# fetch_all_denmark_csv.py
import csv, json, time, math, sys, unicodedata, argparse, pathlib, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://api.dataforsyningen.dk"

def fold_for_match(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

def get_kommuner():
    r = requests.get(f"{BASE}/kommuner", timeout=60)
    r.raise_for_status()
    data = r.json()
    # Normalize to 4-digit strings
    return [f"{int(k['kode']):04d}" for k in data]

def stream_adresser_for_kommune(kommunekode: str, retries: int = 4, backoff: float = 2.0):
    """
    Stream /adresser for a kommune as NDJSON with struktur=mini (flat).
    Yields dicts.
    """
    params = {"struktur": "mini", "ndjson": "", "kommunekode": kommunekode}
    url = f"{BASE}/adresser"
    attempt = 0
    while True:
        try:
            with requests.get(url, params=params, stream=True, timeout=180) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    yield json.loads(line)
            return
        except Exception as e:
            attempt += 1
            if attempt > retries:
                print(f"[{kommunekode}] FAILED after {retries} retries: {e}", file=sys.stderr)
                return
            sleep = backoff * (2 ** (attempt - 1))
            print(f"[{kommunekode}] retry {attempt}/{retries} in {sleep:.1f}s … ({e})", file=sys.stderr)
            time.sleep(sleep)

def write_chunk_csv(chunk_path: pathlib.Path, kommunekode: str):
    fields = [
        "adresse_id","vejnavn","husnr","kommunekode",
        "postnr","postnrnavn","postnavn","lon","lat",
        "vejnavn_norm","line"
    ]
    count = 0
    with chunk_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in stream_adresser_for_kommune(kommunekode):
            vejnavn = a.get("vejnavn","")
            husnr = a.get("husnr","")
            postnr = str(a.get("postnr") or "")
            postnrnavn = a.get("postnrnavn") or a.get("postnavn") or ""
            postnavn = a.get("postnavn") or ""
            lon, lat = a.get("x"), a.get("y")
            line = f"{vejnavn} {husnr}, {postnr} {postnrnavn}".strip()
            w.writerow({
                "adresse_id": a.get("id",""),
                "vejnavn": vejnavn,
                "husnr": husnr,
                "kommunekode": f"{int(a['kommunekode']):04d}" if a.get("kommunekode") else "",
                "postnr": postnr,
                "postnrnavn": postnrnavn,
                "postnavn": postnavn,
                "lon": lon,
                "lat": lat,
                "vejnavn_norm": fold_for_match(vejnavn),
                "line": line
            })
            count += 1
    return count

def merge_chunks(chunks_dir: pathlib.Path, out_csv: pathlib.Path):
    fields = [
        "adresse_id","vejnavn","husnr","kommunekode",
        "postnr","postnrnavn","postnavn","lon","lat",
        "vejnavn_norm","line"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=fields)
        w.writeheader()
        for chunk in sorted(chunks_dir.glob("*.csv")):
            with chunk.open("r", encoding="utf-8", newline="") as in_f:
                r = csv.DictReader(in_f)
                for row in r:
                    w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Fetch ALL Danish addresses to CSV.")
    ap.add_argument("--out", help="Single output CSV (writes directly).")
    ap.add_argument("--chunks_dir", help="If set, writes one CSV per kommune here, then you can merge.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers for chunk mode (default 4).")
    args = ap.parse_args()

    if not args.out and not args.chunks_dir:
        ap.error("Specify either --out (single CSV) or --chunks_dir (per-kommune CSVs).")

    kommuner = get_kommuner()
    print(f"Found {len(kommuner)} municipalities")

    if args.chunks_dir:
        # Parallel per-kommune chunks, then user can merge (or call --out as well after)
        chunks_dir = pathlib.Path(args.chunks_dir)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        totals = {}
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {}
            for kode in kommuner:
                p = chunks_dir / f"adresser_{kode}.csv"
                futs[ex.submit(write_chunk_csv, p, kode)] = (kode, p)
            done = 0
            for fut in as_completed(futs):
                kode, p = futs[fut]
                try:
                    n = fut.result()
                    totals[kode] = n
                    done += 1
                    print(f"[{done}/{len(kommuner)}] {kode}: {n} rows → {p.name}")
                except Exception as e:
                    print(f"[{kode}] ERROR: {e}", file=sys.stderr)

        total_rows = sum(totals.values())
        print(f"Done chunks: {total_rows} rows total in {chunks_dir}")

        if args.out:
            out_csv = pathlib.Path(args.out)
            print(f"Merging chunks → {out_csv}")
            merge_chunks(chunks_dir, out_csv)
            print(f"Merged CSV written: {out_csv}")

    else:
        # Single big CSV (sequential; simpler but longer)
        out_csv = pathlib.Path(args.out)
        fields = [
            "adresse_id","vejnavn","husnr","kommunekode",
            "postnr","postnrnavn","postnavn","lon","lat",
            "vejnavn_norm","line"
        ]
        total = 0
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, kode in enumerate(kommuner, 1):
                rows = 0
                for a in stream_adresser_for_kommune(kode):
                    vejnavn = a.get("vejnavn","")
                    husnr = a.get("husnr","")
                    postnr = str(a.get("postnr") or "")
                    postnrnavn = a.get("postnrnavn") or a.get("postnavn") or ""
                    postnavn = a.get("postnavn") or ""
                    lon, lat = a.get("x"), a.get("y")
                    line = f"{vejnavn} {husnr}, {postnr} {postnrnavn}".strip()
                    w.writerow({
                        "adresse_id": a.get("id",""),
                        "vejnavn": vejnavn,
                        "husnr": husnr,
                        "kommunekode": f"{int(a['kommunekode']):04d}" if a.get("kommunekode") else "",
                        "postnr": postnr,
                        "postnrnavn": postnrnavn,
                        "postnavn": postnavn,
                        "lon": lon,
                        "lat": lat,
                        "vejnavn_norm": fold_for_match(vejnavn),
                        "line": line
                    })
                    rows += 1
                total += rows
                print(f"[{i}/{len(kommuner)}] {kode}: {rows} rows")
        print(f"Done. Wrote {total} rows to {out_csv}")

if __name__ == "__main__":
    main()