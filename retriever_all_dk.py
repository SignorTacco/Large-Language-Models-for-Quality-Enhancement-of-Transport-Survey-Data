# retriever_all_dk.py (OPTIMIZED VERSION)
from __future__ import annotations
import re, unicodedata, pickle, pathlib
import pandas as pd
from rapidfuzz import process, fuzz
from typing import List, Dict, Tuple

IDX_DIR = pathlib.Path("addr_index")

# --- utils ---
SUFFIXES = [
    "vej","gade","gaden","boulevard","allé","alle","plads","stræde","torv",
    "kaj","park", "parken","sti", "stien","bakke","vænge","have","engen","gård","runddel",
    "boulevarden","boulev.", "alléen","vang","vangen", "vejen", "svinget", "skoven", "ringen", "centret",
]
SUFFIX_PATTERN = re.compile(
    r"([A-Za-zÀ-ÖØ-öø-ÿ0-9 .'\-]+?(?:"
    + "|".join(SUFFIXES)
    + r"))\b", flags=re.IGNORECASE
)
HOUSE_RE   = re.compile(r"(?P<num>\d+[a-zA-Z]?)")
POSTNR_RE  = re.compile(r"\b(\d{4})\b")

# Neighborhood to postcode mappings
NEIGHBORHOOD_MAPPING = {
    "nørrebro": "2200", "nørre": "2200", "østerbro": "2100", "øster": "2100",
    "vesterbro": "1620", "vester": "1620", "amager": "2300", "christianshavn": "1400",
    "frederiksberg": "2000", "valby": "2500", "brønshøj": "2700", "vanløse": "2720",
    "hvidovre": "2650", "rødovre": "2610", "glostrup": "2600", "albertslund": "2620",
    "herlev": "2730", "ballerup": "2750", "smørum": "2765", "malmø": "2765",
    "kongens lyngby": "2800", "lyngby": "2800", "gentofte": "2820", "klampenborg": "2930",
    "charlottenlund": "2920", "ordrup": "2920", "hellerup": "2900", "inderhavnen": "2100",
    "holmen": "1437", "christiania": "1437", "refshaleøen": "1432", "tåsinge": "5700",
    "langeland": "5900", "ærø": "5970", "als": "6400", "fyn": "5000",
    "midtjylland": "8000", "østjylland": "8000", "vestjylland": "7500",
    "nordjylland": "9000", "sjælland": "4000", "bornholm": "3700",
    "lolland": "4900", "falster": "4800", "møn": "4780", "hoje taastrup": "2630",
}

def fold_for_match(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

def parse_raw(raw: str) -> dict:
    s = (raw or "").strip()
    
    # Extract postcode
    postnr_candidates = []
    for m in POSTNR_RE.finditer(s):
        candidate = m.group(1)
        score = 0
        score += m.start() * 0.1
        after_context = s[m.end():m.end()+20].lower()
        if any(city in after_context for city in ['skive', 'københavn', 'aarhus', 'odense', 'aalborg']):
            score += 10
        if not candidate.startswith('0'):
            score += 5
        postnr_candidates.append((candidate, score))
    
    postnr = None
    if postnr_candidates:
        postnr_candidates.sort(key=lambda x: -x[1])
        postnr = postnr_candidates[0][0]

    # Find house number
    house_candidates = []
    for m in HOUSE_RE.finditer(s):
        num = m.group("num")
        score = 0
        if any(suffix in s[max(0, m.start()-20):m.start()].lower() for suffix in SUFFIXES):
            score += 10
        if postnr and s[m.end():m.end()+10].strip().startswith(postnr):
            score += 20
        if m.start() > 5:
            score += 5
        house_candidates.append((num, score, m.start()))
    
    house = None
    if house_candidates:
        house_candidates.sort(key=lambda x: (-x[1], x[2]))
        house = house_candidates[0][0]

    # Find Street
    street_guess = None
    if house:
        candidates = []
        segments = [seg.strip() for seg in s.split(',') if seg.strip()]
        for segment in segments:
            words = re.split(r'[,\s]+', segment)
            for i in range(len(words)):
                for j in range(i + 1, min(i + 5, len(words) + 1)):
                    span = ' '.join(words[i:j])
                    span_lower = span.lower()
                    for suffix in SUFFIXES:
                        if span_lower.endswith(suffix.lower()):
                            span_pos = s.lower().find(span_lower)
                            if span_pos != -1:
                                remaining = re.sub(r'^[,\s]+', '', s[span_pos + len(span):].strip())
                                if remaining.startswith(house):
                                    score = len(span)
                                    if ',' in s[:span_pos]: score += 10
                                    candidates.append((span, score, span_pos + len(span)))
                                    break
        if candidates:
            candidates.sort(key=lambda x: (-x[1], -len(x[0]) if len(x[0]) <= 25 else 0, -x[2]))
            street_guess = candidates[0][0]

    if not street_guess:
        for m in SUFFIX_PATTERN.finditer(s):
            cand = m.group(1).strip()
            if (street_guess is None) or (len(cand) > len(street_guess)):
                street_guess = cand

    if not street_guess:
        tmp = HOUSE_RE.sub(" ", s)
        if postnr:
            postnr_pos = s.find(postnr)
            if postnr_pos != -1:
                tmp = s[:postnr_pos].strip()
                tmp = HOUSE_RE.sub(" ", tmp)
        chunks = [c.strip() for c in re.split(r"[,/;]", tmp) if c.strip()]
        street_guess = max(chunks, key=len, default=s)

    low = s.lower()
    local_hint = None
    if any(t in low for t in ["kbh s","københavn s","kobenhavn s","cph s"]):
        local_hint = "2300"
    
    if not postnr and not local_hint:
        for neighborhood, postcode in NEIGHBORHOOD_MAPPING.items():
            if neighborhood in low:
                local_hint = postcode
                break
    
    return {"street": street_guess.strip(), "house": (house or "").lower(), "postnr": postnr or local_hint}

# --- retriever ---
class AddressRetriever:
    def __init__(self, idx_dir: str | pathlib.Path = IDX_DIR):
        idx_dir = pathlib.Path(idx_dir)
        print("Loading Address Index...")
        self.rows = pd.read_parquet(idx_dir / "rows.parquet")
        
        with open(idx_dir / "unique_streets.pkl", "rb") as f:
            self.unique_streets: List[str] = pickle.load(f)
        with open(idx_dir / "street_to_rows.pkl", "rb") as f:
            self.street_to_rows: Dict[str, List[int]] = pickle.load(f)
        with open(idx_dir / "street_house_to_rows.pkl", "rb") as f:
            self.street_house_to_rows: Dict[Tuple[str,str], List[int]] = pickle.load(f)
            
        # --- OPTIMIZATION START ---
        print("Building Optimized Postcode Index...")
        # Ensure postnr is string for consistent lookup
        self.rows['postnr'] = self.rows['postnr'].astype(str)
        # Create a dictionary mapping postcode -> list of unique streets in that area
        self.postcode_index = self.rows.groupby('postnr')['vejnavn'].unique().apply(list).to_dict()
        print("Index Built. Retriever Ready.")
        # --- OPTIMIZATION END ---

    def _display_line(self, i: int) -> str:
        r = self.rows.iloc[i]
        return f"{r['vejnavn']} {r['husnr']}, {r['postnr']} {r['postnrnavn']}".strip()

    def _adjust(self, street_name: str, base: float, q_norm: str) -> float:
        L = len(street_name)
        score = base
        if L <= 3: score -= 25
        elif L <= 5: score -= 10
        cand_norm = fold_for_match(street_name)
        if cand_norm[:4] == q_norm[:4]:
            score += 5
        return score

    def retrieve(
        self,
        raw: str,
        topk_streets: int = 50,
        topk_output: int = 100,
        score_cutoff: int = 72,
        ensure_min3: bool = True,
        scorer = fuzz.WRatio
    ) -> List[Tuple[float, str, int]]:
        q = parse_raw(raw)
        q_norm = fold_for_match(q["street"])
        if not q_norm:
            return []

        # --- OPTIMIZED SEARCH STRATEGY ---
        candidates = self.unique_streets # Default to full search
        using_optimization = False

        # If we found a postcode, try to search ONLY streets in that postcode first
        if q["postnr"] and q["postnr"] in self.postcode_index:
            local_streets = self.postcode_index[q["postnr"]]
            # Only use optimization if the postcode actually has streets
            if len(local_streets) > 0:
                candidates = local_streets
                using_optimization = True

        # Perform the fuzzy extraction
        m_streets = process.extract(q_norm, candidates, scorer=scorer, limit=topk_streets, score_cutoff=score_cutoff)

        # Fallback: If optimization yielded no results (bad postcode?), revert to full search
        if using_optimization and not m_streets:
            candidates = self.unique_streets
            m_streets = process.extract(q_norm, candidates, scorer=scorer, limit=topk_streets, score_cutoff=score_cutoff)
        # -------------------------------

        scored: List[Tuple[float,int]] = []
        street_scores: Dict[int, float] = {} 

        for street_norm, street_score, _ in m_streets:
            # Note: street_norm might be raw string if using optimization, or normalized if using unique_streets
            # We map back using the standard lookup which expects normalized keys?
            # Wait - self.street_to_rows keys are normalized.
            # If candidates are raw strings (from dataframe), we must normalize them to look up rows.
            
            lookup_key = street_norm
            if using_optimization:
                 lookup_key = fold_for_match(street_norm)

            for i in self.street_to_rows.get(lookup_key, []):
                row = self.rows.iloc[i]
                total_score = street_score
                street_scores[i] = street_score

                if street_score >= 95:
                    total_score += 20
                    if q["postnr"] and str(row["postnr"]) == q["postnr"]:
                        total_score += 50

                total_score = self._adjust(row["vejnavn"], total_score, q_norm)

                if q["house"] and row["husnr"].lower() == q["house"]:
                    total_score += 50
                elif q["house"]:
                    try:
                        query_num = int(''.join(c for c in q["house"] if c.isdigit()))
                        row_num = int(''.join(c for c in row["husnr"] if c.isdigit()))
                        if abs(query_num - row_num) <= 5:
                            total_score += 10
                        elif abs(query_num - row_num) <= 10:
                            total_score += 5
                    except (ValueError, AttributeError):
                        pass

                scored.append((total_score, i))

        scored.sort(key=lambda t: t[0], reverse=True)

        postcode_groups = {}
        for score, i in scored:
            row = self.rows.iloc[i]
            postnr = str(row["postnr"])
            if postnr not in postcode_groups or score > postcode_groups[postnr][0]:
                postcode_groups[postnr] = (score, i)

        sorted_postcodes = sorted(postcode_groups.items(), 
                                key=lambda x: (x[1][0] + (100 if q["postnr"] and x[0] == q["postnr"] else 0)), 
                                reverse=True)

        seen = set()
        out: List[Tuple[float, str]] = []
        for postnr, (score, i) in sorted_postcodes:
            line = self._display_line(i)
            if line not in seen:
                seen.add(line)
                row = self.rows.iloc[i]
                precision = 0.0
                
                street_match = fold_for_match(row["vejnavn"]) == q_norm
                house_match = q["house"] and row["husnr"].lower() == q["house"]
                postcode_match = q["postnr"] and str(row["postnr"]) == q["postnr"]
                
                if street_match and house_match and postcode_match:
                    precision = 100.0
                elif street_match and postcode_match:
                    precision = 90.0
                elif street_match and house_match:
                    precision = 80.0
                elif street_match:
                    precision = 70.0
                elif postcode_match:
                    precision = 60.0
                else:
                    base_street_score = street_scores.get(i, 0)
                    precision = max(30.0, min(65.0, base_street_score * 0.8))
                
                out.append((precision, line, i))
            if len(out) >= topk_output:
                break

        return out[:topk_output]