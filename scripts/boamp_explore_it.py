"""BOAMP exploration, step 2: notice types, IT categories, and the structure of full notices.

Writes everything to boamp_explore.txt (UTF-8) and full sample notices to data/raw/samples/.
"""
import json
from pathlib import Path

import requests

BASE = "https://boamp-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/boamp"
SINCE = "dateparution >= date'2022-01-01'"
IT_KEYWORDS = ["informat", "logiciel", "numéri", "internet", "site", "donnée",
               "cloud", "réseau", "télécom", "web", "système"]
OUT = Path("boamp_explore.txt")
SAMPLES = Path("data/raw/samples")
SAMPLES.mkdir(parents=True, exist_ok=True)

# Contact details of buyers' staff are public but personal: we never print or store them.
PERSONAL_KEYS = ("TEL", "MEL", "FAX", "CORRESPONDANT", "CONTACT", "ADRESSE", "NOM", "PRENOM")

lines = []


def log(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    lines.append(text)


def get(path, params):
    r = requests.get(f"{BASE}{path}", params=params, timeout=60)
    if r.status_code != 200:
        log(f"ERROR {r.status_code} on {path} {params}: {r.text[:300]}")
        return None
    return r.json()


def walk(obj, prefix=""):
    """Yield (path, value) for every leaf of a nested JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            yield from walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


# 1. Facets (value counts) since 2022
facets = {}
for facet in ["nature", "type_marche", "descripteur_libelle"]:
    data = get("/facets", {"facet": facet, "where": SINCE})
    if not data:
        continue
    log(f"\n===== FACET {facet} (since 2022) =====")
    values = []
    for group in data.get("facets", []):
        for v in group.get("facets", []):
            values.append((v.get("count", 0), v.get("name")))
    values.sort(reverse=True)
    facets[facet] = values
    for count, name in values[:300]:
        log(f"{count:>9}  {name}")

# 2. IT-related descriptors
it_labels = [(c, n) for c, n in facets.get("descripteur_libelle", [])
             if n and any(k in n.lower() for k in IT_KEYWORDS)]
log("\n===== IT-RELATED DESCRIPTORS (since 2022) =====")
for count, name in it_labels:
    log(f"{count:>9}  {name}")

# 3. Full sample notices: for the 2 largest IT descriptors, one call for tenders and one award notice
for _, label in it_labels[:2]:
    for nature in ["APPEL_OFFRE", "ATTRIBUTION"]:
        data = get("/records", {
            "where": f'{SINCE} and nature="{nature}"',
            "refine": f'descripteur_libelle:"{label}"',
            "order_by": "dateparution desc",
            "limit": 1,
        })
        if not data or not data.get("results"):
            log(f"\n(no {nature} sample for '{label}')")
            continue
        rec = data["results"][0]
        rec["donnees"] = parse_json_field(rec.get("donnees"))
        rec["gestion"] = parse_json_field(rec.get("gestion"))
        fname = SAMPLES / f"{nature.lower()}_{rec.get('id')}.json"
        fname.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\n===== SAMPLE {nature} | {label} | {rec.get('idweb')} -> {fname} =====")
        log("objet:", rec.get("objet"))
        log("descripteur_libelle:", rec.get("descripteur_libelle"))
        log("type_marche:", rec.get("type_marche"))
        log("--- donnees: every field path (values cut to 150 characters) ---")
        for path, value in walk(rec["donnees"]):
            last_key = path.split(".")[-1].split("[")[0].upper()
            shown = "[masqué]" if any(k in last_key for k in PERSONAL_KEYS) else str(value)[:150]
            log(f"{path}: {shown}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved to {OUT}")
