"""Explore the BOAMP open-data API: print the schema and 3 sample records."""
import json

import requests

BASE = "https://boamp-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/boamp"

# 1. Dataset schema: every field name, type and description
meta = requests.get(BASE, timeout=30)
print("SCHEMA STATUS:", meta.status_code)
for f in meta.json().get("fields", []):
    print(f"- {f.get('name')} ({f.get('type')}): {f.get('description') or f.get('label')}")

# 2. Three recent records, every field truncated to 300 characters
r = requests.get(f"{BASE}/records", params={"limit": 3}, timeout=30)
print("\nRECORDS STATUS:", r.status_code)
data = r.json()
print("TOTAL RECORDS:", data.get("total_count"))
for i, rec in enumerate(data.get("results", []), 1):
    print(f"\n===== RECORD {i} =====")
    for k, v in rec.items():
        print(f"{k}: {json.dumps(v, ensure_ascii=False)[:300]}")
