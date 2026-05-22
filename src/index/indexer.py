import json
import os

def build_sample_index():
    here = os.path.dirname(__file__)
    sample = os.path.join(here, "sample_passages.json")
    if os.path.exists(sample):
        print("sample index already exists")
        return
    data = [
        {"id": 0, "text": "Short selling is selling a security you do not own."},
        {"id": 1, "text": "A market order executes immediately at current market prices."},
        {"id": 2, "text": "Diversification reduces portfolio risk."},
    ]
    with open(sample, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("wrote sample_passages.json")
