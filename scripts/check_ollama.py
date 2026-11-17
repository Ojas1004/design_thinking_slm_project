"""Ask the Ollama model one test question per task and compare with the reference answer.

usage (from the repo root): python scripts/check_ollama.py [model] [test.jsonl]   (needs `ollama serve` running)
"""
import json
import sys
import urllib.request

model = sys.argv[1] if len(sys.argv) > 1 else "energy-slm"
path = sys.argv[2] if len(sys.argv) > 2 else "data/energydata_sml_large_test.jsonl"

seen, rows = set(), []
for line in open(path):
    r = json.loads(line)
    key = (r["task"], r.get("source", "uci"))
    if key not in seen:
        seen.add(key)
        rows.append(r)

same = 0
for r in rows:
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": r["prompt"]}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", body, {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=120))["message"]["content"].strip()
    ok = reply == r["completion"].strip()
    same += ok
    print(f"[{r['task']} / {r.get('source', 'uci')}] {'MATCH' if ok else 'differs'}")
    if not ok:
        print("  model:", reply[:300])
        print("  ref  :", r["completion"].strip()[:300])
print(f"\n{same}/{len(rows)} identical to the reference")
