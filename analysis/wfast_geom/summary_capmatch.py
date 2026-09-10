"""Tabulate the capacity-matched scale sweep by backbone size."""
import glob
import json
import statistics as st

BASE = "/root/autodl-tmp/dla_scale/capmatch"
MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")

print(f"{'model':14s} {'n':>3s} {'Delta(EH-HE)':>20s} {'Delta_wfast':>20s} {'neg':>6s}")
print("-" * 70)
for M in MODELS:
    ps = sorted(glob.glob(f"{BASE}/{M}/*.json"))
    if not ps:
        continue
    ds, dw = [], []
    for p in ps:
        d = json.load(open(p))
        ds.append(d["delta"])
        dw.append(d["delta_wfast"])
    n = len(ds)
    sd = st.stdev(ds) if n > 1 else 0.0
    sdw = st.stdev(dw) if n > 1 else 0.0
    neg = sum(1 for d in ds if d < 0)
    print(f"{M:14s} {n:3d} {st.mean(ds):+9.4f} +- {sd:6.4f} "
          f"{st.mean(dw):+9.4f} +- {sdw:6.4f} {neg:3d}/{n}")
    if n > 1 and sd > 0:
        t = st.mean(ds) / (sd / n ** 0.5)
        tw = st.mean(dw) / (sdw / n ** 0.5)
        print(f"{'':14s}     t={t:+6.2f} (full state)      t={tw:+6.2f} (W_fast only)")
