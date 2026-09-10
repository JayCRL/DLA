"""Phase-by-phase memory trace of the full scale protocol at 1.5B."""
import sys

sys.path.insert(0, "/root/dla-v0.2")
sys.path.insert(0, "/root/dla-v0.2/analysis/wfast_geom")
sys.path.insert(0, "/root/llm-lab/nanoGPT")

import torch  # noqa: E402
import audit_scale as S  # noqa: E402

GB = 1024 ** 3
PEAK = [0.0]


def rss(tag):
    a = torch.cuda.memory_allocated() / GB
    r = torch.cuda.memory_reserved() / GB
    PEAK[0] = max(PEAK[0], a)
    print(f"  {tag:38s} alloc={a:6.2f}G resv={r:6.2f}G", flush=True)


M = "gpt2-xl"
model = S.build(M, "bf16", "cuda")
rss("build")
ids = S.get_corpus("/root/llm-lab/hf_gpt2/shakespeare_bpe.npy",
                   model.config.vocab_size)
c = 60000
chunks = {"E": ids[:c], "H": ids[c:2 * c]}
d_train = ids[2 * c:3 * c]
d_eval = ids[2 * c + 60000:2 * c + 60000 + 20000]

snaps = {}
for arm in ("EH", "HE"):
    print(f"--- arm {arm} ---", flush=True)
    state = model.make_state("cuda")
    rss(f"{arm}: make_state")
    for tag in arm:
        S.train_history(model, state, chunks[tag], "cuda", 40, 128, 4, seed=11)
        rss(f"{arm}: after history {tag}")
    snaps[arm] = S.snapshot_fast(state)
    rss(f"{arm}: snapshot (kept)")
    pre, curve, gain = S.run_probe(model, state, d_train, d_eval, "cuda", 40,
                                   128, 4, seed=800000, eval_batches=8)
    rss(f"{arm}: after full probe")
    del state
    torch.cuda.empty_cache()
    rss(f"{arm}: state freed")

for arm in ("EH", "HE"):
    st = model.make_state("cuda")
    S.inject_fast(st, snaps[arm])
    rss(f"{arm}_wfast: make+inject")
    pre, curve, gain = S.run_probe(model, st, d_train, d_eval, "cuda", 40,
                                   128, 4, seed=800000, eval_batches=8)
    rss(f"{arm}_wfast: after probe")
    del st
    torch.cuda.empty_cache()

print(f"  PEAK allocated = {PEAK[0]:.2f} G", flush=True)
