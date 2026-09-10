"""Diagnose the 774M probe: overfitting or numerical divergence?

The probe improves held-out D for ~10 steps and then degrades.  Two explanations
predict different train-split behaviour:

  overfitting   : train loss keeps falling while held-out D rises
  divergence    : train loss rises too (the fast weights are blowing up)

This script logs BOTH on the same run.  It computes no EH/HE contrast.
"""
import sys

sys.path.insert(0, "/root/dla-v0.2")
sys.path.insert(0, "/root/dla-v0.2/analysis/wfast_geom")
sys.path.insert(0, "/root/llm-lab/nanoGPT")

import torch  # noqa: E402
import audit_scale as S  # noqa: E402

M = sys.argv[1] if len(sys.argv) > 1 else "gpt2-large"
model = S.build(M, "bf16", "cuda")
ids = S.get_corpus("/root/llm-lab/hf_gpt2/shakespeare_bpe.npy",
                   model.config.vocab_size)
# chunk sizes from argv: history_chunk, d_train, d_eval
hc = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
dtr_n = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
dev_n = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
c = hc
E, H = ids[:c], ids[c:2 * c]
Dtr = ids[2 * c: 2 * c + dtr_n]
Dev = ids[2 * c + dtr_n: 2 * c + dtr_n + dev_n]
print(f"chunks: E/H={c}  D_train={len(Dtr)}  D_eval={len(Dev)}", flush=True)

st = model.make_state("cuda")
S.train_history(model, st, E, "cuda", 40, 128, 4, seed=11)
S.train_history(model, st, H, "cuda", 40, 128, 4, seed=23)

model.set_dla_state(st)
# a FIXED train batch, so its loss is comparable across steps
gen = torch.Generator().manual_seed(12345)
tx, ty = S.make_batch(Dtr, 128, 4, gen)
tx, ty = tx.cuda(), ty.cuda()

with torch.no_grad():
    pre_eval = S.eval_ppl(model, Dev, "cuda", 128, 8, seed=0)
    pre_train = float(model(tx, ty)[1].exp())
print(f"{M}: pre  eval={pre_eval:.1f}  train={pre_train:.1f}", flush=True)

rng = torch.Generator().manual_seed(800000)
print(f"{'step':>4} {'eval_ppl':>10} {'train_ppl':>10} {'|W_fast|':>10}", flush=True)
for t in range(1, 41):
    x, y = S.make_batch(Dtr, 128, 4, rng)
    model.dla_step(x.to("cuda"), y.to("cuda"), st)
    if t % 4 == 0 or t == 1:
        with torch.no_grad():
            ev = S.eval_ppl(model, Dev, "cuda", 128, 8, seed=0)
            tr = float(model(tx, ty)[1].exp())
            wn = (sum(float(s["w_fast"].float().pow(2).sum())
                      for s in st.store.values())) ** 0.5
        print(f"{t:>4} {ev:>10.1f} {tr:>10.1f} {wn:>10.2f}", flush=True)
