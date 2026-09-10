"""Stability calibration for the scale probe.

Question: at a given backbone size, for which fast-learning rates eta_fast does the
40-step D-probe stay in a STABLE regime?

This is deliberately NOT an effect search.  It looks only at the shape of the probe
perplexity curve on a single arm with no history: a curve that improves and then
rises again (final >> minimum) means the W_fast updates are diverging, and the
protocol's gain@40 then measures divergence rather than adaptation.  The EH/HE
contrast is never computed here.
"""
import sys

sys.path.insert(0, "/root/dla-v0.2")
sys.path.insert(0, "/root/dla-v0.2/analysis/wfast_geom")
sys.path.insert(0, "/root/llm-lab/nanoGPT")

import torch  # noqa: E402
import audit_scale as S  # noqa: E402

M = sys.argv[1]
ETAS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [
    6e-4, 2e-4, 6e-5, 2e-5]

for eta in ETAS:
    model = S.build(M, "bf16", "cuda", eta_fast=eta)
    ids = S.get_corpus("/root/llm-lab/hf_gpt2/shakespeare_bpe.npy",
                       model.config.vocab_size)
    c = 4000
    Dtr, Dev = ids[2 * c:3 * c], ids[3 * c:4 * c]
    st = model.make_state("cuda")
    pre, curve, gain = S.run_probe(model, st, Dtr, Dev, "cuda", 40, 128, 4,
                                   seed=800000, eval_batches=8)
    p = [round(x["ppl"]) for x in curve]
    lo = min(p)
    diverge = p[-1] > 1.5 * lo
    print(f"eta={eta:8.1e} pre={pre:7.1f} min={lo:6d} final={p[-1]:6d} "
          f"gain={gain:+.4f} {'<<< DIVERGES' if diverge else 'stable'}", flush=True)
    print(f"    {p}", flush=True)
    del model
    torch.cuda.empty_cache()
