"""Where does the memory go at 1.5B?  Measure, do not guess."""
import sys

sys.path.insert(0, "/root/dla-v0.2")
sys.path.insert(0, "/root/dla-v0.2/analysis/wfast_geom")
sys.path.insert(0, "/root/llm-lab/nanoGPT")

import torch  # noqa: E402
import audit_scale as S  # noqa: E402

GB = 1024 ** 3


def rss(tag):
    a = torch.cuda.memory_allocated() / GB
    r = torch.cuda.memory_reserved() / GB
    print(f"  {tag:34s} allocated={a:6.2f}G reserved={r:6.2f}G", flush=True)


M = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 4

rss("start")
model = S.build(M, "bf16", "cuda")
rss("after build (backbone)")

ids = S.get_corpus("/root/llm-lab/hf_gpt2/shakespeare_bpe.npy",
                   model.config.vocab_size)
st = model.make_state("cuda")
rss("after make_state (5 tensors/layer)")

n_par = sum(p.numel() for p in model.parameters())
print(f"  backbone params={n_par/1e6:.0f}M  state tensors={len(st.store)}")
print(f"  est state bytes = {5 * n_par * 2 / GB:.2f}G (bf16)", flush=True)

model.set_dla_state(st)
c = 60000
Dtr = ids[2 * c:3 * c]
x, y = S.make_batch(Dtr, 128, batch, torch.Generator().manual_seed(0))
x, y = x.cuda(), y.cuda()
rss("after batch")
model.dla_step(x, y, st)
rss("after 1 dla_step")
for _ in range(3):
    model.dla_step(x, y, st)
rss("after 4 dla_step")
