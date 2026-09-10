# Scale probe: raw measured results

Backbone family: GPT-2 (English), frozen pretrained weights, DLA mounted on every
Linear. Instance: AutoDL RTX 4090 vGPU-48GB, torch 2.8.0+cu128, bf16 DLA state.
Corpus: 338,025 GPT-2 BPE tokens of Shakespeare.

Protocol (capacity-matched): E/H history chunks 60,000 tokens each, 40 wake steps
per chunk; probe trains 40 steps on `D_train` = 60,000 tokens and perplexity is
measured on a held-out `D_eval` = 20,000 tokens that is never trained on.
`Delta(EH-HE)` = gain@40(EH) - gain@40(HE); `Delta_wfast` is the same contrast when
each arm probes from a FRESH state carrying ONLY that history's `W_fast`.

## Results

| backbone | params | n | Delta(EH-HE) | neg | t | Delta_wfast | t |
|---|---|---|---|---|---|---|---|
| gpt2        | 124M | 12 | **-0.1411 ± 0.0366** | 12/12 | -13.37 | **-0.1461 ± 0.0348** | -14.55 |
| gpt2-medium | 355M | 12 | **-0.1182 ± 0.0477** | 12/12 | -8.59  | **-0.1434 ± 0.0540** | -9.21  |
| gpt2-large  | 774M | 12 | **-0.1662 ± 0.1072** | 12/12 | -5.37  | **-0.1932 ± 0.0998** | -6.71  |
| gpt2-xl     | 1.5B | 3  | **-0.1155 ± 0.0484** | 3/3   | -4.13  | **-0.0913 ± 0.0483** | -3.27  |

Per-seed `Delta(EH-HE)` at 1.5B: -0.1582, -0.0629, -0.1254.

All 39 measured seeds have a negative sign; the effect carries over a 12.5x
parameter range with no systematic trend in magnitude.

## Protocol validity boundary (measured)

The paper's original probe uses `D_train = D_eval = 4,000` tokens. At that size the
40-step probe leaves the stable regime above 124M: the model memorises the probe
train slice and held-out perplexity turns around.

| backbone | held-out ppl minimum at step | value at step 40 | verdict |
|---|---|---|---|
| gpt2 (124M)        | ~28 | 226.9 | valid |
| gpt2-medium (355M) | 12  | 358.6 | **breaks (2.9x worse than minimum)** |
| gpt2-large (774M)  | 12  | 406.1 | **breaks (2.6x worse than minimum)** |

Diagnostic at 774M with `D = 4,000` over the 40 probe steps:

| step | eval_ppl | train_ppl | ‖W_fast‖ |
|---|---|---|---|
| 1  | 392.9 | 606.0 | 50.41 |
| 8  | 166.5 | 118.3 | 50.36 |
| 12 | **154.4** | 91.0 | 51.03 |
| 20 | 230.7 | 65.6 | 52.34 |
| 40 | 406.1 | **17.1** | 52.57 |

Train perplexity falls monotonically to 17.1 while held-out rises after step 12, and
‖W_fast‖ moves by only 4%: this is **overfitting**, not numerical divergence. A
per-scale `eta_fast` calibration (6e-4, 2e-4, 6e-5, 2e-5) with no history showed all
four rates stable (gain +0.94 to +0.96), so the fast-learning rate is not the cause.

## Engineering notes

- Memory: at 1.5B the DLA state is 14.5 GB (5 tensors per layer, 194 layers) and the
  backbone 5.8 GB. `set_dla_state` keeps its argument alive, so allocating the second
  history state before releasing the first spikes to 46.4 GB and overflows a 48 GB
  card. Releasing the model's state reference before each `make_state` keeps the peak
  near 33 GB. Verified numerically identical (774M seed 0: -0.165591 -> -0.1656).
- W_fast snapshots are 1.5 GB at 774M and 3 GB at 1.5B per arm; writing them for a
  whole sweep fills the 30 GB system disk. Outputs go to the 50 GB data disk and
  snapshots are opt-in (`--save-wfast`).
