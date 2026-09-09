# A1 preliminary — write-strength robustness of allocation selectivity (HE, seeds 0–2)

direct vs shufwrite gain@40 gap by consolidation strength γ (n=3 seeds):

| γ | direct | shufwrite | gap | nocons(ref) |
|---|---|---|---|---|
| 1.0 (default) | +0.0272 | +0.0097 | **+0.0175** | +0.0119 |
| 0.5 | +0.0123 | +0.0103 | +0.0020 | +0.0119 |
| 0.05 | +0.0118 | +0.0096 | +0.0022 | +0.0119 |

Reading (preliminary, n=3): the allocation-selectivity benefit is **write-strength
dependent** — at the default γ=1.0 the direct-vs-shuffled gap is large; at weak write
strength the system behaves like no-consolidation (direct ≈ shufwrite ≈ nocons).
This is consistent with a mechanism whose benefit scales with how much structure is
written into W_slow, not a knife-edge at one hyperparameter — but n=3 warrants
expansion (seeds 0–11 × γ∈{0.5, 1.0, 1.5}) before any claim. Note default-γ=1 files
for seeds 0–2 were re-run after an earlier overwrite; unified-batch numbers already
reported are unaffected (aggregated before the A1 launch).
