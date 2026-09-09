# Minimal B3 ablation results — which consolidation pathway carries the history effect

Machine: Mac M2 (parity-validated harness). n=4 seeds (0–3) × EH/HE.
Variants (only the sleep write term is changed; everything else identical to baseline):
- `full`  : archive (server Stage 5.5e bodies/probes, seeds 0–3)
- `direct`: sleep writes only γ·W_fast (Q→W_slow off)
- `nocons`: sleep writes nothing (both off; decays & moment resets kept)
- `qonly` : sleep writes only β·Q (direct bypass off) — added because nocons ≠ direct

## A. gain@40 per arm (mean ± sd over 4 seeds)

| variant | EH | HE | HE − EH (paired) | dz |
|---|---|---|---|---|
| full | +0.0006 ± 0.0080 | +0.0239 ± 0.0081 | +0.0232 ± 0.0136 | +1.71 |
| direct | −0.0001 ± 0.0066 | +0.0229 ± 0.0102 | +0.0230 ± 0.0137 | +1.68 |
| nocons | −0.0041 ± 0.0100 | +0.0103 ± 0.0048 | +0.0144 ± 0.0113 | +1.27 |
| qonly | −0.0061 ± 0.0055 | +0.0094 ± 0.0022 | +0.0155 ± 0.0067 | +2.31 |

HE arm per-seed Δ vs full: `direct −0.0010±0.0021 (dz −0.46)` · `nocons −0.0136±0.0043 (dz −3.19)` · `qonly −0.0145±0.0059 (dz −2.43)`.

## B. Consolidation write magnitudes (mean over the 3-sleep history, seeds 0–3)

| variant | Σ‖β·Q‖ per history | Σ‖γ·W_fast‖ per history | post-history ‖Q‖ | ‖W_fast‖ | ‖W_slow‖ |
|---|---|---|---|---|---|
| direct | 0 (off) | ≈3.01–3.11 | 0.0024 | 3.71 | 120.3 |
| nocons | 0 | 0 | 0.0024 | 3.72 | 120.1 |
| qonly | ≈0.0066–0.0089 | 0 (off) | 0.0025 | 3.74 | 120.1 |

Q writes stay ≈3 orders below the direct write even when Q is the only channel.

## C. Verdicts on the three comparisons

1. **nocons ≈ direct? No.** HE Δ vs full: −0.0136 (dz −3.19) vs −0.0010 (dz −0.46). The consolidation channel has a real causal role for the HE advantage.
2. **direct ≈ Full? Yes.** HE Δ vs full = −0.0010±0.0021 (dz −0.46) and HE−EH effect +0.0230 vs +0.0232 (dz 1.68 vs 1.71). The consolidation effect is essentially the **direct W_fast→W_slow writeback**.
3. **nocons ≈ Full? No.** HE Δ vs full = −0.0136 (dz −3.19). Turning consolidation off substantially reduces the HE advantage.
4. **qonly ≈ nocons (Q channel adds ~nothing).** HE: +0.0094 vs +0.0103; with direct off, adding Q back does not restore the effect → the Q pathway has **no independent causal contribution** in this setup (consistent with Q writes ≈0.007/history).

Interpretation: with no consolidation the HE−EH effect is still positive (+0.0144, dz 1.27; ~60% of full) → a **wake-dynamics component (W_fast/P) carries part of the history effect**; the direct writeback adds the rest (~0.009 on HE) and is required to reach full level. β·Q is causally inert here.

## D. Caveats
- n=4 seeds, one backbone/setting; dz from paired within-seed differences (large but noisy at n=4).
- Only future adaptation on unseen D measured; retention (old-task forgetting, the Stage-4-style sleep benefit) is NOT tested by this ablation — those probes never sleep.
- Baseline `full` from server archive; nocons/direct/qonly from Mac harness (parity: seed0 full EH −0.0076 vs −0.0066, HE +0.0285 vs +0.0309).
- Per audit rule: no tuning, no seed picking, no post-hoc metric; all vs baseline.
