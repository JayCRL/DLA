# A5 — second-backbone (Shakespeare char GPT) carrier replication, n=9

10.65M Shakespeare char GPT (vocab 65); 3-chunk EH/HE histories (span 26k, nine
disjoint seeds); unseen-slice probe; arms: HE native, HE + EH `W_fast` (destructive
swap), HE + energy-matched shuffled EH `W_fast`. Paired gain@40 (n=9):

| contrast | mean Δ | sd | t |
|---|---|---|---|
| HE+EH vs HE | −0.0673 | 0.0283 | −7.13 |
| HE+shuffled-EH vs HE | −0.0721 | 0.0285 | −7.60 |
| HE+EH vs HE+shuffled-EH | +0.0048 | 0.0030 | +4.83 |

HE native +0.0706±0.0294 → +0.0033±0.0032 after swap (every seed), and the
shuffled injection is at least as destructive. Same signature as the primary
setting (not magnitude, not exact structure), now on a second corpus/backbone at
n=9 instead of n=2.
