# Cloud queue scripts

These are the exact drivers that were pushed to the AutoDL instance and launched
with `setsid nohup <script>.sh > <script>.log 2>&1 < /dev/null &`. They are kept
in the repository because the queue script, not the analysis script, is what
determines the configuration, the seed list and the output path of a run -- a
result that cannot be traced back to its queue script is not reproducible.

## The scripts that produced the current results

| script | what it ran | output |
|---|---|---|
| `leaksel.sh` | **the decisive selector experiment**: `fast_decay` x `{0.25,0.5,1.0,2.0}` x `selwrite/unifq/shufsel/direct`, 6 seeds | `dla_leaksel/` |
| `leak2d.sh` | 2D `(fast_decay, lambda)` map of the content-write contrast, 6 seeds | `dla_leak2d/` |
| `leakquick.sh` | reduced-config lambda sweep that produced the first lambda result (4 seeds) | `dla_leakquick/` |
| `leakqB.sh` / `leakq.sh` / `leakphase1.sh` | the earlier staged single-knob sweeps | `dla_leakgain/` |
| `leakgate.sh` | the `--write-gate` comparison (superseded: gating `softplus(P)*W_fast` still writes W_fast as content) | `dla_leakgate/` |
| `clgain.sh` | the `(fast_decay, lambda)` DLA variants on the continual-learning benchmark | `dla_cl_gain/` |

## Conventions every queue script follows

* `export OMP_NUM_THREADS=8` / `MKL_NUM_THREADS=8`. Without a thread cap, two
  concurrent jobs each spawn a full thread pool, the machine load reaches ~30,
  and the GPU starves to 5% utilisation -- the jobs look hung while both are
  still running.
* `grep --line-buffered -E "^\[lg\]|Error|Traceback|out of memory"`.
  `grep` block-buffers when its stdout is a file, so a healthy run's log stays
  empty for minutes; and a bare `^\[lg\]` filter discards every traceback, which
  makes a total failure look like a silent no-op.
* Resume by `SKIP` when the per-seed JSON already exists.
* `** FAILED` is printed when the run produced no output file, so a failed seed
  can never be mistaken for a finished one.
