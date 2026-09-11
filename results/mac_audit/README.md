# Mac-side audit runs — per-seed raw archive

This directory is the **verbatim per-seed output** of the audit/ablation runs that were
executed on the Mac (M2) harness. Those runs are the evidence behind the reports in
`analysis/wfast_geom/report_output/` and the claims graded in `docs/proven_claims.md`,
but until now only the *summaries* were archived: the reports cite paths such as
`~/llm-lab/dla_audit/nocons/` while the JSON behind them lived outside the repository.
This directory closes that gap, so that "the per-seed data, the scripts and the commit are
all in the repository" is now literally true.

Nothing here is a re-run or a re-analysis: every file is a byte-for-byte copy of the run
output, with one documented exception (path redaction, see *Redaction* below).

## Layout

Paths mirror the on-disk locations the reports refer to (`~/llm-lab/X` → `results/mac_audit/X`),
so a provenance line in a report can be resolved to a file without a lookup table:

```
results/mac_audit/
├── dla_audit/            # audit_b3.py — allocation selectivity, write-strength (γ) and sleep-decay (δ) sweeps
├── dla_audit_second/     # audit_second.py — second-backbone replication (n=9)
├── dla_audit_t10/        # t10_forgetting.py — 10-task continual learning (n=8 x 3 arms)
└── dla_audit_t10_timing.log
```

Per-seed files are `probe_seed{S}_{ORDER}.json` (ablation arms; `ORDER ∈ {HE, EH}`) or
`t10_seed{S}_{variant}.json` (long-horizon arms). Every JSON carries its own `seed`, `order`
and `variant`, so a file's arm can always be checked against its directory.

## Coverage, arm by arm

| arm (path in this archive) | variant | n | seeds | report / claim it backs |
|---|---|---|---|---|
| `dla_audit/direct/` | `direct` (γ_scale=1.0 default) | 12 HE + 12 EH | 0–11 | `audit_batch2_retention_alloc.md`, `selectivity_test_n12.md`, `proven_claims.md` §1 |
| `dla_audit/nocons/` | no write (floor) | 12 HE + 12 EH | 0–11 | same; the floor every arm is anchored to |
| `dla_audit/nosleep/` | no task boundary | 12 HE | 0–11 | `audit_batch2_retention_alloc.md` (retention, t=+10.26), `proven_claims.md` §4 |
| `dla_audit/shufwrite/` | energy-matched coordinate shuffle | 12 HE + 12 EH | 0–11 | the central contrast (`proven_claims.md` §1, §2) |
| `dla_audit/uniformwrite/` | uniform write | 12 HE | 0–11 | `audit_batch2_retention_alloc.md`, `mechanism_audit.md` §3.2 |
| `dla_audit/topwrite/` | top-20% \|W_fast\| write | 12 HE | 0–11 | `audit_batch2_retention_alloc.md`; **known implementation defect** (sign dropped — `mechanism_audit.md` §3.2) |
| `dla_audit/qonly/` | Q path only | 4 HE + 4 EH | 0–3 | `proven_claims.md` §6 (negative: Q is causally inert) |
| `dla_audit/full/` | full history writeback | 1 HE + 1 EH | 0 | comparison arm; **only seed 0 here** — the rest of this arm is the server archive |
| `dla_audit/g0.5/{direct,shufwrite}/` | γ_scale = 0.5 | 12 HE each | 0–11 | `a1_gamma_final.md` §2 dose–response |
| `dla_audit/g1.5/{direct,shufwrite}/` | γ_scale = 1.5 | 12 HE each | 0–11 | `a1_gamma_final.md` §2 |
| `dla_audit/g0.05/{direct,shufwrite}/` | γ_scale = 0.05 | 3 HE each | 0–2 | weak-write tail point, **n=3 only** (`a1_gamma_prelim.md`) |
| `dla_audit/d0.25/direct/`, `dla_audit/d0.75/direct/` | sleep decay δ = 0.25 / 0.75 | 6 HE each | 0–5 | `a1_gamma_final.md` δ sweep |
| `dla_audit_second/` | W_fast swap, second backbone | 9 | 0–8 | `a5_second_backbone.md`, `proven_claims.md` §3 |
| `dla_audit_t10/` | `direct` / `nocons` / `nosleep` | 8 seeds × 3 arms | 0–7 | `t10_continual_learning.md`, `t10_forgetting/`, `proven_claims.md` §4–5 |

`γ` here is the **scale** passed as `--gamma`, multiplied into the *learned*
`consolidate_fast_direct` coefficient; `γ_scale=1.0` is the project default, not a
coefficient of 1.0 (see README §五 and `a1_gamma_final.md` for the notation).

## Provenance

* **Harness**: Mac (M2, 16 GB), `~/.venv` (torch 2.10). Parity with the server archive was
  validated before these runs (ppl deviation ≈0.24), which is what licenses pooling them
  with the server-side `full` arm.
* **Window**: 2026-09-09 23:05 → 2026-09-10 11:31 (CST). The `a1x` batch finished with the
  sentinel `A1X_DONE` in `dla_audit/a1x_run.log`.
* **Code state**: the run window spans repo commits `a365bcf` → `828c49c`. The *runner*
  (`analysis/wfast_geom/audit_b3.py`, and `audit_second.py` / `t10_forgetting.py` for the
  other two arms) is byte-identical to the version in this commit range, so these JSONs were
  produced by exactly the code now in the repository. The *aggregation* script
  `analysis/wfast_geom/a1_final.py` was extended at `cac3b23` (2026-09-10 13:52) — i.e. after
  the runs but before the report — and that revision is the one that produced
  `a1_gamma_final.md`; no aggregation step is missing from the repository.
* **Reproducing the γ/δ tables**: `a1_final.py` reads a hard-coded `OUT = ~/llm-lab/dla_audit`.
  To re-derive the tables from this archive, either symlink/copy `results/mac_audit/dla_audit`
  to that path or point `OUT` at it — no other change is needed, and no run has to be repeated.
* **Drivers**: the batch scripts that produced these arms are versioned in
  `analysis/wfast_geom/queue/` (`run_a1x.sh`, `run_shuf.sh`, `run_min_b3.sh`, `run_ext.sh`,
  `run_qonly.sh`, `run_sec.sh`, `t10job.sh`, `run_a1.sh`/`v2`/`v3`, `master.sh`/`master2.sh`).
  The `*jobs.txt` files kept beside the logs are the exact per-seed job lists those drivers
  expanded, in `tag|seed|variant|gamma|decay` form — the same job-list-plus-log pair the
  cloud queue README requires, so each result is traceable to the configuration that made it.
* **Logs kept**: `*_launcher.log`, `*_run.log`, `*jobs.txt` are retained as the launch record
  (they are what proves a seed ran to completion, and they carry the per-seed `pre`/`gain40`
  lines). They are stdout/stderr of the runs, not curated text.

## Redaction

The eight `*_run.log` files under `dla_audit/` contained absolute paths with the account name
of the machine that ran them (`/Users/<user>/…`, ~500 occurrences). Those occurrences were
replaced by the literal `$HOME` so that this public repository does not carry a personal
account identifier. **No other byte was changed**: no number, warning, seed line or sentinel
was touched, and the redaction is confined to the path prefix. All JSON output is unmodified.

## Known gaps (deliberate, not oversights)

* `full/` holds seed 0 only — the remaining `full` bodies/probes are the archived server run,
  which is why reports compare against it separately.
* `g0.05` (n=3) and `qonly` (n=4) are small by design; the reports label them as preliminary
  and as a negative result respectively.
* `topwrite` is archived as-is although its implementation drops the sign; the defect is
  documented in `docs/mechanism_audit.md` §3.2 and the arm is not used as evidence for the
  "concentration" reading (`topk_signed` was never run — `docs/proven_claims.md` §四).
* The 10-task run stores no intermediate retention checkpoints (documented in
  `t10_forgetting/`); only end-of-run PPL per task is available.
* Not archived here on purpose: the ~288 MB domain corpus (`data/domains/`, re-downloadable)
  and the raw weights/checkpoints, which are not part of the scientific record.
