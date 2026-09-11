# Server archive (linghang1) — per-seed results of the CPU era

Verbatim mirror of the archive that lived only on the server
(`wust_1@192.168.2.2:~/llm-lab/dla-v0.2/`). The repository already carried the *derived*
artifacts of these runs — the figures under `results_stage4/`, `results_stage5/`,
`paper/figures/`, and the reports in `docs/experiments.md`,
`analysis/wfast_geom/report_output/` — but not the per-seed JSON they were computed from.
For the paper's `direct ≈ full` comparison that mattered: the `full` baseline *is* this
archive, so the claim was not reproducible from the repository alone. It is now.

## Layout

Paths mirror the server's own tree (`~/llm-lab/dla-v0.2/X` → `results/server_archive/X`):

```
results/server_archive/
├── results/     the server's results/ tree, minus the *.pt checkpoints
│   ├── stage1 … stage8_physics_20      stage experiments, per seed
│   ├── stage55b–stage55e               the 2x2 body/W_fast matrix (see below)
│   ├── validation, validation_baselines, validation_second
│   ├── analysis_wfast/                 P1/P2 geometry, p3, p4, audit_cheap, replays
│   ├── dna/, mech_chain/, quick/, smoke/
│   └── (4 files at the results/ root: effective_rank, wfast_decompose, …)
└── logs/        the server's logs/ tree (126 stage/probe/baseline logs)
```

418 files, ~47 MB. Inventory by group: `analysis_wfast` 83 · `stage7` 23 ·
`stage8_physics_20` 22 · `stage55e` 17 · `smoke` 17 · `validation` 16 · `stage6` 13 ·
`quick` 12 · the remaining stage groups 3–9 each.

## Where the paper's `full` baseline lives

`results/stage55e/seeds/seed{0..11}.json` is the archive the reports call `full`
(`min_b3_results.md` line 5: "`full`: archive (server Stage 5.5e bodies/probes)";
`selectivity_test_n12.md`: "full +0.0143 (archive)"). Each file holds the four
body/W_fast conditions — `HE/HE`, `HE_body+EH_fast`, `EH/EH`, `EH_body+HE_fast` — so the
`full` arm is the `HE/HE` condition.

Checked on archive, not assumed: mean `gain@40` of `HE/HE` over the 12 seeds is **+0.0143**,
matching the number quoted in `selectivity_test_n12.md` exactly. The Mac parity replays
(`analysis/wfast_geom/report_output/mechanism_chain/data/replays/seeds/*_HE_HE.json`, n=12)
give +0.0138 — the same quantity through a different harness, differing by the parity
tolerance (~0.0005) that licenses pooling the two.

## Provenance

* **Machine**: linghang1 (CPU, `wust_1@192.168.2.2`) — the pre-Mac era, before the audits
  moved to the M2 harness.
* **Snapshot**: the server copy was last modified 2026-09-08; the server-side repository
  directory is not a git checkout (no `.git`), so these files are the only copy of that tree
  and are archived here in the state the server held them.
* **Logs**: `logs/` holds the stdout of the runs (stage/probe/baseline/second-setting/dna).
  They are the run record, kept for the same reason the Mac archive keeps `*_run.log`.
* **`.pt` checkpoints are excluded** (3.9 GB: `results/stage55e/bodies/*.pt` and 25 others).
  They are trained learner bodies — inputs to re-running a probe, not the record of a
  result. Re-deriving any number in the reports needs only the JSON archived here.

## Not redacted, deliberately

Some JSON files record absolute paths as *data values* (e.g. corpus and output paths,
`/home/wust_1/llm-lab/...`), and the logs do the same. Unlike the Mac archive's stdout
narration — where the account prefix carried no information and was rewritten to `$HOME` —
here the paths are recorded configuration, so they are kept byte-for-byte. Note that the
repository README already publishes the same server account and its LAN address, and that a
LAN address is not routable from the internet.

## Known gaps

* The `mech_chain/wfast_stats.json` (39 MB) is included by explicit decision. The repository
  also keeps the compact mirror (`analysis/wfast_geom/report_output/mechanism_chain/data/wfast_stats_small.json`)
  that the mechanism-chain figures were built from; the full statistics are the raw form.
* `smoke/` and `quick/` are the throwaway probe runs of the era, archived for completeness
  rather than as evidence.
* 14 of the server's figures have no same-named file in the repository (e.g.
  `stage1_curves.png`, `dna_forgetting.png`, `stage3_lambda.png`); the rest are duplicates of
  figures already versioned under `results_stage5/` or `paper/figures/`.
* Stage-4 numbers were produced before the harness fixes noted in the reports; where a report
  marks a result as superseded, this archive does not change that.
