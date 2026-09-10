P1  write-allocation selectivity -- gpt2, n=12 seeds
================================================================================================

Per-arm levels (a relative gain rewards whichever arm starts worst)
  arm                   pre PPL         final PPL              drop            gain40
  direct         55.49 ±  6.26    40.92 ±  4.43    14.56 ±  7.89  +0.2529 ± 0.1268
  shufwrite      55.16 ±  6.15    41.97 ±  4.84    13.19 ±  7.83  +0.2300 ± 0.1303
  nocons         54.24 ±  6.27    41.80 ±  4.67    12.44 ±  7.87  +0.2194 ± 0.1328

================================================================================================
direct  vs  shufwrite     (positive = direct better)
================================================================================================
  metric                  effect       t               95% CI      d    wins   pre imb  corr(pre,eff)
  final_ppl     +1.0457 ±  0.6103   +5.94 [+0.7260,+1.3738]  +1.71  11/12     +0.326         +0.158
  drop          +1.3713 ±  2.1034   +2.26 [+0.2108,+2.4867]  +0.65   8/12     +0.326         +0.958
  gain40        +0.0229 ±  0.0289   +2.74 [+0.0068,+0.0381]  +0.79   8/12     +0.326         +0.916
  -> metrics AGREE on sign  (claim is robust)
  -> gain40 at EQUAL pre (slope=+0.0138): raw=+0.0229 -> adjusted=+0.0184  t=+6.82

================================================================================================
direct  vs  nocons     (positive = direct better)
================================================================================================
  metric                  effect       t               95% CI      d    wins   pre imb  corr(pre,eff)
  final_ppl     +0.8710 ±  0.3992   +7.56 [+0.6572,+1.0819]  +2.18  12/12     +1.248         +0.084
  drop          +2.1185 ±  1.8400   +3.99 [+1.1004,+3.1139]  +1.15  10/12     +1.248         +0.976
  gain40        +0.0334 ±  0.0243   +4.78 [+0.0203,+0.0465]  +1.38  11/12     +1.248         +0.909
  -> metrics AGREE on sign  (claim is robust)
  -> gain40 at EQUAL pre (slope=+0.0125): raw=+0.0334 -> adjusted=+0.0178  t=+11.46

================================================================================================
shufwrite  vs  nocons     (positive = shufwrite better)
================================================================================================
  metric                  effect       t               95% CI      d    wins   pre imb  corr(pre,eff)
  final_ppl     -0.1747 ±  0.2733   -2.21 [-0.3264,-0.0298]  -0.64   3/12     +0.922         +0.142
  drop          +0.7473 ±  0.6680   +3.88 [+0.4042,+1.1215]  +1.12  11/12     +0.922         +0.914
  gain40        +0.0106 ±  0.0103   +3.54 [+0.0051,+0.0163]  +1.02  11/12     +0.922         +0.893
  -> metrics DISAGREE on sign  (claim is metric-dependent)
  -> gain40 at EQUAL pre (slope=+0.0161): raw=+0.0106 -> adjusted=-0.0043  t=+7.87

================================================================================================
Verdict
================================================================================================
  direct     > shufwrite  on all three raw metrics : True   (t = +5.94, +2.26, +2.74)
                          at equal pre             : yes  (adjusted gain40 = +0.0184)
  direct     > nocons     on all three raw metrics : True   (t = +7.56, +3.99, +4.78)
                          at equal pre             : yes  (adjusted gain40 = +0.0178)
  shufwrite  > nocons     on all three raw metrics : False  (t = -2.21, +3.88, +3.54)
                          at equal pre             : NO  (adjusted gain40 = -0.0043)

  direct > shufwrite is the allocation claim: identical write energy, only the
  coordinates differ.  It survives every metric and the level correction.
  shufwrite vs nocons is the stronger claim 'a random write is as bad as no
  write'; on the raw metrics it flips sign, and only the equal-pre correction
  makes all three agree -- so report it as 'at least as bad', with the
  correction stated, not as a clean equality.
