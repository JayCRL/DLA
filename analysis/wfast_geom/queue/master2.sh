#!/bin/bash
S="0 1 2 3 4 5 6 7 8 9 10 11"
# 1) 修正协议后的载体检验（4 个规模）
/root/run_job.sh v2 gpt2        $S
/root/run_job.sh v2 gpt2-medium $S
/root/run_job.sh v2 gpt2-large  $S
/root/run_job.sh v2 gpt2-xl     $S
# 2) 选择性写回（核心主张）
/root/run_job.sh alloc gpt2-large $S
/root/run_job.sh alloc gpt2-xl    $S
# 3) CL 基线
/root/run_job.sh cl gpt2 $S
echo MASTER2_DONE
