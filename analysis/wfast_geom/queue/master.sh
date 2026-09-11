#!/bin/bash
# 串行跑完全部剩余任务，每个 seed 断点续跑
S="0 1 2 3 4 5 6 7 8 9 10 11"
/root/run_job.sh scale gpt2-large $S
/root/run_job.sh scale gpt2-xl    $S
/root/run_job.sh alloc gpt2-large $S
/root/run_job.sh alloc gpt2-xl    $S
/root/run_job.sh cl    gpt2       $S
echo MASTER_DONE
