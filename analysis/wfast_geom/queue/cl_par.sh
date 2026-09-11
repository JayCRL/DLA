#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/run_job.sh cl gpt2 0 1 2 3 4 5 6 7 8 9 10 11
echo CL_PAR_DONE
