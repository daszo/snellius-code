#!/bin/env/usr bash
#
# 1. Submit the primary task and capture its Job ID
# --parsable returns only the integer ID
FIRST_JOB_ID=$(sbatch --parsable pre_pipeline.sh)
echo "Submitted primary job: $FIRST_JOB_ID"

# 2. Submit two new tasks that depend on the success (afterok) of the first
# Each will be its own separate job with its own GPU allocation
SECOND_JOB_ID=$(sbatch --parsable --dependency=afterok:$FIRST_JOB_ID task_A.sh)
THIRD_JOB_ID=$(sbatch --parsable --dependency=afterok:$FIRST_JOB_ID task_B.sh)

echo "Submitted dependent job A: $SECOND_JOB_ID"
echo "Submitted dependent job B: $THIRD_JOB_ID"
