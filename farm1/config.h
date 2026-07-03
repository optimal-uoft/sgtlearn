# Configuration file for the current farm

# If WHOLE_NODE=1, use whole nodes only for scheduling
# (This mode should be used on niagara / trillium.)
# Whole node mode only works with serial codes (serial farming)
WHOLE_NODE=0

# How many cores each node has. Matters only if WHOLE_NODE is defined above.
NWHOLE=40

# This input parameter is cluster specific - maximum number of jobs one can submit on the cluster.
# If using the -auto feature and/or using the post-processing job feature,
# leave room for at least one more job.
# E.g., if the cluster job limit is 999, set this to 998 or less.
NJOBS_MAX=998

# Minimum and maximum allowed runtimes for farm jobs (seconds):
RUNTIME_MIN=2
RUNTIME_MAX=3000000

# Case is considered to fail if the actual runtime is shorter than this number of seconds:
dt_failed=2

# The meta-job fails if first N_failed_max cases all fail:
N_failed_max=5


export WHOLE_NODE NWHOLE NJOBS_MAX RUNTIME_MIN RUNTIME_MAX dt_failed N_failed_max
