#!/bin/bash
# A single case computation, for the case No. $2 in the table $1.

TABLE1=$1
i1=$2

# Total number of cases:
## If the env. variable N_cases is defined, using it, otherwise computing the number of lines in the table:
if test -z $N_cases
  then
  N_cases=`cat "$TABLE1" | wc -l`
  fi

# Exiting if $i1 goes beyond $N_cases (can happen in bundled farms):
if test $i1 -lt 1 -o $i1 -gt $N_cases  
  then
  exit
  fi
  
# Extracing the $i1-th line from file $TABLE1:
LINE=`sed -n ${i1}p $TABLE1`
# Case id (from the original cases table):
ID=`echo "$LINE" | cut -d" " -f1`
# The rest of the line:
COMM=`echo "$LINE" | cut -d" " -f2-`

METAJOB_ID=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# ++++++++++++++++++++++  This part can be customized:  ++++++++++++++++++++++++
#  Here:
#  $ID contains the case id from the original table (can be used to provide a unique seed to the code etc)
#  $COMM is the line corresponding to the case $ID in the original table, without the ID field
#  $METAJOB_ID is the jobid for the current meta-job (convenient for creating per-job files)

# If you do not want each case to be computed inside a separate subdirectory, comment out the following two lines
# and also comment out the line "cd .." below!
# mkdir -p RUN$ID
# cd RUN$ID

# echo "Case $ID:"

# Executing the command (a line from table.dat)
# It's allowed to use more than one shell command (separated by semi-columns) on a single line
eval "$COMM"

# Exit status of the code:
STATUS=$?

# Comment out this line if not creating a seoarate subdirectory for each case:
# cd ..

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Saving all current metajob statuses to a single file with a unique name:
echo $ID $STATUS >> STATUSES/status.$METAJOB_ID
