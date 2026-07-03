#!/bin/bash
# This is an auxiliary job which runs after the farm is done. It is used
# for auto farm resubmission (the option -auto for submit.run), and for running an optional
# post-processing job (when the job script file final.sh is present in the root farm directory).

#  You have to replace Your_account_name below with the name of your account:
#SBATCH --account=def-coheneld

#SBATCH --time=0-00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M

# Don't change anything below this line

module load meta-farm

autojob.run
