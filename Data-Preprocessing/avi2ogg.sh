#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a multi-core MPI job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
#
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J ConvJPGtoOGG



#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user $sk1846@rit.edu



# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL


# Request 5 hours run time MAX, anything over will be KILLED
#SBATCH -t 05:00:00


# Put the job in the "work" partition and request FOUR cores
# "work" is the default partition so it can be omitted without issue.
#SBATCH -p work -c 40



# Job memory requirements in MB
#SBATCH --mem=30000



# Explicitly state you are a free user
#SBATCH --qos=free


if [[ -z $SLURM_CPUS_ON_NODE ]]; then
    echo "SUBMIT THIS AS A BATCH JOB WITH sbatch <ConvJPGtoOGG>"
    exit 1
fi

module load ffmpeg
function to_ogg()
{
    ffmpeg -threads 1 -y -i "$1" -c:v theora -an -q:v 7 "${1%.avi}.ogv"
}

export -f to_ogg
find ~/skdata/UCF-101/ \*.avi -print0 | xargs -0 -i -P"$SLURM_CPUS_ON_NODE" -n1 bash -c 'to_ogg "{}"' {}
