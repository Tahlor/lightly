#SBATCH --gres=gpu:1
####SBATCH -C 'rhel7&pascal'
#SBATCH --qos=cs
#SBATCH --mem-per-cpu 5000MB
#SBATCH --ntasks 32
#SBATCH --nodes=1
#SBATCH --output="./barlow_twins.slurm"
#SBATCH --time 12:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge

export PATH="/home/tarch/env/de:$PATH"
eval "$(conda shell.bash hook)"
conda activate /home/tarch/env/lightly

# get the PARENT of the folder of THIS script
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
PARENT="$(dirname "$SCRIPT_PATH")"

export COMET_OFFLINE_DIRECTORY="./experiments/comet_offline"
DATA_DIR="/home/tarch/datasets/1950s448"
DATASET_NAME=$(basename $DATA_DIR)
DEST_DIR="/tmp/tarch/$DATASET_NAME"

mkdir -p $DEST_DIR
rsync -a --info=progress2 --timeout=1 $DATA_DIR/ $DEST_DIR

which python
nvidia-smi
python -u $PARENT/master.py --model_name barlowtwins --dataset_path "/tmp/tarch/1950s448" --batch_size 200 \
--epochs 200 --num_workers 30 --max_items 100000
