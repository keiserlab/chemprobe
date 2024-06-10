# chemprobe

## installation
Requires `python<=3.11`

To access model & dataset modules:
```
pip install chemprobe
```
To access scripts for preprocessing/training/inference you must install from source:
```
git clone https://github.com/keiserlab/chemprobe.git
cd chemprobe
# use a virutal env...
# conda activate env
pip install .
```

## download data
Required to run inference
```
bash download_data.sh
```

## preprocess
Required to run inference
```
python preprocess.py \
    --data_path ../data
```

## train
```
python train.py \
    --name TEST \
    --exp film \
    --fold 0 \
    --n_blocks 4 \
    --data_path ../data/preprocessed \
    --batch_size 16384 \
    --gpus 0,1,2,3 \
    --num_workers 4 \
    --lr 1e-3
```

permuted label model
```
python train.py \
--study_path /scratch/wconnell/danger/chemprobe/optuna/exp=film/fold=0/ \
--data_path ../data/preprocessed \
--name perm-fold=0 \
--exp film \
--fold 0 \
--max_epochs 5 \
--batch_size 16384 \
--gpus 3, \
--permute_labels
```

## optimize
```
python optimize.py \
    --study_path /srv/danger/scratch/wconnell/chemprobe/optuna/ \
    --data_path ../data/preprocessed \
    --exp film \
    --fold 0 \
    --n_trials 20 \
    --prune \
    --batch_size 16384 \
    --gpus 1,
```

## predict
```
python predict.py \
    --cpds neratinib \ # Do not specify to run on all compounds
    --data_path ../data/your_data/ \
    --attribute \ # optional to run attribution
    --batch_size 128 \
    --gpus 2,
```