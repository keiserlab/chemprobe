# chemprobe

## download data
```
bash download_data.sh
```
## preprocess
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
    --cpds ceranib-2 CAY10618 \
    --data_path ../data/hani \
    --batch_size 128 \
    --gpus 2,
```