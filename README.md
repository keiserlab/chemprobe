# chemprobe

## download data
```
bash download_data.sh
```
## preprocess
```
python scripts/preprocess.py \
    --data_path ../data
```

## train
```
python scripts/train.py \
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
python scripts/optimize.py \
    --study_path /srv/danger/scratch/wconnell/chemprobe/optuna/TEST \
    --data_path ../data/preprocessed \
    --exp concat \
    --fold 0 \
    --ntrials 5 \
    --prune \
    --batch_size 16384 \
    --gpus 0,
```

## predict
```
python scripts/predict.py \
    --cpds ceranib-2 CAY10618 \
    --data_path ../data/hani \
    --batch_size 128 \
    --gpus 2,
```