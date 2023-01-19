#!/bin/bash
dvc pull
wandb login $1
python -u src/models/train_model.py $2

python3.8 -u src/data/make_dataset.py
python3.8 -u src/models/train_model.py
gsutil -m cp -r outputs gs://might-as-well-see-what-happens