#!/bin/bash
dvc pull
wandb login $1
python3.9 -u src/data/make_dataset.py $2
python3.9 -u src/models/train_model.py
gsutil -m cp -r outputs gs://might-as-well-see-what-happens