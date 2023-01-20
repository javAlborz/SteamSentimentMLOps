#!/bin/bash
echo "step 1"
dvc pull
echo "step 2"
wandb login $1
echo "step 3"
python3.9 -u src/data/make_dataset.py $2
echo "step 4"
python3.9 -u src/models/train_model.py
echo "step 5"
gsutil -m cp -r outputs gs://might-as-well-see-what-happens
echo "step 6"