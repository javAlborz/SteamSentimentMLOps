import torch
import pandas as pd
from src.data.make_dataset import ReviewDataset

dataset = ReviewDataset('data/raw','data/processed')

print(dataset.df.head())