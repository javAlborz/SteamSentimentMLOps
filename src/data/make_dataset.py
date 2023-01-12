# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


import os
import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Value


MODEL_CKPT =  "distilbert-base-uncased"
SAMPLE_SIZE = 10000
OUT_FILE = '/preprocessed'

class ReviewDataset:

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

    @classmethod
    def tokenize(cls, batch):
        return ReviewDataset.tokenizer(batch['review_text'], padding=True, truncation=True)


    def __init__(self, in_folder: str = '', out_folder: str = ''):
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.df= pd.read_csv(in_folder +'\dataset.csv', usecols = ['review_text', 'review_score'], nrows = SAMPLE_SIZE)

        #preprocessing dataset in pandas
        self.df.review_text = self.df.review_text.astype(str)
        self.df = self.df[self.df['review_score'].notnull()]
        self.df['review_text'] = self.df['review_text'].apply(lambda x: x.strip())
        self.df["review_score"] = np.where(self.df["review_score"]==-1, 0, self.df["review_score"])
        self.df = self.df[self.df.review_text != "Early Access Review"]
        self.df = self.df[~self.df.review_text.isin(['nan'])]
        self.df['review_text'] = self.df['review_text'].apply(lambda x: re.sub(r"[♥]+", ' **** ' ,x))


        #converting to Dataset
        self.ds = Dataset.from_pandas(self.df)
        new_features = self.ds.features.copy()
        new_features["review_score"] = ClassLabel(names=[0, 1])
        self.ds = self.ds.cast(new_features)

        # 90% train, 10% test + validation
        train_testvalid = self.ds.train_test_split(test_size=0.4)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        self.processed = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})  
        print(self.processed)
        #self.ds = self.ds.train_test_split(test_size=0.2)
        self.processed = self.processed.map(ReviewDataset.tokenize, batched=True, batch_size=None)
        print(self.processed)
        #print(self.processed["train"].column_names)

        self.processed.save_to_disk(self.out_folder)
        #self.processed.to_csv(self.out_folder)


    

    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
