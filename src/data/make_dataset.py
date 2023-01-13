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
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Value, load_from_disk


MODEL_CKPT =  "distilbert-base-uncased"
SAMPLE_SIZE = 10000
OUT_FILE = '/preprocessed'

class ReviewDataset:

    def __init__(self, in_folder: str = '', out_folder: str = '', name ='', force = False):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.in_folder = in_folder
        self.out_folder = out_folder

        if self.out_folder and not force:  # try loading from proprocessed
            try:
                self.processed = load_from_disk(out_folder)
                print("Loaded from pre-processed files")
                return
            except ValueError:  # not created yet, we create instead
                pass
            

        self.df= pd.read_csv(in_folder +'\dataset.csv', usecols = ['review_text', 'review_score'], nrows = SAMPLE_SIZE)

        #preprocessing dataset in pandas
        self.df = self.df.rename(columns={"review_text": "text", "review_score": "labels"})
        self.df.text = self.df.text.astype(str)
        self.df = self.df[self.df['labels'].notnull()]
        self.df['text'] = self.df['text'].apply(lambda x: x.strip())
        self.df["labels"] = np.where(self.df["labels"]==-1, 0, self.df["labels"])
        self.df = self.df[self.df.text != "Early Access Review"]
        self.df = self.df[~self.df.text.isin(['nan'])]
        self.df['text'] = self.df['text'].apply(lambda x: re.sub(r"[♥]+", ' **** ' ,x))

        #converting to Dataset
        self.ds = Dataset.from_pandas(self.df)
        new_features = self.ds.features.copy()
        new_features["labels"] = ClassLabel(names=[0, 1])
        self.ds = self.ds.cast(new_features)

        #splitting train, test, validation
        train_testvalid = self.ds.train_test_split(test_size=0.4)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        self.processed = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']})  
        print(self.processed)
        self.processed = self.processed.map(self.tokenize, batched=True, batch_size=None, remove_columns = ['__index_level_0__'])
        print(self.processed)

        self.processed.save_to_disk(self.out_folder)



    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True)



    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    dataset = ReviewDataset(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
