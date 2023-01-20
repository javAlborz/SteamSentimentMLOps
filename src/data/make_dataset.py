# -*- coding: utf-8 -*-
import logging
import re
import click
#from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

#import os
import pandas as pd
#import numpy as np
#import torch
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
#from datasets import load_dataset, Valus


MODEL_CKPT = "bert-base-uncased"
SAMPLE_SIZE = 100


class ReviewDataset:

    def __init__(self, in_folder: str = '', out_folder: str = '', name='', sample_size=SAMPLE_SIZE, force=False):
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
        self.df = pd.read_csv(in_folder + '/dataset.csv',
                              usecols=['review_text', 'review_score'])
        self.df = self.df.rename(
            columns={"review_text": "text", "review_score": "label"})
        self.df = self.df[~self.df.text.isin(['nan'])]
        self.df = self.df[self.df['label'].notnull()]
        self.df = self.df[self.df.text != "Early Access Review"]
        self.df = self.df.sample(n=sample_size)
        # preprocessing dataset in pandas
        self.df.text = self.df.text.astype(str)
        self.df.label = self.df.label.astype(int)
        self.df['text'] = self.df['text'].apply(lambda x: x.strip())
        # np.where(self.df["label"]==-1, 0, self.df["label"])
        self.df["label"] = self.df["label"].apply(
            lambda x: 'pos' if x > 0 else 'neg')
        self.df['text'] = self.df['text'].apply(
            lambda x: re.sub(r"[♥]+", ' **** ', x))

        # converting to Dataset
        self.ds = Dataset.from_pandas(self.df).remove_columns(['__index_level_0__']).cast_column(
            "label", ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None))
        #new_features = self.ds.features.copy()
        #new_features["label"] = ClassLabel(num_classes=2, names=['neg', 'pos'], names_file=None, id=None)
        #self.ds = self.ds.cast(new_features)

        # splitting train, test, validation
        train_testvalid = self.ds.train_test_split(test_size=0.4)
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        self.processed = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'valid': test_valid['train']})
        # , remove_columns = ['__index_level_0__'])
        self.processed = self.processed.map(
            self.tokenize, batched=True, batch_size=None)

        self.processed.save_to_disk(self.out_folder)

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath = '', output_filepath = ''):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    ReviewDataset(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
