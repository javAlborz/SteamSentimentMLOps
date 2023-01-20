import os.path
import pytest
#from tests import _PATH_DATA
from src.data.make_dataset import ReviewDataset
#
def test_something_two():
    assert True

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_train_test_size():
    sample_size = 1000
    reviews = ReviewDataset(in_folder='data/raw', out_folder='data/processed',
    name='distilbert-base-uncased', sample_size=sample_size, force=True)
    dataset = reviews.processed
    size = len(dataset['train']) + len(dataset['test']) + len(dataset['valid'])
    assert size == sample_size

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_dataset_columns():
    sample_size = 1000
    reviews = ReviewDataset(in_folder='data/raw', out_folder='data/processed',
    name='distilbert-base-uncased', sample_size=sample_size, force=True)
    dataset = reviews.processed
    columns = set( ['label', 'input_ids', 'attention_mask', 'text'])
    bool_list = [columns == set(dataset[x].column_names) for x in ['train', 'test', 'valid']]
    assert all(bool_list)

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_n_labels():
    sample_size = 1000
    reviews = ReviewDataset(in_folder='data/raw', out_folder='data/processed',
    name='distilbert-base-uncased', sample_size=sample_size, force=True)
    dataset = reviews.processed
    num_labels=2
    total_labels = len(set(dataset['train']['label']))
    assert total_labels == num_labels

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_all_labels_train_test():
    sample_size = 1000
    reviews = ReviewDataset(in_folder='data/raw', out_folder='data/processed',
    name = 'distilbert-base-uncased', sample_size=sample_size, force=True)
    dataset = reviews.processed
    num_labels=2
    bool_list = [num_labels == len(set(dataset[x]['label'])) for x in ['train', 'test', 'valid']]
    assert all(bool_list)