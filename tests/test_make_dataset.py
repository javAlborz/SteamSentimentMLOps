from tests import _PATH_DATA
import os.path
import pytest
from src.data.make_dataset import ReviewDataset

sample_size = 1000
num_labels = 2
dataset = ReviewDataset(in_folder='data/raw', out_folder='data/processed',
    name='distilbert-base-uncased', sample_size=sample_size, force=True)
dataset = dataset.processed

def test_something_two():
    assert True

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_train_test_size(dataset=dataset):

    size = len(dataset['train']) + len(dataset['test']) + len(dataset['valid'])
    assert size == sample_size

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_dataset_columns(dataset=dataset):

    columns = set( ['label', 'input_ids', 'attention_mask', 'text'])
    bool_list = [columns == set(dataset[x].column_names) for x in ['train', 'test', 'valid']]
    assert all(bool_list)

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_n_labels(dataset=dataset, num_labels=num_labels):

    total_labels = len(set(dataset['train']['label']))
    assert total_labels == num_labels

@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_all_labels_train_test(dataset=dataset, num_labels=num_labels):

    bool_list = [num_labels == len(set(dataset[x]['label'])) for x in ['train', 'test', 'valid']]
    assert all(bool_list)