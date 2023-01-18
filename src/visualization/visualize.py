import torch
import hydra
import os
import numpy as np
from hydra.core.config_store import ConfigStore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoTokenizer, AutoModel, AutoConfig
from src.models.model import SteamModel, SteamConfig
from src.models.config import SteamConfigClass
from src.data.make_dataset import ReviewDataset

cs = ConfigStore.instance()
cs.store(name='steam_config', node = SteamConfigClass)
cwd = os.getcwd()
@hydra.main(config_path=cwd+'\src\models\conf', config_name='config.yaml')
def visualize(cfg:SteamConfigClass) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    AutoConfig.register("SteamModel", SteamConfig)
    AutoModel.register(SteamConfig, SteamModel)
    model = AutoModel.from_pretrained(cwd+"\models", cfg.params.model_ckpt, cfg.params.num_labels)
    tokenizer =  AutoTokenizer.from_pretrained(cwd+'\models')

    processed_data = ReviewDataset(cfg.paths.in_folder, cfg.paths.out_folder, name=cfg.params.model_ckpt, sample_size=cfg.params.sample_size, force=True)
    emotions_encoded = processed_data.processed

    lists = []
    test_set =torch.tensor(emotions_encoded['test']['input_ids']).to(device)
    loader = DataLoader(test_set, batch_size=8)
    for batch in loader:
        model_predictions = model(batch)
        y_preds = torch.argmax(model_predictions['logits'],1).to('cpu').numpy()
        lists.append(y_preds)
    preds = np.concatenate(lists)

    plt.rcParams["figure.figsize"] = (10,10)
    confusion_mat = confusion_matrix(emotions_encoded['test']['label'], preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    disp.plot()
    plt.savefig(cwd+"/reports/figures/confusion_matrix.png")


if __name__ == "__main__":
    visualize()