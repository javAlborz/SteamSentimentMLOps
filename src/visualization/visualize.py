import os
import torch
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, RocCurveDisplay
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoConfig
from src.models.model import SteamModel, SteamConfig
from src.models.config import SteamConfigClass
from src.data.make_dataset import ReviewDataset

cs = ConfigStore.instance()
cs.store(name='steam_config', node = SteamConfigClass)
cwd = os.getcwd()
@hydra.main(config_path=cwd+'/src/models/conf', config_name='config.yaml')
def visualize(cfg:SteamConfigClass) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    AutoConfig.register("SteamModel", SteamConfig)
    AutoModel.register(SteamConfig, SteamModel)
    model = AutoModel.from_pretrained(cwd+"/models", cfg.params.model_ckpt, cfg.params.num_labels)

    processed_data = ReviewDataset(cfg.paths.in_folder, cfg.paths.out_folder, name=cfg.params.model_ckpt, sample_size=cfg.params.sample_size, force=True)
    emotions_encoded = processed_data.processed

    list_preds, list_probas = [], []
    test_set =torch.tensor(emotions_encoded['test']['input_ids']).to(device)
    loader = DataLoader(test_set, batch_size=cfg.params.batch_size)
    for batch in loader:
        model_predictions = model(batch)
        y_preds = torch.argmax(model_predictions['logits'],1).to('cpu').numpy()
        y_probas = torch.nn.functional.softmax(model_predictions['logits'], dim=-1).to('cpu')[:,1].detach().numpy()
        list_preds.append(y_preds)
        list_probas.append(y_probas)
    preds = np.concatenate(list_preds)
    probas = np.concatenate(list_probas)

    test = emotions_encoded['test']['label']
    plt.rcParams["figure.figsize"] = (10,10)
    confusion_mat = confusion_matrix(test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    disp.plot()
    acc_score = accuracy_score(preds, test)
    plt.title('accuracy = '+ str(acc_score))
    plt.savefig(cwd+"/reports/figures/confusion_matrix.png")

    RocCurveDisplay.from_predictions(
        test,
        probas, 
        name="positive vs negative",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\nPositive vs Negative Reviews")
    plt.legend()
    plt.savefig(cwd+"/reports/figures/RocCurve.png")
    plt.show()

if __name__ == "__main__":
    visualize()