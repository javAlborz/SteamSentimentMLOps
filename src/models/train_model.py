import torch
import evaluate
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import wandb
import os

# from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Value, load_from_disk
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score


from src.data.make_dataset import ReviewDataset
from src.models.model import SteamModel, SteamConfig
from src.models.config import SteamConfigClass

os.environ["WANDB_PROJECT"] = 'steam_sentiment_analysis'
os.environ["WANDB_LOG_MODEL"] = 'true'
wandb.login()


def compute_metrics(eval_preds) -> (dict | None):
    """
    Function utilized by transformers.Trainer

    Args:
        eval_preds (_type_): _description_

    Returns:
        (dict | None): _description_
    """
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


cs = ConfigStore.instance()
cs.store(name='steam_config', node=SteamConfigClass)


@hydra.main(config_path='conf', config_name='config.yaml')
def main(cfg: SteamConfigClass) -> None:
    """
    Train model based on config

    Args:
        cfg (SteamConfigClass): configuration file
    """

    processed_data = ReviewDataset(cfg.paths.in_folder, cfg.paths.out_folder,
                                   model_ckpt=cfg.params.model_ckpt, sample_size=cfg.params.sample_size, force=True)
    emotions_encoded = processed_data.processed
    tokenizer = processed_data.tokenizer

    config = SteamConfig()
    model = SteamModel(config, cfg.params.model_ckpt, cfg.params.num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #config = AutoConfig.from_pretrained(cfg.params.model_ckpt)
    #model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)

    #logging_steps = len(emotions_encoded["train"]) // cfg.params.batch_size
    model_name = f"{cfg.params.model_ckpt}-finetuned-Steam"

    training_args = TrainingArguments(output_dir=model_name,
                                      num_train_epochs=cfg.params.epochs,
                                      learning_rate=cfg.params.lr,
                                      per_device_train_batch_size=cfg.params.batch_size,
                                      per_device_eval_batch_size=cfg.params.batch_size,
                                      weight_decay=cfg.params.weight_decay,
                                      evaluation_strategy="steps",
                                      save_strategy = "steps", #"epoch" "steps" "no"
                                      disable_tqdm=False,
                                      logging_steps=1,
                                      push_to_hub=False,
                                      log_level="error",
                                      report_to='wandb',
                                      run_name=cfg.params.run_name)

    trainer = Trainer(model=model, args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=emotions_encoded["train"],
                      eval_dataset=emotions_encoded["valid"],
                      tokenizer=tokenizer)

    # model takes a hella lot of memory. To run locally try decreasing batch_size by a lot :/
    trainer.train()
    print("gonna save model")
    trainer.save_model('models2/')
    print("saved model")
    #model.save_pretrained('models2/')


if __name__ == "__main__":
    print("Testing trigger")
    main()
