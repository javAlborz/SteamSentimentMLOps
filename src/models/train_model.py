import torch
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Value, load_from_disk


from src.data.make_dataset import ReviewDataset
from src.models.model import SteamModel


from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


MODEL_CKPT =  "distilbert-base-uncased"


dataset = ReviewDataset('data/raw','data/processed')

emotions_encoded = load_from_disk('data/processed')

print(emotions_encoded['train'][:1])

num_labels = 2

model = SteamModel()
#model = AutoModelForTokenClassification(MODEL_CKPT, num_labels=2)

tokenizer = ReviewDataset.tokenizer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{MODEL_CKPT}-finetuned-Steam"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False, 
                                  log_level="error")
                                  
trainer = Trainer(model=model, args=training_args, 
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["valid"],
                  tokenizer=tokenizer)
trainer.train()




