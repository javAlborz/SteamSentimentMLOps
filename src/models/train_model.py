import torch
import evaluate
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel, Value, load_from_disk
from transformers import AutoModelForSequenceClassification,Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score

from src.data.make_dataset import ReviewDataset
from src.models.model import SteamModel




MODEL_CKPT =  "bert-base-uncased"


dataset = ReviewDataset('data/raw','data/processed', name=MODEL_CKPT, force=True)

tokenized_dataset = dataset.processed

tokenizer = dataset.tokenizer

tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_dataset['train'][:1])

NUM_LABELS = 2

model = SteamModel(MODEL_CKPT, NUM_LABELS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#config = AutoConfig.from_pretrained(MODEL_CKPT)
#model = AutoModelForSequenceClassification.from_config(config)
model.to(device)

tokenizer = dataset.tokenizer

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions[0].argmax(-1)
  f1 = f1_score(labels, preds, average="weighted")
  acc = accuracy_score(labels, preds)
  return {"accuracy": acc, "f1": f1}


batch_size = 32
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{MODEL_CKPT}-finetuned-Steam"

#model(input, MODEL_CKPT)

training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=3,
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
                  #compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

# model is stuck on training, still gotta find out why
#trainer.train()