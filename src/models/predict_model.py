import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, TextClassificationPipeline, AutoModel, AutoTokenizer, AutoModel, AutoConfig
from src.models.model import SteamModel, SteamConfig


def predict() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    #parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    AutoConfig.register("SteamModel", SteamConfig)
    AutoModel.register(SteamConfig, SteamModel)
    new_model = AutoModel.from_pretrained('models/', "distilbert-base-uncased", 2)
    tokenizer =  AutoTokenizer.from_pretrained('models/')

    #new_model = SteamModel("distilbert-base-uncased", 2).from_pretrained('models/model_huggingface/')
      
    #tokenizer = AutoTokenizer.from_pretrained('models/model_huggingface/')
    #model = AutoModel.from_pretrained('models/model_huggingface/')

    new_model.to(device)


    text = args.text
    pipe = TextClassificationPipeline(model=new_model, tokenizer=tokenizer, return_all_scores=True)
    print(pipe(text))



if __name__ == "__main__":
    predict()