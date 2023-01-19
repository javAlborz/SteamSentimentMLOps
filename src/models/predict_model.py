import argparse

# import numpy as np
import torch
# import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModel, AutoConfig
from src.models.model import SteamModel, SteamConfig


def predict(text: str) -> TextClassificationPipeline:
    """
    Predict whether review is positive or negative based on text.

    Parameters
    ----------
    text : str
        Review text.

    Returns
    -------
    TextClassificationPipeline
        Result of prediction
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AutoConfig.register("SteamModel", SteamConfig)
    AutoModel.register(SteamConfig, SteamModel)
    new_model = AutoModel.from_pretrained(
        'models/', "distilbert-base-uncased", 2)
    tokenizer = AutoTokenizer.from_pretrained('models/')

    #new_model = SteamModel("distilbert-base-uncased", 2).from_pretrained('models/model_huggingface/')

    #tokenizer = AutoTokenizer.from_pretrained('models/model_huggingface/')
    #model = AutoModel.from_pretrained('models/model_huggingface/')

    new_model.to(device)

    pipe = TextClassificationPipeline(
        model=new_model, tokenizer=tokenizer, return_all_scores=True)

    return pipe(text)


if __name__ == "__main__":
    # when predict_model.py is being run from command line it takes in review text to predict with

    parser = argparse.ArgumentParser(description="Training arguments")
    #parser.add_argument("model_checkpoint", type=str)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    result = predict(args.text)

    print(result)
