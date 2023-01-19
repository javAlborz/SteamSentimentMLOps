import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
import torch
from PIL import Image
import streamlit as st

import argparse
# import numpy as np
import torch
# import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModel, AutoConfig
from src.models.model import SteamModel, SteamConfig



# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")       
# feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

@st.cache
def predict_step(text: str) -> TextClassificationPipeline:
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

    
def frontend():
    st.set_page_config(page_title="Image Captioning Model", page_icon=":guardsman:", layout="wide")
    st.title("Image Captioning Model")
    text = st.text_input("Enter your text")

    if text is not None:
        st.write("Input text: " + text)
        if st.button("Predict"):
            with st.spinner("Making prediction..."):
                preds = predict_step(text)
                print(type(preds))
                print(preds)
                st.success("Prediction complete.")

                for item in preds:
                    label_0_score = 0
                    label_1_score = 0
                    for label_data in item:
                        if label_data['label'] == 'LABEL_0':
                            label_0_score = label_data['score']
                        elif label_data['label'] == 'LABEL_1':
                            label_1_score = label_data['score']
                    if label_0_score > label_1_score:
                        st.write("BAD review, score:" + str(label_0_score))
                    else:
                        st.write("GOOD review, score:" + str(label_1_score))


if __name__ == "__main__":
    # when predict_model.py is being run from command line it takes in review text to predict with

    # parser = argparse.ArgumentParser(description="Training arguments")
    # #parser.add_argument("model_checkpoint", type=str)
    # parser.add_argument("--text", type=str, required=True)
    # args = parser.parse_args()

    #result = predict(args.text)

    frontend()







# @app.post("/ML_model/")
# async def cv_model(data: UploadFile = File(...)):
#     image_path = 'image.jpg'
#     with open(image_path, 'wb') as f:
#         content = await data.read()
#         f.write(content)
#     preds=predict_step(image_path)

#     response = {
#         "input": data,
#         "output": preds,
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response

# async def post_image():
#     async with aiohttp.ClientSession() as session:
#         async with session.post('http://localhost:8000/ML_model/', data=data) as resp:
#             response = await resp.json()
#             logging.info(response)
#             return response

# st.set_page_config(page_title="Hello World", layout="wide")
# st.title("Last op dit billede her")

# if st.button('Predict'):
#     file_upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#     if file_upload is not None:
#         data = {'file': file_upload}
#         response = asyncio.run(post_image())
#         st.success(response['output'])
