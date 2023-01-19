import logging
logging.basicConfig(filename='app.log', level=logging.INFO)

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

import streamlit as st


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")       
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

@st.cache
def predict_step(image_path):
   print('here')
   images=[]
   i_image = Image.open(image_path)
   if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

   images.append(i_image)
   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   logging.info(preds)
   print(preds)
   return preds

def frontend():
    st.set_page_config(page_title="Image Captioning Model", page_icon=":guardsman:", layout="wide")
    st.title("Image Captioning Model")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width='auto')
        if st.button("Predict"):
            with st.spinner("Making prediction..."):
                preds = predict_step(uploaded_file)
                st.success("Prediction complete.")
                st.write("Caption: " + preds[0])

if __name__ == "__main__":
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
