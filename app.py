from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
from PIL import Image
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

st.header('Create :green[Images] from your Imagination')

prompt = st.text_input('Enter Prompt')

if st.button("Generate", type="primary"):
    if prompt == "":
        st.warning("Please enter a prompt")
    else:
        image = pipe(prompt).images[0]
        image.save("thumbnail.png")
        image = Image.open('/content/thumbnail.png')
        st.image(image)
        with open("/content/thumbnail.png", "rb") as file:
            btn = st.download_button(
            label="Download image",
            type = "primary",
            data=file,
            file_name="/content/thumbnail.png",
            mime="image/png"
          )
