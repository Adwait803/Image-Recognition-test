import time

import requests
import streamlit as st
from PIL import Image



st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Async partial__ CIFAR")

image = st.file_uploader("Choose an image")

if image is not None:
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    files = {"file": image}
    res = requests.post(f"http://localhost:8080/a", files=files)
    result=res.json()
    print(result[0])
    st.title("Object: "+result[0])
    res1=result[1]
    st.title("Probability: "+str(res1[0]))
    


         