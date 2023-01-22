"""
Streamlit web application
"""

import json
import numpy as np
import requests
import streamlit as st

st.set_page_config(
    page_title="Image Segmentation",
    page_icon="✌️",
    initial_sidebar_state="expanded"
)


def main():
    """
    Main website function.
    """
    st.write("# Image Segmentation")
    st.write("This website exposes an api for Image segmentation.")
    st.write("Upload an Image, and choose the type of objects to mask.")

    st.write("### Category")
    radio = st.radio("Pick Ctegory 👇", ["Car", "Person"], horizontal=True)

    st.write("### Input Image")
    uploaded_file = st.file_uploader("Choose an image", type="jpg")

    if uploaded_file is not None:

        st.write("### Output")
        col1, col2 = st.columns([1, 1])

        col1.subheader("Original Image")
        col1.image(uploaded_file, use_column_width="auto")
        endpoint = "car-mask" if radio == "Car" else "person-mask"
        files = {
            "file": (uploaded_file.name, uploaded_file, "multipart/form-data")
            }
        response = requests.post(f"http://localhost:8000/{endpoint}",
                                 files=files)
        arr = np.asarray(json.loads(response.json()))

        col2.subheader("Masked Image")
        col2.image(arr, use_column_width="auto")


if __name__ == "__main__":
    main()
