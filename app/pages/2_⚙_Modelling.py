import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

if len(datasets) == 0:
    st.warning("No datasets found. Please create a dataset first.")
else:
    # Display available datasets in a selectbox
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

    # Retrieve the selected dataset based on its name
    selected_dataset = next((dataset for dataset in datasets if dataset.name == selected_dataset_name), None)

    if selected_dataset:
        st.write(f"**Dataset:** {selected_dataset.name}")
        st.write(f"**Version:** {selected_dataset.version}")
        st.write(f"**path:** {selected_dataset.asset_path}")

        # Load the dataset into a DataFrame
        data_df = automl.registry.get(selected_dataset.id).read()

        # Detect the features in the dataset
        features = detect_feature_types(selected_dataset)
        
        st.write("### Features")
        st.write("The following features have been detected in the dataset:")
        for feature in features:
            st.write(f"- **{feature.name}**: {feature.type}")
        