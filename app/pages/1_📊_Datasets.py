import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize the AutoML system singleton instance
automl = AutoMLSystem.get_instance()

# Page title and description
st.title("Datasets")
st.write(
    "Datasets are used to train and evaluate machine learning models. "
    "You can view, upload, and delete datasets here."
)

# Section for creating a new dataset
st.header("Create a new dataset")
dataset_name = st.text_input("Name of the dataset")
dataset_file = st.file_uploader("Upload a dataset file", type=["csv"])

# Create and save dataset if a file is uploaded
if dataset_file is not None:
    # Read the CSV file into a DataFrame
    dataframe = pd.read_csv(dataset_file)
    
    # Define a path for the dataset's asset (storage location)
    asset_path = f"{dataset_name}.csv"
    
    # Create the Dataset artifact using from_dataframe
    dataset = Dataset.from_dataframe(
        data=dataframe,
        asset_path=asset_path,
        name=dataset_name,  # Provide the name as required
        version="1.0.0",  # Specify version if required by Dataset
    )
    
    # Save the dataset artifact using the AutoML system's artifact registry
    save_button = st.button("Save Dataset")
    if save_button:
        # Register the dataset with AutoML system
        automl.registry.register(dataset)
        st.success(f"Dataset '{dataset_name}' has been successfully saved.")
else:
    st.info("Please upload a CSV file to create a new dataset.")

# Section for listing existing datasets
st.header("Existing datasets")
datasets = automl.registry.list(type="dataset")
if len(datasets) == 0:
    st.info("No datasets found.")
else:
    for dataset in datasets:
        st.write(f"**{dataset.name}** (Version: {dataset.version})")
