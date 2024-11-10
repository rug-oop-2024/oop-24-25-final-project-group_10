import streamlit as st
import pandas as pd
import json
from io import StringIO
from app.core.system import AutoMLSystem
import autoop.core.ml.model as ml_model

st.set_page_config(page_title="Deployment", page_icon="📈")

automl = AutoMLSystem.get_instance()


def write_helper_text(text: str):
    """Display helper text with a custom style."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def display_model_selection(models):
    """Display a selection box for available models."""
    model_names = [model.name for model in models]
    model_artifact_name = st.selectbox("Select a model", model_names)
    return next((model for model in models
                 if model.name == model_artifact_name), None)


def load_and_display_model_summary(model_artifact, model):
    """Load the model and display its summary."""

    st.write("### Model Summary")
    st.write(f"**Name:** {model_artifact.name}")
    st.write(f"**Version:** {model_artifact.version}")
    st.write(f"**Type:** {model_artifact.type}")
    st.write(f"**Parameters:** {model.parameters}")
    st.write("**Trained on dataset:** "
             f"{model_artifact.metadata['dataset']}.csv")
    st.write("**Metrics:**")
    for metric, score in model_artifact.metadata["metrics"].items():
        st.write(f"- **{metric}**: {score}")


def select_dataset(datasets: list) -> object:
    """Deploy the selected model."""
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    selected_dataset = next(
        (dataset for dataset in datasets
         if dataset.name == selected_dataset_name),
        None
    )
    if selected_dataset:
        st.write(f"**Dataset:** {selected_dataset.name}",
                 f"**Version:** {selected_dataset.version}",
                 f"**Path:** {selected_dataset.asset_path}")
    return selected_dataset


def deploy_model(model, selected_dataset):
    """Deploy the selected model."""
    data = selected_dataset.read()
    predictions = model.predict(data)
    st.write("### Predictions")
    st.write(predictions)


def main():
    st.write("# 🚀 Deployment")
    automl = AutoMLSystem.get_instance()
    models = automl.registry.list(type="model")

    if not models:
        st.warning("No models found. Please train a model first.")
        return

    model_artifact = display_model_selection(models)
    model = ml_model.get_model(model_artifact.metadata["type_of"])
    model.load(model_artifact)

    if model_artifact:
        load_and_display_model_summary(model_artifact, model)

    datasets = automl.registry.list(type="dataset")
    selected_dataset = select_dataset(datasets)

    if selected_dataset:
        deploy_model(model, selected_dataset)


if __name__ == "__main__":
    main()
