import streamlit as st
import pandas as pd
import json
from io import StringIO
from app.core.system import AutoMLSystem
import autoop.core.ml.model as ml_model
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Deployment", page_icon="📈")

automl = AutoMLSystem.get_instance()


def write_helper_text(text: str):
    """Display helper text with a custom style."""
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def display_model_selection(models):
    """
    Display a selection box for available models.

    parameters:
        models: list
            A list of model artifacts.
    """
    model_names = [model.name for model in models]
    model_artifact_name = st.selectbox("Select a model", [""] + model_names)
    return next((model for model in models
                 if model.name == model_artifact_name), None)


def load_and_display_model_summary(model_artifact, model):
    """
    Load the model and display its summary.

    parameters:
        model_artifact: object
            The model artifact to load.
        model: object
            The model to load.
    """

    st.write("### Model Summary")
    st.write(f"**Name:** {model_artifact.name}")
    st.write(f"**Version:** {model_artifact.version}")
    st.write(f"**Type:** {model_artifact.type}")
    st.write("**Trained on dataset:** "
             f"{model.metadata['dataset']}.csv")
    st.write("**target feature of the model:**")
    st.write(f"- **{model.metadata['target_feature']}**")
    st.write("**features trained on:**")
    for feature in model.metadata["features"]:
        st.write(f"- **{feature}**")
    st.write("**Metrics:**")
    for metric, score in model.metadata["metrics"].items():
        st.write(f"- **{metric}**: {score}")


def select_dataset(datasets: list) -> object:
    """
    Deploy the selected model.

    parameters:
        datasets: list
            A list of dataset artifacts.
    """
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", [""]
                                         + dataset_names)
    selected_dataset = next(
        (dataset for dataset in datasets
         if dataset.name == selected_dataset_name),
        None
    )
    return selected_dataset


def load_dataset(selected_dataset: object) -> pd.DataFrame:
    """
    Load the selected dataset into a DataFrame.

    parameters:
        selected_dataset: object
            The selected dataset artifact.
    """
    data_df = selected_dataset.read()
    if isinstance(data_df, bytes):
        decoded_data = data_df.decode('utf-8')
        try:
            data_df = pd.DataFrame(json.loads(decoded_data))
        except json.JSONDecodeError:
            data_df = pd.read_csv(StringIO(decoded_data))
    return data_df


def display_features(selected_dataset):
    """
    Detect and display features in the dataset.

    parameters:
        selected_dataset: object
            The selected dataset artifact.
    """
    features = detect_feature_types(selected_dataset)
    st.write("### Features")
    st.write("The following features have been detected in the dataset:")
    for feature in features:
        # Accessing attributes of each feature object directly
        st.write(f"- **{feature.name}**: {feature.type}")
    return features


def select_features(model_features: list, data_features: list) -> list:
    """
    Select and verify features to deploy the model.

    parameters:
        model_features: list
            The features used during model training.
        data_features: list
            The features detected in the dataset.
    """
    model_features_set = set(model_features)
    data_features_names = [feature.name for feature in data_features]

    selected_features = st.multiselect("Select features to deploy the model:",
                                       data_features_names)

    selected_features_set = set(selected_features)
    missing_in_data = model_features_set - selected_features_set
    extra_in_data = selected_features_set - model_features_set

    if missing_in_data:
        st.error(
            "The following features are required but are not selected: "
            f"{missing_in_data}"
        )
    if extra_in_data:
        st.warning(
            f"The following features were selected but were not used during "
            f"model training: {extra_in_data}"
        )

    valid_selected_features = list(
        model_features_set.intersection(selected_features_set)
    )

    return valid_selected_features


def deploy_model(model: object, dataset: pd.DataFrame, features: list):
    """
    Deploy the selected model with the matched features.

    parameters:
        model: object
            The model to deploy.
        dataset: pd.DataFrame
            The dataset to make predictions on.
        features: list
            The selected features to use for deployment.

    """
    aligned_dataset = dataset[features]
    predictions = model.predict(aligned_dataset)
    st.write("### Predictions")
    table = pd.concat([aligned_dataset,
                       pd.DataFrame(
                           predictions,
                           columns=[f"predictions of "
                                    f"{model.metadata['target_feature']}"])],
                      axis=1)
    st.write(table)


def main():
    st.write("# 🚀 Deployment")
    automl = AutoMLSystem.get_instance()
    models = automl.registry.list(type="model")

    selected_model = display_model_selection(models)
    if selected_model is not None:

        model_artifact = automl.registry.get(selected_model.id)
        model = ml_model.get_model(selected_model.metadata["type_of"])
        model.load(model_artifact)

        load_and_display_model_summary(model_artifact, model)

        datasets = automl.registry.list(type="dataset")
        selected_dataset = select_dataset(datasets)

        if selected_dataset is not None:
            data_df = load_dataset(selected_dataset)
            features = display_features(selected_dataset)
            model_features = model.metadata["features"]
            selected_features = select_features(model_features, features)

            if selected_features:
                deploy_model(model, data_df, selected_features)
            else:
                st.write("No valid features selected for deployment.")
    else:
        st.write("No models found.")


if __name__ == "__main__":
    main()
