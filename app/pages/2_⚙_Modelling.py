import streamlit as st
import pandas as pd
import json
from io import StringIO
from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
import autoop.core.ml.metric as metrics
import autoop.core.ml.model as ml_model

st.set_page_config(page_title="Modelling", page_icon="📈")

automl = AutoMLSystem.get_instance()


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def display_datasets(datasets: object) -> object:
    """Display dataset selection and information."""
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", [""]
                                         + dataset_names)
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


def load_dataset(selected_dataset: object) -> pd.DataFrame:
    """
    Load the selected dataset into a DataFrame.

    parameters:
        selected_dataset: object
            The selected dataset artifact.

    returns:
        pd.DataFrame
            The dataset as a DataFrame.
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
        st.write(f"- **{feature.name}**: {feature.type}")
    return features


def select_model(features: dict, target_feature: str) -> tuple:
    """
    Allow user to select model type and specific model.

    parameters:
        features: dict
            The features in the dataset.
        target_feature: str
            The target feature for the model.
    """
    model_type = (
        "Classification"
        if any(
            feature.name == target_feature and feature.type == "categorical"
            for feature in features
        )
        else "Regression"
    )
    st.write("### Model Selection")
    st.write(f"**Model Type:** {model_type}")

    model_names = (
        ml_model.CLASSIFICATION_MODELS
        if model_type == "Classification"
        else ml_model.REGRESSION_MODELS
    )
    model_name = st.selectbox("Select a model", [""] + model_names)
    return model_type, model_name


def display_pipeline_summary(
    selected_dataset: object,
    target_feature: str,
    selected_input_features: list,
    model_name: str,
        selected_metrics: list | None) -> tuple:

    """
    Display a summary of the pipeline configuration.

    parameters:
        selected_dataset: object
            The selected dataset artifact.
        target_feature: str
            The target feature for the model.
        selected_input_features: list
            The input features for the model.
        model_name: str
            The name of the model.
        selected_metrics: list
            The metrics to evaluate the model.
    """
    st.write("### Pipeline Summary")
    model_name = st.text_input("Model Name", value=model_name)
    model_version = st.text_input("Model Version", value="1.0.0")
    st.write("#### Configuration")
    st.write(f"- **Dataset:** {selected_dataset.name}")
    st.write(f"- **Target Feature:** {target_feature}")
    st.write(f"- **Input Features:** {', '.join(selected_input_features)}")
    st.write(f"- **Model:** {model_name}")
    if selected_metrics:
        st.write(f"- **Metrics:** {', '.join(selected_metrics)}")
    return model_name, model_version


def train_and_save_model(
    model: object,
    data_df: pd.DataFrame,
    selected_input_features: list,
    target_feature: str,
    selected_metrics: list,
    selected_dataset: object,
    custom_name: str,
        model_version: str) -> None:

    """
    Train and save the model, displaying metric results.

    parameters:
        model: object
            The model to train.
        data_df: pd.DataFrame
            The dataset to train the model on.
        selected_input_features: list
            The input features for the model.
        target_feature: str
            The target feature for the model.
        selected_metrics: list
            The metrics to evaluate the model.
        selected_dataset: object
            The dataset the model is trained on.
        custom_name: str
            The custom name for the model.
        model_version: str
            The version of the model.
    """
    st.write("Training the model...")
    model.fit(data_df[selected_input_features], data_df[target_feature])

    model.set_trained_dataset(selected_dataset.name)
    model.set_target_feature(target_feature)
    model.set_features(selected_input_features)

    for metric_name in selected_metrics:
        metric = metrics.get_metric(metric_name)
        metric_value = metric(data_df[target_feature],
                              model.predict(data_df[selected_input_features]))
        model.set_metric_score(metric_name, metric_value)
        st.write(f"{metric_name}: {metric_value}")

    st.write("Training complete.")
    automl.registry.register(model.save(custom_name, model_version))
    st.write("Model saved successfully.")


def main() -> None:
    st.write("# ⚙ Modelling")
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")

    if not datasets:
        st.warning("No datasets found. Please create a dataset first.")
        return

    selected_dataset = display_datasets(datasets)

    if selected_dataset:
        data_df = load_dataset(selected_dataset)
        features = display_features(selected_dataset)

        # Feature selection
        target_feature = st.selectbox("Select the target feature",
                                      [feature.name for feature in features])

        input_features = [feature.name for feature
                          in features if feature.name != target_feature]

        selected_input_features = st.multiselect("Select the input features",
                                                 input_features)
        if selected_input_features:
            # Model selection
            model_type, model_name = select_model(features, target_feature)

            if model_name:
                # Metric selection
                st.write("### Metrics")
                metric_names = (
                    metrics.CLASSIFICATION_METRICS
                    if model_type == "Classification"
                    else metrics.REGRESSION_METRICS
                )
                selected_metrics = st.multiselect("Select metrics",
                                                  metric_names)
                # Pipeline summary
                custom_name, model_version = display_pipeline_summary(
                    selected_dataset,
                    target_feature,
                    selected_input_features,
                    model_name,
                    selected_metrics
                )

                # Train and save model
                model = ml_model.get_model(model_name)
                if st.button("Train And Save Model"):
                    train_and_save_model(model,
                                         data_df,
                                         selected_input_features,
                                         target_feature,
                                         selected_metrics,
                                         selected_dataset,
                                         custom_name,
                                         model_version)


if __name__ == "__main__":
    main()
