import streamlit as st
import pandas as pd
import json
from io import StringIO
from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
import autoop.core.ml.metric as metrics
import autoop.core.ml.model as ml_model


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

if len(datasets) == 0:
    st.warning("No datasets found. Please create a dataset first.")
else:
    # Display available datasets in a selectbox
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

    # Retrieve the selected dataset based on its name
    selected_dataset = next((dataset for dataset in datasets
                             if dataset.name == selected_dataset_name), None)

    if selected_dataset:
        st.write(f"**Dataset:** {selected_dataset.name}",
                 f"**Version:** {selected_dataset.version}",
                 f"**path:** {selected_dataset.asset_path}")
        st.write(f"**Version:** {selected_dataset.version}")
        st.write(f"**path:** {selected_dataset.asset_path}")

        # Load the dataset into a DataFrame
        data_df = selected_dataset.read()
        if isinstance(data_df, bytes):
            decoded_data = data_df.decode('utf-8')
            try:
                # Try loading as JSON
                data_df = pd.DataFrame(json.loads(decoded_data))
            except json.JSONDecodeError:
                # If JSON loading fails, try CSV
                data_df = pd.read_csv(StringIO(decoded_data))

        # Detect the features in the dataset
        features = detect_feature_types(selected_dataset)

        # Display and Feature selection
        st.write("### Features")
        st.write("The following features have been detected in the dataset:")
        for feature in features:
            st.write(f"- **{feature.name}**: {feature.type}")

        target_feature = st.selectbox("Select the target feature",
                                      [feature.name for feature in features])

        input_features = [feature.name for feature in features
                          if feature.name != target_feature]
        selected_input_features = st.multiselect("Select the input features",
                                                 input_features)

        # Prompt the user to select a model based on the task type.
        st.write("### Model")
        model_types = ["Regression", "Classification"]
        if target_feature in [feature.name for feature in features
                              if feature.type == "categorical"]:
            model_types = ["Classification"]
        model_type = st.selectbox("Select a model type", model_types)

        if model_type == "Regression":
            model_names = ml_model.REGRESSION_MODELS
        else:
            model_names = ml_model.CLASSIFICATION_MODELS

        model_name = st.selectbox("Select a model", model_names)

        # Prompt the user to select a dataset split.
        st.write("### Dataset Split")
        split_types = ["Train-Test Split", "Cross-Validation"]
        split_type = st.selectbox("Select a split type", split_types)

        # Prompt the user to select a set of compatible metrics.
        st.write("### Metrics")
        if model_type == "Regression":
            metric_names = metrics.REGRESSION_METRICS
        else:
            metric_names = metrics.CLASSIFICATION_METRICS

        selected_metrics = st.multiselect("Select metrics", metric_names)
        model = ml_model.get_model(model_name)
        # Prompt the user with a beautifuly formatted pipeline.
        st.write("### Pipeline Summary")
        st.write("#### Configuration")
        st.write(f"- **Dataset:** {selected_dataset.name}")
        st.write(f"- **Target Feature:** {target_feature}")
        st.write(f"- **Input Features:** {', '.join(selected_input_features)}")
        st.write(f"- **Model:** {model_name}")
        st.write(f"- **Split Type:** {split_type}")
        st.write(f"- **Metrics:** {', '.join(selected_metrics)}")
        model.fit(data_df[selected_input_features],
                  data_df[target_feature])
        # Train the class and report the results of the pipeline.
        if st.button("Train"):
            st.write("Training the model...")
            for metric_name in selected_metrics:
                metric = metrics.get_metric(metric_name)
                metric_value = metric(
                    data_df[target_feature],
                    model.predict(data_df[selected_input_features]))
                st.write(f"{metric_name}: {metric_value}")

            st.write("Training complete.")

        model_name = st.text_input("Model Name", value=model_name)
        model_version = st.text_input("Model Version", value="1.0.0")

        # save the trained model
        if st.button("Save Model"):
            automl.registry.register(model.save(model_name, model_version))
            st.write("Model saved successfully.")
        # display saved models and their weights
        st.write("### Saved Models")
        saved_models = automl.registry.list(type="model")

        if len(saved_models) == 0:
            st.write("No models saved yet.")
        else:
            for saved_model in saved_models:
                st.write(f"**{saved_model.name}**",
                         f"**Version:** {saved_model.version}",
                         f"**path:** {saved_model.asset_path}")
                st.write(f"**Version:** {saved_model.version}")
                st.write(f"**path:** {saved_model.asset_path}")
                model.load(saved_model)
                st.write(f"**Model:** {model}")
                st.write(f"**Parameters:** {model._parameters}")
                st.write(f"**Is Fitted:** {model._is_fitted}")
                st.write(f"**Artifact ID:** {saved_model.id}")
                st.write(f"**Metadata:** {saved_model.data}")
