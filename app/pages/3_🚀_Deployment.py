import streamlit as st
import pandas as pd
import json
from io import StringIO
from app.core.system import AutoMLSystem
from autoop.functional.feature import detect_feature_types
import autoop.core.ml.metric as metrics
import autoop.core.ml.model as ml_model


st.set_page_config(page_title="Deployment", page_icon="📈")

# A page where the user can deploy a trained model.


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# 🚀 Deployment")

automl = AutoMLSystem.get_instance()

models = automl.registry.list(type="model")

if len(models) == 0:
    st.warning("No models found. Please train a model first.")
else:
    # Display available models in a selectbox
    model_names = [model.name for model in models]
    selected_model_name = st.selectbox("Select a model", model_names)

    # Retrieve the selected model based on its name
    selected_model = next((model for model in models
                           if model.name == selected_model_name), None)

    if selected_model:
        st.write(f"**Model:** {selected_model.name}",
                 f"**Version:** {selected_model.version}",
                 f"**path:** {selected_model.asset_path}")

        # Load the model
        model = ml_model.get_model(selected_model.name)
        model.load(selected_model)

        # pipeline summary
        st.write("### Model Summary")
        st.write(f"**Type:** {model.type}")
        st.write(f"**Parameters:** {model.parameters}")
        st.write(f"**Trained on dataset:** "
                 f"{selected_model.metadata['dataset']}")
        st.write(f"**Metrics:** {[x for x in selected_model.metadata['metrics']]}")
