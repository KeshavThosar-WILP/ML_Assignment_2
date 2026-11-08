import streamlit as st
import pandas as pd
from utils import AVAILABLE_MODELS, evaluate_dataset

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("ML Assignment 2 - Model Evaluation")
st.write("Upload test dataset, select a model, and evaluate its performance.")

with st.form(key="evaluation_form"):
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    model_choice = st.selectbox("Select Model", AVAILABLE_MODELS)
    submit_button = st.form_submit_button("Evaluate")

if submit_button:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        metrics, confusion_matrix = evaluate_dataset(df, model_choice)
        print(metrics)
    else:
        st.warning("Please upload a CSV file before evaluating.")
