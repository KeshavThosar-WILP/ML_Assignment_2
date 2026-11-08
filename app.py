import streamlit as st

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("ML Assignment 2 - Model Evaluation")
st.write("Upload test dataset, select a model, and evaluate its performance.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

submit_button = st.button("Evaluate")
