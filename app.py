import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        st.subheader(f"Results for {metrics.get('Model', model_choice)}")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("AUC", f"{metrics['AUC']:.4f}")
        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        col4.metric("Recall", f"{metrics['Recall']:.4f}")
        col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        col6.metric("MCC", f"{metrics['MCC']:.4f}")
        
        st.divider()

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        with col_right:
            st.subheader("Metrics Summary Table")
            metrics_df = pd.DataFrame([metrics])
            st.dataframe(metrics_df, use_container_width=True)

    else:
        st.warning("Please upload a CSV file before evaluating.")
