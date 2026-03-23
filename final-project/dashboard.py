"""Streamlit HITL Dashboard for Movie Review Sentiment Analysis Pipeline."""

import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Sentiment Pipeline Dashboard", layout="wide")
st.title("Movie Review Sentiment Analysis — HITL Dashboard")

tabs = st.tabs(["EDA", "Data Quality", "Annotation", "HITL Review", "Active Learning", "Model", "Reports"])

# --- Tab 1: EDA ---
with tabs[0]:
    st.header("Exploratory Data Analysis")
    raw_path = os.path.join(BASE, "data/raw/dataset.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df_raw))
        col2.metric("Sources", df_raw["source"].nunique())
        col3.metric("Median Length", f"{df_raw['text'].str.len().median():.0f} chars")

        st.subheader("GT Label Distribution")
        st.bar_chart(df_raw["label"].value_counts())

        st.subheader("Source Distribution")
        st.bar_chart(df_raw["source"].value_counts())

        st.subheader("Text Length Distribution")
        st.bar_chart(df_raw["text"].str.len().describe().to_frame("stats"))

    eda_path = os.path.join(BASE, "plots/eda_overview.png")
    if os.path.exists(eda_path):
        st.image(eda_path, caption="EDA Overview")
    top_words_path = os.path.join(BASE, "plots/eda_top_words.png")
    if os.path.exists(top_words_path):
        st.image(top_words_path, caption="Top-20 Words")

# --- Tab 2: Data Quality ---
with tabs[1]:
    st.header("Data Quality")
    clean_path = os.path.join(BASE, "data/raw/dataset_clean.csv")
    report_path = os.path.join(BASE, "reports/quality_report.md")

    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())

    if os.path.exists(raw_path) and os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Before", len(df_raw))
        col2.metric("After", len(df_clean))
        col3.metric("Removed", len(df_raw) - len(df_clean))

# --- Tab 3: Annotation ---
with tabs[2]:
    st.header("Annotation (BART Zero-Shot Audit)")
    ann_report_path = os.path.join(BASE, "reports/annotation_report.md")
    if os.path.exists(ann_report_path):
        with open(ann_report_path) as f:
            st.markdown(f.read())

    labeled_path = os.path.join(BASE, "data/labeled/dataset_labeled.csv")
    if os.path.exists(labeled_path):
        df_labeled = pd.read_csv(labeled_path)
        st.subheader("BART Prediction Distribution (200 sample)")
        st.bar_chart(df_labeled["predicted_label"].dropna().value_counts())

        if "confidence" in df_labeled.columns:
            st.subheader("Confidence Distribution")
            st.bar_chart(df_labeled["confidence"].describe().to_frame("stats"))

    spec_path = os.path.join(BASE, "specs/annotation_spec.md")
    if os.path.exists(spec_path):
        with st.expander("Annotation Specification"):
            with open(spec_path) as f:
                st.markdown(f.read())

# --- Tab 4: HITL Review ---
with tabs[3]:
    st.header("Human-in-the-Loop Review")

    rq_path = os.path.join(BASE, "review_queue.csv")
    rq_corr_path = os.path.join(BASE, "review_queue_corrected.csv")

    if os.path.exists(rq_corr_path):
        df_corr = pd.read_csv(rq_corr_path)
        st.subheader("Reviewed Disagreements")
        st.dataframe(df_corr.head(10), use_container_width=True)

    final_path = os.path.join(BASE, "data/labeled/final_dataset.csv")
    if os.path.exists(final_path):
        df_final = pd.read_csv(final_path)
        st.subheader("Interactive Label Editor")
        st.info("Select a row to edit its label. Changes are saved to final_dataset.csv.")

        if "edit_idx" not in st.session_state:
            st.session_state.edit_idx = 0

        idx = st.number_input("Row index", 0, len(df_final) - 1, st.session_state.edit_idx)
        row = df_final.iloc[idx]

        st.text_area("Text", row["text"], height=150, disabled=True)
        col1, col2 = st.columns(2)
        col1.write(f"**Current label:** {row['label']}")
        bart_label = row.get("predicted_label", "N/A")
        col2.write(f"**BART predicted:** {bart_label}")

        new_label = st.selectbox("New label", ["positive", "negative"], index=0 if row["label"] == "positive" else 1)
        if st.button("Save correction"):
            df_final.at[idx, "label"] = new_label
            df_final.to_csv(final_path, index=False)
            st.success(f"Row {idx} label updated to '{new_label}'")

# --- Tab 5: Active Learning ---
with tabs[4]:
    st.header("Active Learning")

    al_report_path = os.path.join(BASE, "reports/al_report.md")
    if os.path.exists(al_report_path):
        with open(al_report_path) as f:
            st.markdown(f.read())

    strat_path = os.path.join(BASE, "plots/strategy_comparison.png")
    if os.path.exists(strat_path):
        st.image(strat_path, caption="Strategy Comparison")

    lc_path = os.path.join(BASE, "plots/learning_curve.png")
    if os.path.exists(lc_path):
        st.image(lc_path, caption="Learning Curve (Entropy)")

    hist_path = os.path.join(BASE, "data/results/al_histories.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            histories = json.load(f)
        st.subheader("AL Histories (raw data)")
        for strat, hist in histories.items():
            with st.expander(f"Strategy: {strat}"):
                st.dataframe(pd.DataFrame(hist))

# --- Tab 6: Model ---
with tabs[5]:
    st.header("Final Model")

    model_path = os.path.join(BASE, "models/sentiment_model.joblib")
    if os.path.exists(model_path):
        st.success("Model loaded: sentiment_model.joblib")

        # Compute metrics dynamically from model + test data
        final_path_model = os.path.join(BASE, "data/labeled/final_dataset.csv")
        if os.path.exists(final_path_model):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            df_m = pd.read_csv(final_path_model)
            _, test_m = train_test_split(df_m, test_size=0.2, stratify=df_m["label"], random_state=42)
            model = joblib.load(model_path)
            preds_m = model.predict(test_m["text"])
            acc_m = accuracy_score(test_m["label"], preds_m)
            f1_m = f1_score(test_m["label"], preds_m, average="weighted")

            st.subheader("Model Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{acc_m:.4f}")
            col2.metric("F1 (weighted)", f"{f1_m:.4f}")

            st.subheader("Per-Class Metrics")
            report_dict = classification_report(test_m["label"], preds_m, output_dict=True)
            metrics_df = pd.DataFrame({
                "Class": ["negative", "positive", "weighted avg"],
                "Precision": [report_dict["negative"]["precision"], report_dict["positive"]["precision"], report_dict["weighted avg"]["precision"]],
                "Recall": [report_dict["negative"]["recall"], report_dict["positive"]["recall"], report_dict["weighted avg"]["recall"]],
                "F1": [report_dict["negative"]["f1-score"], report_dict["positive"]["f1-score"], report_dict["weighted avg"]["f1-score"]],
                "Support": [int(report_dict["negative"]["support"]), int(report_dict["positive"]["support"]), int(report_dict["weighted avg"]["support"])],
            }).round(4)
            st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Try the Model")
        user_text = st.text_area("Enter a movie review:", height=100)
        if st.button("Predict") and user_text.strip():
            model = joblib.load(model_path)
            pred = model.predict([user_text])[0]
            proba = model.predict_proba([user_text])[0]
            st.write(f"**Prediction:** {pred}")
            st.write(f"**Confidence:** {max(proba):.4f}")

# --- Tab 7: Reports ---
with tabs[6]:
    st.header("Reports")

    report_files = {
        "Final Report": "reports/final_report.md",
        "Quality Report": "reports/quality_report.md",
        "Annotation Report": "reports/annotation_report.md",
        "AL Report": "reports/al_report.md",
        "Data Card": "data/labeled/data_card.md",
    }
    for name, path in report_files.items():
        full_path = os.path.join(BASE, path)
        if os.path.exists(full_path):
            with st.expander(name):
                with open(full_path) as f:
                    st.markdown(f.read())
