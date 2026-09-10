
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Final_model (1).pkl"
PREPROCESS_PATH = BASE_DIR / "model_scaler (1).pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESS_PATH, "rb") as f:
    preprocess = pickle.load(f)

scaler = preprocess["scaler"]
feature_names = preprocess["feature_names"]
numeric_cols = preprocess["numeric_cols"]
categorical_cols = preprocess["categorical_cols"]

st.set_page_config(
    page_title="Bank Deposit Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("Bank Deposit Predictor")

st.write(
    "Enter the customer details to predict whether "
    "the customer will subscribe to a deposit."
)

user_inputs = {}

st.subheader("Customer Information")

user_inputs["job"] = st.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

user_inputs["education"] = st.selectbox(
    "Education",
    [
        "primary",
        "secondary",
        "tertiary",
        "unknown"
    ]
)

user_inputs["marital"] = st.selectbox(
    "Marital",
    [
        "married",
        "single",
        "divorced"
    ]
)

user_inputs["default"] = st.selectbox(
    "Default",
    ["yes", "no"]
)

user_inputs["housing"] = st.selectbox(
    "Housing",
    ["yes", "no"]
)

user_inputs["loan"] = st.selectbox(
    "Loan",
    ["yes", "no"]
)

user_inputs["month"] = st.selectbox(
    "Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ]
)

user_inputs["loan_housing"] = st.selectbox(
    "Loan Housing",
    ["yes", "no"]
)

user_inputs["age_group"] = st.selectbox(
    "Age Group",
    [
        "young",
        "adult",
        "middle",
        "old"
    ]
)

st.subheader("Numerical Information")

for col in numeric_cols:
    user_inputs[col] = st.number_input(
        col.replace("_", " ").title(),
        min_value=0.0,
        value=0.0,
        step=1.0
    )

if st.button("🔮 Predict", use_container_width=True):

    try:
        df = pd.DataFrame([user_inputs])

        df_encoded = pd.get_dummies(
            df,
            columns=categorical_cols,
            dtype=int
        )

        df_encoded = df_encoded.loc[
            :,
            ~df_encoded.columns.duplicated()
        ]

        existing_numeric_cols = [
            col for col in numeric_cols
            if col in df_encoded.columns
        ]

        if existing_numeric_cols:
            df_encoded[existing_numeric_cols] = scaler.transform(
                df_encoded[existing_numeric_cols]
            )

        expected_features = list(feature_names)

        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded.reindex(
            columns=expected_features,
            fill_value=0
        )

        if hasattr(model, "n_features_in_"):
            if df_encoded.shape[1] != model.n_features_in_:
                st.error(
                    f"Feature mismatch: model expects "
                    f"{model.n_features_in_} features, but received "
                    f"{df_encoded.shape[1]}."
                )
                st.stop()

        prediction = model.predict(df_encoded)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                " The customer will subscribe to a deposit."
            )
        else:
            st.info(
                "The customer will not subscribe to a deposit."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_encoded)[0]
            probability = probabilities[1] * 100

            st.write(
                f"Probability of subscription: "
                f"**{probability:.2f}%**"
            )

    except Exception as e:
        st.error(
            "An error occurred while making the prediction."
        )
        st.exception(e)



