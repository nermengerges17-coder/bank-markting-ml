

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(**file**).resolve().parent

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

st.title("🏦 Bank Deposit Predictor")

st.write(
"Enter the customer's basic information "
"to predict whether they will subscribe to a deposit."
)

user_inputs = {}

st.subheader("Customer Information")

user_inputs["age"] = st.number_input(
"Age",
min_value=18,
max_value=100,
value=30,
step=1
)

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
"Marital Status",
[
"married",
"single",
"divorced"
]
)

user_inputs["default"] = st.selectbox(
"Has Credit Default?",
["no", "yes"]
)

user_inputs["housing"] = st.selectbox(
"Has Housing Loan?",
["no", "yes"]
)

user_inputs["loan"] = st.selectbox(
"Has Personal Loan?",
["no", "yes"]
)

user_inputs["month"] = st.selectbox(
"Contact Month",
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

for col in numeric_cols:
  if col not in user_inputs:
   if col == "age":
    user_inputs[col] = user_inputs["age"]
   else:
    user_inputs[col] = 0.0

if st.button("🔮 Predict", use_container_width=True):


try:
    df = pd.DataFrame([user_inputs])

    df_encoded = pd.get_dummies(
        df,
        columns=[
            col for col in categorical_cols
            if col in df.columns
        ],
        dtype=int
    )

    df_encoded = df_encoded.loc[
        :,
        ~df_encoded.columns.duplicated()
    ]

    existing_numeric_cols = [
        col
        for col in numeric_cols
        if col in df_encoded.columns
    ]

    if existing_numeric_cols:
        try:
            df_encoded[existing_numeric_cols] = scaler.transform(
                df_encoded[existing_numeric_cols]
            )
        except Exception:
            pass

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
            "✅ The customer will subscribe to a deposit."
        )
    else:
        st.info(
            "❌ The customer will not subscribe to a deposit."
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

