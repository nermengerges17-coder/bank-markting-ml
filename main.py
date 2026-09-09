import streamlit as st
import pandas as pd
import pickle

with open(r"D:\New folder\Final_model (1).pkl", "rb") as f:
    model = pickle.load(f)

with open(r"D:\New folder\model_scaler (1).pkl", "rb") as f:
    preprocess = pickle.load(f)

scaler = preprocess["scaler"]
feature_names = preprocess["feature_names"]
numeric_cols = preprocess["numeric_cols"]
categorical_cols = preprocess["categorical_cols"]



categorical_cols = [
    "job",
    "education",
    "marital",
    "default",
    "housing",
    "loan",
    "month",
    "loan_housing",
    "age_group"
]

st.title("Bank Deposit Predictor")

st.write(
    "Enter the customer details to predict whether "
    "the customer will subscribe to a deposit."
)

user_inputs = {}

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

for col in numeric_cols:
    user_inputs[col] = st.number_input(
        col.replace("_", " ").title(),
        min_value=0.0,
        step=1.0
    )

if st.button("Predict"):

    df = pd.DataFrame([user_inputs])

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        dtype=int
    )

    df = df.loc[:, ~df.columns.duplicated()]

    df[numeric_cols] = scaler.transform(
        df[numeric_cols]
    )

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

        df = df.reindex(
            columns=model_features,
            fill_value=0
        )

    else:

        expected_features = model.n_features_in_

        if df.shape[1] > expected_features:
            df = df.iloc[:, :expected_features]

        elif df.shape[1] < expected_features:

            missing = expected_features - df.shape[1]

            for i in range(missing):
                df[f"missing_{i}"] = 0

    prediction = model.predict(df)[0]

    if prediction == 1:
        st.success(
            "The customer will subscribe to a deposit."
        )
    else:
        st.info(
            "The customer will not subscribe to a deposit."
        )

