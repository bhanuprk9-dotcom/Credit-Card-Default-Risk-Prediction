
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and threshold
model = joblib.load('credit_default_model.pkl')
threshold = joblib.load('threshold.pkl')

# Page configuration
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Application heading
st.title("💳 Credit Card Default Risk Prediction System")

st.write(
    "Enter customer financial and demographic details "
    "to predict the probability of credit card default."
)

st.divider()

# Customer input form
with st.form("customer_form"):

    st.subheader("1. Customer Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
            "Credit Limit (LIMIT_BAL)",
            min_value=0,
            value=50000
        )

    with col2:
        sex = st.selectbox(
    "Sex",
    ["F", "M"] 
        )

    with col3:
        education = st.selectbox(
    "Education",
    [
        "Graduate school",
        "High School",
        "Others",
        "University",
        "Unknown"
    ]
)

    col4, col5 = st.columns(2)

    with col4:
       marriage = st.selectbox(
    "Marriage",
    ["Married", "Other", "Single"]
)

    with col5:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    st.divider()

    st.subheader("2. Repayment History")

    payment_status = {}

    cols = st.columns(3)

    pay_features = [
        'PAY_0', 'PAY_2', 'PAY_3',
        'PAY_4', 'PAY_5', 'PAY_6'
    ]

    for i, feature in enumerate(pay_features):
        with cols[i % 3]:
            payment_status[feature] = st.selectbox(
                feature,
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
            )

    st.divider()

    st.subheader("3. Bill Statement Amounts")

    bill_amounts = {}

    cols = st.columns(3)

    for i in range(1, 7):
        with cols[(i - 1) % 3]:
            bill_amounts[f'BILL_AMT{i}'] = st.number_input(
                f'Bill Amount {i}',
                value=0
            )

    st.divider()

    st.subheader("4. Previous Payments")

    payment_amounts = {}

    cols = st.columns(3)

    for i in range(1, 7):
        with cols[(i - 1) % 3]:
            payment_amounts[f'PAY_AMT{i}'] = st.number_input(
                f'Payment Amount {i}',
                min_value=0,
                value=0
            )

    submit = st.form_submit_button(
        "Predict Default Risk",
        use_container_width=True
    )

# Prediction
if submit:

    input_data = {
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age
    }

    input_data.update(payment_status)
    input_data.update(bill_amounts)
    input_data.update(payment_amounts)

    input_df = pd.DataFrame([input_data])

    # Generate probability
    classes = list(model.classes_)
    positive_index = classes.index('Y')

    st.write("Input data:")
    st.write(input_df)

    st.write("Input data types:")
    st.write(input_df.dtypes)

    st.write("Model probabilities:", model.predict_proba(input_df))

    probability = model.predict_proba(input_df)[0][positive_index]

    # Apply threshold
    prediction = 'Y' if probability >= threshold else 'N'

    
    st.divider()

    st.header("📊 Prediction Dashboard")

    # Risk classification
    if probability < 0.30:
        risk_level = "Low Risk"
        risk_color = "success"

    elif probability < 0.50:
        risk_level = "Medium Risk"
        risk_color = "warning"

    else:
        risk_level = "High Risk"
        risk_color = "error"

    # Prediction metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Classification Threshold",
            f"{threshold * 100:.0f}%"
        )

    with col3:
        st.metric(
            "Risk Level",
            risk_level
        )

    # Probability progress bar
    st.subheader("Default Probability Indicator")

    st.progress(float(probability))

    st.caption(
        f"Model default score: {probability * 100:.2f}%"
    )

    # Final prediction
    st.subheader("Final Prediction")

    if prediction == 'Y':
        st.error(
            "⚠️ Customer is predicted to DEFAULT."
        )
    else:
        st.success(
            "✅ Customer is predicted NOT TO DEFAULT."
        )

    # Customer summary
    st.divider()

    st.subheader("Customer Risk Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Credit Limit",
            "Customer Age",
            "Default Probability",
            "Risk Level",
            "Final Prediction"
        ],
        "Value": [
            f"{limit_bal:,.0f}",
            age,
            f"{probability * 100:.2f}%",
            risk_level,
            "Default" if prediction == 'Y' else "No Default"
        ]
    })

    st.table(summary)

    st.caption(
        "Note: Predictions are model estimates for educational "
        "and demonstration purposes, not actual credit decisions."
    )
