import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- File Paths ---
MODEL_PATH = 'model_credit_risk.pkl'
SCALER_PATH = 'numerical_vars_scaler.pkl'
DATA_PATH = 'cr_loan_clean.csv'

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    """Loads the pre-trained model and scaler from pkl files."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: A required file was not found. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the same directory as this app.")
        return None, None

model, scaler = load_model_and_scaler()

# --- Load Data for Data Insights Page ---
@st.cache_data
def load_data():
    """Loads the clean credit loan data for the insights page."""
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Error: The dataset '{DATA_PATH}' was not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

df_clean = load_data()

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Credit Risk", "Data Insights"])

# --- Home Page ---
if page == "Home":
    st.title("Credit Risk Prediction App")
    st.markdown("""
        Welcome to the Credit Risk Prediction App! This tool helps you understand and predict loan defaults.
        
        Using a machine learning model, this app provides insights into the factors influencing credit risk. You can:
        
        1. **Predict Credit Risk**: Input a potential borrower's details to get a real-time risk assessment.
        2. **Explore Data Insights**: Visualize and analyze the underlying loan data to understand trends and correlations.
        
        This application is designed to be a helpful starting point for understanding and leveraging data science in financial services.
    """)
    st.image("https://placehold.co/800x400/0175B5/FFFFFF?text=Credit+Risk+Prediction")

# --- Prediction Page ---
elif page == "Predict Credit Risk":
    if model and scaler:
        st.title("Predict Credit Risk")
        st.markdown("Enter the borrower's details to predict their loan default risk.")

        # --- Input Widgets ---
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
                person_income = st.number_input("Annual Income", min_value=1000, value=60000, help="e.g., $60,000")
                person_emp_length = st.slider("Employment Length (years)", min_value=0.0, max_value=60.0, value=5.0)
                loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=35000, value=15000)
                loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.0)
                loan_percent_income = st.slider("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.25)
                cb_person_cred_hist_length = st.slider("Credit History Length (years)", min_value=2, max_value=30, value=5)
                cb_person_default_on_file = st.selectbox("Credit Default on File?", ["No", "Yes"])

            with col2:
                person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
                loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
                loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

            submitted = st.form_submit_button("Predict")

        if submitted:
            # --- Data Preprocessing ---
            input_df = pd.DataFrame({
                'person_age': [person_age],
                'person_income': [person_income],
                'person_home_ownership': [person_home_ownership],
                'person_emp_length': [person_emp_length],
                'loan_intent': [loan_intent],
                'loan_grade': [loan_grade],
                'loan_amnt': [loan_amnt],
                'loan_int_rate': [loan_int_rate],
                'loan_percent_income': [loan_percent_income],
                'cb_person_default_on_file': ["Y" if cb_person_default_on_file == "Yes" else "N"],
                'cb_person_cred_hist_length': [cb_person_cred_hist_length]
            })

            # One-hot encode categorical features
            categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
            encoded_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)

            # Ensure all one-hot encoded columns are present, even if not in this single input row
            all_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                        'loan_percent_income', 'cb_person_cred_hist_length',
                        'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
                        'person_home_ownership_OWN', 'person_home_ownership_RENT',
                        'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
                        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                        'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_A',
                        'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E',
                        'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_N',
                        'cb_person_default_on_file_Y']
            
            for col in all_cols:
                if col not in encoded_df.columns:
                    encoded_df[col] = 0

            # Scale numerical features
            numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
            encoded_df[numerical_cols] = scaler.transform(encoded_df[numerical_cols])

            # Reorder columns to match the training data
            encoded_df = encoded_df[all_cols]

            # --- Prediction ---
            prediction = model.predict(encoded_df)
            prediction_proba = model.predict_proba(encoded_df)

            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.success("The model predicts a **low risk** of loan default.")
            else:
                st.error("The model predicts a **high risk** of loan default.")

            st.info(f"Probability of No Default (Class 0): **{prediction_proba[0][0]:.2f}**")
            st.warning(f"Probability of Default (Class 1): **{prediction_proba[0][1]:.2f}**")

# --- Data Insights Page ---
elif page == "Data Insights":
    if not df_clean.empty:
        st.title("Data Insights")
        st.markdown("Explore the key trends and patterns in the loan dataset.")

        # --- Visualizations ---
        st.subheader("Loan Status Distribution")
        fig1 = px.histogram(df_clean, x='loan_status', color='loan_status',
                            title="Distribution of Loan Status (0 = No Default, 1 = Default)",
                            labels={'loan_status': 'Loan Status'})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Loan Intent vs. Loan Status")
        fig2 = px.histogram(df_clean, x='loan_intent', color='loan_status',
                            title="Loan Intent by Default Status",
                            labels={'loan_intent': 'Loan Intent', 'loan_status': 'Loan Status'},
                            barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Loan Grade vs. Loan Status")
        fig3 = px.histogram(df_clean, x='loan_grade', color='loan_status',
                            title="Loan Grade by Default Status",
                            labels={'loan_grade': 'Loan Grade', 'loan_status': 'Loan Status'},
                            barmode='group')
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Loan Amount vs. Interest Rate")
        fig4 = px.scatter(df_clean, x='loan_amnt', y='loan_int_rate', color='loan_status',
                          title="Loan Amount vs. Interest Rate by Default Status",
                          labels={'loan_amnt': 'Loan Amount', 'loan_int_rate': 'Interest Rate', 'loan_status': 'Loan Status'})
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Data insights are not available as the dataset could not be loaded.")
