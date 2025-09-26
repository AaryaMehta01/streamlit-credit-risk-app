import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise Credit Risk Management",
    page_icon="🏢",
    layout="wide"
)

# --- Helper Function for Currency Formatting ---
def format_currency(value):
    """
    Formats a number into a readable currency string with M or B suffix.
    """
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    else:
        return f"${value:,.0f}"

# --- Helper Function for Dashboard Content ---
def render_enterprise_dashboard(df, page_title):
    """
    Renders an enterprise-grade dashboard with advanced analytics.
    """
    # ---------------- Header & Introduction ----------------
    st.title(page_title)
    st.markdown("""
        <style>
            .title-divider {
                border-bottom: 2px solid #E0E0E0;
                margin-top: 0.5rem;
                margin-bottom: 1.5rem;
            }
        </style>
        <div class="title-divider"></div>
        <p style="font-size:1.1rem;">
            Welcome to the Enterprise Credit Risk Dashboard. This tool provides a comprehensive, 
            360-degree view of your loan portfolio, allowing you to manage risk and 
            make data-driven decisions.
        </p>
    """, unsafe_allow_html=True)
    
    df_original = df.copy()

    # ---------------- Sidebar Controls - Logical Sections ----------------
    with st.sidebar:
        st.header("⚙️ Model Parameters")
        threshold = st.slider("Probability Threshold (accept if PD ≤ threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 1.0, 0.05)
        ead_multiplier = st.number_input("EAD Multiplier (scale loan amounts)", value=1.0, step=0.1)

        # Dynamic Filters
        st.header("🗂️ Portfolio Filters")
        
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in ['acceptance_status']]
        numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['prob_default', 'loan_amnt', 'expected_loss', 'accept']]

        filters = {}
        for col in categorical_cols:
            unique_values = df[col].unique()
            selected_values = st.multiselect(f"Filter by {col}", unique_values, unique_values)
            filters[col] = selected_values
        
        for col in numerical_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            selected_range = st.slider(f"Filter by {col}", min_val, max_val, (min_val, max_val))
            filters[col] = selected_range

    # Apply filters
    for col, values in filters.items():
        if col in categorical_cols:
            df = df[df[col].isin(values)]
        elif col in numerical_cols:
            df = df[df[col].between(values[0], values[1])]

    # ---------------- Pre-calculate metrics based on controls ----------------
    df["loan_amnt"] = df["loan_amnt"] * ead_multiplier
    df["expected_loss"] = df["prob_default"] * lgd * df["loan_amnt"]
    df["accept"] = (df["prob_default"] <= threshold).astype(int)
    df["acceptance_status"] = df["accept"].map({1: 'Accepted', 0: 'Rejected'})
    df['true_label'] = df['loan_status'] # Assuming loan_status is the ground truth
    df['prediction'] = df['accept']

    # --- Main Dashboard Content ---
    
    st.markdown("### Portfolio Health & Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        total_loans = len(df)
        st.metric(label="Total Loans", value=f"{total_loans:,.0f}")
    with col2:
        accepted_loans = df["accept"].sum()
        acceptance_rate = accepted_loans / total_loans * 100 if total_loans > 0 else 0
        st.metric(label="Acceptance Rate", value=f"{acceptance_rate:,.1f}%")
    with col3:
        total_loan_amount = df["loan_amnt"].sum()
        st.metric(label="Total Loan Amount", value=format_currency(total_loan_amount))
    with col4:
        total_expected_loss = df["expected_loss"].sum()
        st.metric(label="Total Expected Loss", value=format_currency(total_expected_loss))
    with col5:
        el_accepted = df[df['accept'] == 1]['expected_loss'].sum()
        st.metric(label="EL (Accepted)", value=format_currency(el_accepted))

    st.markdown("---")

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.subheader("Distribution of Probability of Default (PD)")
        fig1 = px.histogram(df, x="prob_default", nbins=50, color='acceptance_status',
                            title="PD Distribution by Acceptance Status")
        fig1.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red", annotation_text=f'Threshold: {threshold:.2f}')
        fig1.update_layout(xaxis_title="Probability of Default", yaxis_title="Number of Loans")
        st.plotly_chart(fig1, use_container_width=True)

    with col_vis2:
        st.subheader("Expected Loss by Acceptance Status")
        loss_by_bucket = df.groupby('acceptance_status')['expected_loss'].sum().reset_index()
        fig2 = px.bar(loss_by_bucket, x='acceptance_status', y='expected_loss', title='Total Expected Loss by Acceptance Status')
        fig2.update_layout(xaxis_title="Acceptance Status", yaxis_title="Total Expected Loss ($)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Portfolio Segmentation")
    
    col_seg1, col_seg2 = st.columns(2)
    with col_seg1:
        segment_by_1 = st.selectbox("Segment Portfolio by:", ['None'] + categorical_cols, key='seg_1')
        if segment_by_1 != 'None':
            fig_seg = px.box(df, x=segment_by_1, y="prob_default", 
                             title=f"Probability of Default Distribution by {segment_by_1}",
                             color=segment_by_1)
            st.plotly_chart(fig_seg, use_container_width=True)
    
    with col_seg2:
        segment_by_2 = st.selectbox("Segment Portfolio by:", ['None'] + categorical_cols, key='seg_2')
        if segment_by_2 != 'None':
            fig_seg = px.box(df, x=segment_by_2, y="loan_amnt", 
                             title=f"Loan Amount Distribution by {segment_by_2}",
                             color=segment_by_2)
            st.plotly_chart(fig_seg, use_container_width=True)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Intro", "Main Dashboard", "Upload Your Data", "About"])

# --- Main Page Content ---
if page == "Intro":
    st.title("Welcome to the Enterprise Credit Risk Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://placehold.co/400x300/E0E0E0/white?text=Enterprise+Analytics", use_column_width=True)
        st.info("Navigate using the sidebar on the left.")

    with col2:
        st.header("What is Enterprise Credit Risk Management?")
        st.write(
            """
            At an enterprise level, credit risk management moves beyond simple analysis. It's about a holistic framework that integrates data, 
            advanced models, and sophisticated decision-making tools to manage a diverse loan portfolio at scale.
            """
        )
        st.header("How This App Helps")
        st.write(
            """
            This application provides a powerful and interactive dashboard to analyze and visualize credit risk data. 
            It is designed to give you an immediate, hands-on experience without the need to upload your own CSV file. 
            This page allows you to test the interactive controls, understand the various charts and KPIs, 
            and see how the analysis works before you begin a custom analysis.
            """
        )
        st.markdown("---")
        st.subheader("Key Features:")
        st.markdown(
            """
            * **Portfolio Management:** A comprehensive overview of your loan portfolio with key financial metrics.
            * **Deep Dive Analysis:** A flexible section for custom visualizations and segmented analysis.
            * **Custom Data Upload:** The ability to upload your own scored loan data for analysis.
            """
        )

elif page == "Main Dashboard":
    try:
        df = pd.read_csv("cr_loan_clean.csv")
        df['prob_default'] = df['loan_status']
        df['loan_amnt'] = df['loan_amnt']
        render_enterprise_dashboard(df.copy(), "Main Dashboard")
    except FileNotFoundError:
        st.error("Sample data file 'cr_loan_clean.csv' not found. Please make sure it's in the same directory as this script.")

elif page == "Upload Your Data":
    st.title("Upload Your Data")
    st.markdown("---")
    st.subheader("📥 Upload Your CSV File")
    st.caption("The dashboard requires a **'loan_amnt'** column and a **'prob_default'** (or binary **'loan_status'**) column. Additional columns will be used for filtering.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Please upload a CSV file to begin the analysis.")
    else:
        try:
            df = pd.read_csv(uploaded)
            st.success("File uploaded successfully! The dashboard is now updated with your data.")
            
            # Smartly map columns
            if 'prob_default' not in df.columns:
                if 'loan_status' in df.columns:
                    df['prob_default'] = df['loan_status']
                else:
                    st.warning("Could not find a 'prob_default' or 'loan_status' column. Please rename a column in your file or select it below.")
                    prob_col = st.selectbox("Select the Probability of Default/Loan Status column", options=[None] + list(df.columns))
                    if prob_col:
                        df['prob_default'] = df[prob_col]
                    else:
                        st.stop()
            
            if 'loan_amnt' not in df.columns:
                st.warning("Could not find a 'loan_amnt' column. Please rename a column or select it below.")
                loan_col = st.selectbox("Select the Loan Amount column", options=[None] + list(df.columns))
                if loan_col:
                    df['loan_amnt'] = df[loan_col]
                else:
                    st.stop()
                    
            if not pd.api.types.is_numeric_dtype(df['prob_default']):
                st.error("The 'prob_default' column must be numeric. Please check your data.")
                st.stop()
            if not pd.api.types.is_numeric_dtype(df['loan_amnt']):
                st.error("The 'loan_amnt' column must be numeric. Please check your data.")
                st.stop()
            
            render_enterprise_dashboard(df.copy(), "Uploaded Data Dashboard")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "About":
    st.title("About this Project")
    st.markdown("---")
    st.write(
        """
        This Streamlit application was developed as an enterprise-grade tool for performing quick and insightful credit risk analysis. 
        It is designed to be a professional, user-friendly, and interactive platform for financial analysts, data scientists, and students 
        to explore key concepts in credit risk management.
        """
    )
    st.header("Behind the Scenes")
    st.markdown(
        """
        The application is built using **Python** and the **Streamlit** library. The visualizations are created with **Plotly**,
        which enables the interactive charts you see on the dashboards. The underlying data manipulation is handled by **Pandas** and **NumPy**.
        """
    )
    st.markdown(
        """
        This project demonstrates the power of these tools in creating powerful, accessible, and enterprise-grade data applications.
        """
    )
    st.markdown("---")
    st.markdown("Thank you for using this application!")
