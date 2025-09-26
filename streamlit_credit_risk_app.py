import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Intro", "Main Dashboard", "Upload Your Data", "About"])

# --- Helper Function for Dashboard Content ---
def render_dashboard(df, page_title):
    """
    Renders the dashboard KPIs, charts, and tables based on the provided dataframe.
    """
    # ---------------- Header ----------------
    st.title(page_title)
    st.markdown("---")

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("🔧 Controls")
        threshold = st.slider("Probability Threshold (accept if PD ≤ threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 1.0, 0.05)
        ead_multiplier = st.number_input("EAD Multiplier (scale loan amounts)", value=1.0, step=0.1)
        
    # Apply controls to the dataframe
    df["loan_amnt"] = df["loan_amnt"] * ead_multiplier
    df["expected_loss"] = df["prob_default"] * lgd * df["loan_amnt"]
    df["accept"] = (df["prob_default"] <= threshold).astype(int)

    # ---------------- KPIs ----------------
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_loans = len(df)
        st.metric(label="Total Loans", value=f"{total_loans:,.0f}")
    with col2:
        accepted_loans = df["accept"].sum()
        acceptance_rate = accepted_loans / total_loans * 100
        st.metric(label="Acceptance Rate", value=f"{acceptance_rate:,.1f}%")
    with col3:
        total_expected_loss = df["expected_loss"].sum()
        st.metric(label="Total Expected Loss", value=f"${total_expected_loss:,.0f}")
    with col4:
        total_accepted_loss = df[df['accept'] == 1]['expected_loss'].sum()
        st.metric(label="Expected Loss (Accepted Loans)", value=f"${total_accepted_loss:,.0f}")

    st.markdown("---")

    # ---------------- Visualizations (Plotly) ----------------
    st.markdown("### 📈 Interactive Data Visualizations")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig1 = px.histogram(df, x="prob_default", nbins=50, title="Distribution of Probability of Default (PD)")
        fig1.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red", annotation_text=f'Threshold: {threshold:.2f}')
        fig1.update_layout(xaxis_title="Probability of Default", yaxis_title="Frequency")
        st.plotly_chart(fig1, use_container_width=True)
    with col_chart2:
        # Corrected code to explicitly create a column for the x-axis
        df['acceptance_status'] = df['accept'].map({1: 'Accepted', 0: 'Rejected'})
        loss_by_bucket = df.groupby('acceptance_status')['expected_loss'].sum().reset_index()
        fig2 = px.bar(loss_by_bucket, x='acceptance_status', y='expected_loss', title='Expected Loss by Acceptance Status')
        fig2.update_layout(xaxis_title="Acceptance Status", yaxis_title="Total Expected Loss ($)")
        st.plotly_chart(fig2, use_container_width=True)

    # --- New Visualizations ---
    st.markdown("---")
    st.markdown("### 📊 Additional Visualizations")
    col_new_charts1, col_new_charts2 = st.columns(2)
    with col_new_charts1:
        # Histogram for loan amount distribution
        fig_loan_amount = px.histogram(df, x="loan_amnt", nbins=50, title="Distribution of Loan Amounts")
        fig_loan_amount.update_layout(xaxis_title="Loan Amount ($)", yaxis_title="Frequency")
        st.plotly_chart(fig_loan_amount, use_container_width=True)
        
    with col_new_charts2:
        # Pie chart for categorical distribution, dynamically selected
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in ['acceptance_status']]
        if categorical_cols:
            selected_cat = st.selectbox("Select a category for the pie chart:", categorical_cols)
            fig_pie = px.pie(
                df,
                names=selected_cat,
                title=f"Distribution of Loans by {selected_cat}",
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No categorical columns found in the dataset for a pie chart.")

    st.markdown("---")

    # ---------------- Strategy Curve ----------------
    st.markdown("### 📉 Strategy Curve")
    st.write("Analyze the trade-off between Acceptance Rate and Expected Loss at different PD thresholds.")
    sweep = np.linspace(0.01, 1.0, 100)
    strategy_data = []
    for t in sweep:
        accepted_df = df[df['prob_default'] <= t]
        acceptance_rate = len(accepted_df) / len(df)
        expected_loss = accepted_df['expected_loss'].sum()
        strategy_data.append({'threshold': t, 'acceptance_rate': acceptance_rate, 'expected_loss': expected_loss})
    strategy_df = pd.DataFrame(strategy_data)
    
    fig_strategy = go.Figure()
    fig_strategy.add_trace(go.Scatter(x=strategy_df['acceptance_rate'], y=strategy_df['expected_loss'], mode='lines', name='Strategy Curve'))
    fig_strategy.update_layout(
        title='Strategy Curve: Acceptance Rate vs. Total Expected Loss',
        xaxis_title='Acceptance Rate',
        yaxis_title='Total Expected Loss ($)',
    )
    fig_strategy.add_annotation(
        x=strategy_df[strategy_df['threshold'] == threshold]['acceptance_rate'].iloc[0],
        y=strategy_df[strategy_df['threshold'] == threshold]['expected_loss'].iloc[0],
        text=f'Current Threshold: {threshold}',
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    st.plotly_chart(fig_strategy, use_container_width=True)

    st.markdown("---")

    # ---------------- Strategy Table ----------------
    st.markdown("#### 📋 Strategy Table (Sweep Thresholds)")
    st.write("This table shows key metrics at various probability thresholds.")
    sweep_table = np.linspace(0.1, 0.95, 18)
    rows = []
    for t in sweep_table:
        acc_rate = (df["prob_default"] <= t).mean()
        acc_df = df[df["prob_default"] <= t]
        bad_rate_acc = (acc_df['prob_default'] > 0.5).mean() if not acc_df.empty else 0.0
        el_total = acc_df["expected_loss"].sum()
        rows.append({"Threshold": t, "Acceptance Rate": acc_rate, "Bad Rate (Accepted)": bad_rate_acc, "Expected Loss": el_total})
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({
        "Threshold": "{:.2f}",
        "Acceptance Rate": "{:.0%}",
        "Bad Rate (Accepted)": "{:.1%}",
        "Expected Loss": "${:,.0f}"
    }))

    st.markdown("---")

    # ---------------- Segmented Analysis ----------------
    st.markdown("### 🔍 Segmented Analysis")
    segment_by = st.selectbox("Select a variable to segment by:", ['None'] + [col for col in df.columns if df[col].dtype == 'object'])
    if segment_by != 'None':
        fig_seg = px.box(df, x=segment_by, y="prob_default", title=f"Probability of Default Distribution by {segment_by}")
        st.plotly_chart(fig_seg, use_container_width=True)
        st.markdown("#### Key Metrics by Segment")
        agg_df = df.groupby(segment_by).agg(
            total_loans=('loan_amnt', 'count'),
            total_loan_amount=('loan_amnt', 'sum'),
            total_expected_loss=('expected_loss', 'sum'),
            avg_expected_loss=('expected_loss', 'mean')
        ).reset_index()
        st.dataframe(agg_df.style.format({
            "total_loan_amount": "${:,.0f}",
            "total_expected_loss": "${:,.0f}",
            "avg_expected_loss": "${:,.2f}"
        }))

# --- Page Content ---
if page == "Intro":
    # --- Intro Page Logic ---
    try:
        image = Image.open('credit_risk_image.png')
    except FileNotFoundError:
        st.warning("Image 'credit_risk_image.png' not found. Using a placeholder instead.")
        image = None

    st.title("Welcome to the Credit Risk Analyzer")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        if image:
            st.image(image, use_column_width=True)
        else:
            st.write("A placeholder for a relevant image would go here.")
        st.info("Navigate using the sidebar on the left.")

    with col2:
        st.header("What is Credit Risk Analysis?")
        st.write(
            """
            Credit risk analysis is the process of determining the probability of a borrower defaulting on their financial obligations. 
            It's a critical function for banks and other financial institutions to manage their portfolios and ensure long-term profitability.
            """
        )
        st.header("How This App Helps")
        st.write(
            """
            This application provides a powerful and interactive dashboard to analyze and visualize credit risk data. 
            You can explore pre-loaded sample data or upload your own dataset to assess loan portfolios, evaluate risk strategies, 
            and gain insights into factors influencing credit defaults.
            """
        )
        st.write(
            """
            ### Key Features:
            * **Test Dashboard:** Explore a pre-loaded, cleaned dataset with various interactive charts.
            * **Upload Your Data:** Analyze your own CSV file by mapping its columns to the required fields.
            * **Interactive Controls:** Adjust key parameters like the Probability of Default (PD) threshold to see how it impacts acceptance rates and expected loss.
            """
        )

    st.markdown("---")

    st.markdown("### Getting Started")
    st.markdown("1. Use the navigation menu on the left to select a page.")
    st.markdown("2. Start with the **Test Dashboard** to familiarize yourself with the features.")
    st.markdown("3. Go to the **Upload Your Data** page to perform a custom analysis.")

elif page == "Main Dashboard":
    # --- Main Dashboard Logic ---
    try:
        df = pd.read_csv("cr_loan_clean.csv")
        df['prob_default'] = df['loan_status']
        df['loan_amnt'] = df['loan_amnt']
        render_dashboard(df.copy(), "Main Dashboard")
    except FileNotFoundError:
        st.error("Sample data file 'cr_loan_clean.csv' not found. Please make sure it's in the same directory as this script.")

elif page == "Upload Your Data":
    # --- Upload Your Data Page Logic ---
    st.title("Upload Your Data")
    st.markdown("---")
    st.subheader("📥 Upload Your CSV File")
    st.caption("The dashboard requires two key columns: one for **Loan Amount** and one for **Default Probability** (or a binary **Loan Status**).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Please upload a CSV file to begin the analysis.")
    else:
        try:
            df = pd.read_csv(uploaded)
            st.success("File uploaded successfully! Now, please select the columns for analysis.")
            st.markdown("---")
            st.markdown("### 🗺️ Map Your Columns")
            col_mapping = st.columns(2)
            with col_mapping[0]:
                prob_col = st.selectbox("Select the 'Probability of Default' or 'Loan Status' column", options=[None] + list(df.columns))
            with col_mapping[1]:
                loan_col = st.selectbox("Select the 'Loan Amount' column", options=[None] + list(df.columns))

            if prob_col is None or loan_col is None:
                st.warning("Please select both a probability/status column and a loan amount column to proceed.")
            else:
                df['prob_default'] = df[prob_col]
                df['loan_amnt'] = df[loan_col]
                is_binary = not df['prob_default'].between(0, 1, inclusive='both').all()
                if is_binary:
                    st.warning("The selected 'Probability of Default' column does not appear to be a probability score. Assuming it is a binary loan status column.")
                render_dashboard(df.copy(), "Uploaded Data Dashboard")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "About":
    # --- About Page Logic ---
    st.title("About this Project")
    st.markdown("---")
    st.write(
        """
        This Streamlit application was developed as a tool for performing quick and insightful credit risk analysis. 
        It is designed to be a user-friendly and interactive platform for financial analysts, data scientists, and students 
        to explore key concepts in credit risk management.
        """
    )
    st.write(
        """
        The application is structured into several sections to provide a clean and logical workflow:
        * **Intro:** A high-level overview of the project and its purpose.
        * **Main Dashboard:** An interactive dashboard pre-loaded with sample data for immediate exploration.
        * **Upload Your Data:** A flexible tool for analyzing your own credit data with smart column mapping.
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
